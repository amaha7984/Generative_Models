# train_latent_sbcfm_ot.py
# Inspired from paper "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"
# We are using Schrödinger bridge CFM as mentioned in the paper
import os
import argparse
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from datasets import GenericI2IDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNetModel

from ot_coupling_sb import minibatch_ot_sample_plan

from fid_eval_i2i_sbcfm import eval_fid_i2i_sbcfm

from diffusers.models import AutoencoderKL


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        d = self.decay
        for k in esd.keys():
            esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)

    def to(self, device):
        self.ema_model.to(device)
        return self

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def adjust_lr(optimizer, step, base_lr, warmup_steps):
    if warmup_steps <= 0:
        return
    if step >= warmup_steps:
        return
    scale = float(step) / float(max(1, warmup_steps))
    for g in optimizer.param_groups:
        g["lr"] = base_lr * scale


def _append_line(path: str, line: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


# -----------------------------
# GLOBAL OT MATCHING (DDP-safe)
# -----------------------------
@torch.no_grad()
def _gather_cat(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


@torch.no_grad()
def global_minibatch_ot_sample_plan(
    z0_local: torch.Tensor,
    z1_local: torch.Tensor,
    ot_method: str,
    ot_eps: float,
    ot_iters: int,
    ot_replace: bool,
    ot_num_threads: int,
    normalize_cost: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (dist.is_available() and dist.is_initialized()):
        return minibatch_ot_sample_plan(
            z0_local,
            z1_local,
            ot_method=ot_method,
            eps=ot_eps,
            iters=ot_iters,
            replace=ot_replace,
            num_threads=ot_num_threads,
            normalize_cost=normalize_cost,
        )

    rank = dist.get_rank()
    B = z0_local.size(0)

    z0_all = _gather_cat(z0_local)
    z1_all = _gather_cat(z1_local)

    z0_pi_all, z1_pi_all = minibatch_ot_sample_plan(
        z0_all,
        z1_all,
        ot_method=ot_method,
        eps=ot_eps,
        iters=ot_iters,
        replace=ot_replace,
        num_threads=ot_num_threads,
        normalize_cost=normalize_cost,
    )

    start = rank * B
    end = start + B
    return z0_pi_all[start:end], z1_pi_all[start:end]


def main(rank, world_size, args):
    local_rank = setup_ddp(rank, world_size)

    DATA_ROOT = args.data_root
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    rank0 = (rank == 0)

    loss_log_path = os.path.join(SAVE_DIR, "loss_scores_sbcfm.txt")
    best_loss = float("inf")
    best_fid = float("inf")

    if rank0 and (not os.path.exists(loss_log_path)):
        _append_line(loss_log_path, "epoch\tavg_loss")

    # -----------------------------
    # Data Loading (unpaired A/B)
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
    ])

    dataset = GenericI2IDataset(DATA_ROOT, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # -----------------------------
    # Model (latent drift v_theta)
    # -----------------------------
    model = UNetModel(
        in_channels=4,
        model_channels=192,
        out_channels=4,
        num_res_blocks=3,
        attention_resolutions=(2, 4, 8),
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        with_fourier_features=False,
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    ema = EMA(model.module, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # -----------------------------
    # Frozen SD-VAE
    # -----------------------------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(local_rank)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    scale_factor = args.scale_factor

    # -----------------------------
    # SB-CFM parameters
    # -----------------------------
    sb_sigma = float(args.sb_sigma)
    # TorchCFM SB-CFM sets OT sinkhorn reg = 2*sigma^2

    ot_eps = args.ot_eps
    if ot_eps is None:
        ot_eps = 2.0 * (sb_sigma ** 2)

    # Loss & AMP
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        if rank0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}] SB-CFM + OT")

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=(not rank0))

        for batch in pbar:
            x_src = batch["sat"].to(local_rank, non_blocking=True)  # [-1,1]
            x_tgt = batch["map"].to(local_rank, non_blocking=True)  # [-1,1]
            B = x_src.size(0)

            # Sample t ~ Uniform(eps, 1-eps) to avoid division blowups
            t_eps = float(args.t_eps)
            t = torch.rand(B, device=local_rank) * (1.0 - 2.0 * t_eps) + t_eps
            t_b = t.view(B, 1, 1, 1)

            with torch.no_grad():
                # Encode to latents
                z0 = vae.encode(x_src).latent_dist.sample() * scale_factor
                z1 = vae.encode(x_tgt).latent_dist.sample() * scale_factor

                # OT pairing (DDP-safe global)
                z0_pi, z1_pi = global_minibatch_ot_sample_plan(
                    z0_local=z0,
                    z1_local=z1,
                    ot_method=args.ot_method,
                    ot_eps=ot_eps,
                    ot_iters=args.ot_iters,
                    ot_replace=args.ot_replace,
                    ot_num_threads=args.ot_num_threads,
                    normalize_cost=args.ot_normalize_cost,
                )

                # -----------------------------
                # SB-CFM path sampling (matches torchcfm)
                # mu_t = (1-t)z0 + t z1
                # sigma_t = sigma * sqrt(t(1-t))
                # z_t = mu_t + sigma_t * eps
                # -----------------------------
                mu_t = (1.0 - t_b) * z0_pi + t_b * z1_pi
                sigma_t = sb_sigma * torch.sqrt(t_b * (1.0 - t_b))
                eps = torch.randn_like(mu_t)
                z_t = mu_t + sigma_t * eps

                # -----------------------------
                # SB-CFM target conditional flow u_t (matches torchcfm)
                # u_t = ((1 - 2t) / (2t(1-t))) * (z_t - mu_t) + (z1 - z0)
                # -----------------------------
                denom = (2.0 * t_b * (1.0 - t_b) + 1e-8)
                ut = ((1.0 - 2.0 * t_b) / denom) * (z_t - mu_t) + (z1_pi - z0_pi)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(z_t, t, extra={})
                loss = criterion(pred, ut)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            global_step += 1
            adjust_lr(optimizer, global_step, base_lr=args.lr, warmup_steps=args.warmup_steps)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model.module)

            total_loss += loss.item()
            if rank0:
                pbar.set_postfix(loss=loss.item())

        dist.barrier()
        # end of epoch
        loss_sum = torch.tensor([total_loss, len(dataloader)], device=local_rank, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum[0] / loss_sum[1]).item()


        if rank0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.6f}")
            _append_line(loss_log_path, f"{epoch+1}\t{avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, "best_loss.pth"))
                torch.save(ema.state_dict(), os.path.join(SAVE_DIR, "best_loss_ema.pth"))

        dist.barrier()

        # -----------------------------
        # FID eval
        # -----------------------------
        if (not args.no_fid) and ((epoch + 1) % args.eval_interval == 0):
            ema_model = ema.ema_model.to(local_rank)
            fid_val = eval_fid_i2i_sbcfm(
                model=ema_model,
                device=local_rank,
                sat_dir=args.testA,
                map_dir=args.testB,
                out_dir=SAVE_DIR,
                epoch=epoch + 1,
                steps=args.fid_steps,
                batch_size=args.fid_batch_size,
                num_workers=args.num_workers,
                save_samples=args.save_samples,
                scale_factor=scale_factor,
            )

            dist.barrier()

            if rank0:
                print(f"[Epoch {epoch+1}] FID: {fid_val:.6f}")
                if float(fid_val) < best_fid:
                    best_fid = float(fid_val)
                    torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, "best_fid.pth"))
                    torch.save(ema.state_dict(), os.path.join(SAVE_DIR, "best_fid_ema.pth"))

            dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True,
                        help="Path containing trainA/trainB folders (unpaired dataset).")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save checkpoints & results.")

    parser.add_argument("--testA", type=str, required=True,
                        help="Path to testA folder for FID (source domain, e.g., sat).")
    parser.add_argument("--testB", type=str, required=True,
                        help="Path to testB folder for FID (target domain, e.g., map).")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--no_fid", action="store_true")

    parser.add_argument("--scale_factor", type=float, default=0.18215)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=2000)

    # SB-CFM
    parser.add_argument("--sb_sigma", type=float, default=0.2,
                        help="Schrodinger Bridge sigma (controls bridge noise).")
    parser.add_argument("--t_eps", type=float, default=1e-3,
                        help="Sample t in [t_eps, 1-t_eps] for stability.")

    # OT / coupling
    parser.add_argument("--ot_method", type=str, default="exact", choices=["exact", "sinkhorn"],
                        help="OT solver: exact (POT emd, CPU) or sinkhorn (GPU).")
    parser.add_argument("--ot_eps", type=float, default=None,
                        help="Sinkhorn epsilon/reg. If None, uses 2*(sb_sigma^2) (torchcfm default).")
    parser.add_argument("--ot_iters", type=int, default=50)
    parser.add_argument("--ot_replace", action="store_true",
                        help="Sample OT pairs with replacement.")
    parser.add_argument("--ot_num_threads", type=int, default=1)
    parser.add_argument("--ot_normalize_cost", action="store_true",
                        help="Normalize OT cost matrix by max(cost). Usually keep OFF.")

    parser.add_argument("--fid_steps", type=int, default=200,
                        help="RK4 steps for ODE sampling during FID.")
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--save_samples", type=int, default=10)

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size, args)
