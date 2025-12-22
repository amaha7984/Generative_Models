# train_latent_ot_coupling.py
# Inspired from paper "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"

import os
import time
import argparse
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
from ot_coupling import minibatch_ot_pairing

# Rectified-flow FID evaluator
from fid_eval_i2i_rectified import eval_fid_i2i_rectified

# Frozen SD-VAE
from diffusers.models import AutoencoderKL
import copy


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
        """
        Exponential Moving Average of model parameters.
        `model` should be the *unwrapped* model (not DDP).
        """
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
    """
    Linear LR warmup: from 0 -> base_lr over `warmup_steps` steps.
    After warmup, LR stays at base_lr.
    """
    if warmup_steps <= 0:
        return
    if step >= warmup_steps:
        return

    scale = float(step) / float(max(1, warmup_steps))
    for g in optimizer.param_groups:
        g["lr"] = base_lr * scale


# -----------------------------
# GLOBAL OT MATCHING (DDP-safe)
# -----------------------------
@torch.no_grad()
def _gather_cat(x: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor from all ranks and concatenate along dim=0."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


@torch.no_grad()
def global_minibatch_ot_match(
    z0_local: torch.Tensor,
    z1_local: torch.Tensor,
    ot_eps: float,
    ot_iters: int,
    ot_pairing: str,
) -> torch.Tensor:
    """
    Compute OT coupling over the *global* batch (across all ranks),
    then return the matched z1 for the *local* z0 slice.
    """
    # Single-process fallback
    if not (dist.is_available() and dist.is_initialized()):
        perm = minibatch_ot_pairing(z0_local, z1_local, eps=ot_eps, iters=ot_iters, method=ot_pairing)
        return z1_local[perm]

    rank = dist.get_rank()
    B = z0_local.size(0)

    z0_all = _gather_cat(z0_local)
    z1_all = _gather_cat(z1_local)

    # OT pairing on global batch
    perm_all = minibatch_ot_pairing(z0_all, z1_all, eps=ot_eps, iters=ot_iters, method=ot_pairing)

    # slice the permutation corresponding to this rank's local block
    start = rank * B
    end = start + B
    perm_local = perm_all[start:end]  # indices into z1_all

    z1m_local = z1_all[perm_local]
    return z1m_local


def main(rank, world_size, args):
    local_rank = setup_ddp(rank, world_size)

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    DATA_ROOT = args.data_root
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Data Loading (unpaired A/B)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Model Setup (latent-space rectified / flow matching)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Frozen SD-VAE (encode/decode)
    # ---------------------------------------------------------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(local_rank)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    scale_factor = args.scale_factor

    # ---------------------------------------------------------
    # Loss & AMP
    # ---------------------------------------------------------
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    eps = 1e-3

    global_step = 0

    # ---------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        if rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=(rank != 0))

        for batch in pbar:
            x_src = batch["sat"].to(local_rank, non_blocking=True)  # [-1,1]
            x_tgt = batch["map"].to(local_rank, non_blocking=True)  # [-1,1]
            B = x_src.size(0)

            # Sample t ~ Uniform(eps, 1)
            t = torch.rand(B, device=local_rank) * (1.0 - eps) + eps
            t_b = t.view(B, 1, 1, 1)

            # ------------ VAE Encoding (π0, π1) + GLOBAL OT coupling ------------
            with torch.no_grad():
                z0 = vae.encode(x_src).latent_dist.sample() * scale_factor
                z1 = vae.encode(x_tgt).latent_dist.sample() * scale_factor

                # GLOBAL minibatch OT matching (DDP-safe)
                z1m = global_minibatch_ot_match(
                    z0_local=z0,
                    z1_local=z1,
                    ot_eps=args.ot_eps,
                    ot_iters=args.ot_iters,
                    ot_pairing=args.ot_pairing,  # "argmax" or "sample"
                )

                # Deterministic ODE-style linear path
                z_t = (1.0 - t_b) * z0 + t_b * z1m
                target_vel = (z1m - z0)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred_vel = model(z_t, t, extra={})
                loss = criterion(pred_vel, target_vel)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            global_step += 1
            adjust_lr(optimizer, global_step, base_lr=args.lr, warmup_steps=args.warmup_steps)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model.module)

            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix(loss=loss.item())

        # End epoch
        dist.barrier()
        avg_loss = total_loss / len(dataloader)

        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # ---------------------------------------------------------
        # Eval / Save
        # ---------------------------------------------------------
        if (epoch + 1) % args.eval_interval == 0:
            if args.no_fid:
                if rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth"))
                dist.barrier()
            else:
                ema_model = ema.ema_model.to(local_rank)
                fid_val = eval_fid_i2i_rectified(
                    ema_model,
                    device=local_rank,
                    sat_dir=args.testA,
                    map_dir=args.testB,
                    out_dir=SAVE_DIR,
                    epoch=epoch + 1,
                    steps=200,
                    batch_size=32,
                    num_workers=args.num_workers,
                    save_samples=10,
                    scale_factor=scale_factor,
                )

                dist.barrier()

                if rank == 0:
                    print(f"[Epoch {epoch+1}] FID: {fid_val:.6f}")
                    torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth"))
                    torch.save(ema.state_dict(), os.path.join(SAVE_DIR, f"model_epoch{epoch+1}_ema.pth"))

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
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay for model parameters.")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps for learning rate.")

    parser.add_argument("--ot_eps", type=float, default=0.05,
                        help="Entropic regularization epsilon for minibatch OT (Sinkhorn).")
    parser.add_argument("--ot_iters", type=int, default=50,
                        help="Sinkhorn iterations for minibatch OT.")
    parser.add_argument("--ot_pairing", type=str, default="argmax", choices=["argmax", "sample"],
                        help="How to convert OT plan into discrete pairing.")

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)