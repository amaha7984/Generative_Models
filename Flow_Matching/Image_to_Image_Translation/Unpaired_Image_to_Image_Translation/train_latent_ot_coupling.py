# train_latent_ot_coupling.py
# Inspired from paper "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unet import UNetModel

from ot_coupling import (
    minibatch_pair_sample_plan,
    minibatch_pair_indices_from_cost,
)

from fid_eval_i2i_ot_coupling import eval_fid_i2i_rectified

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


@torch.no_grad()
def _gather_cat(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


class DINOv3FeatureExtractor(nn.Module):
    def __init__(
        self,
        repo_dir,
        weights,
        stats="LVD",
        freeze_backbone=True,
        arch="dinov3_vits16",
    ):
        super().__init__()
        self.backbone = torch.hub.load(
            repo_dir,
            arch,
            source="local",
            weights=weights,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        if stats.upper() == "SAT":
            self.mean, self.std = (0.496, 0.496, 0.496), (0.244, 0.244, 0.244)
        else:
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        self.feat_dim = getattr(self.backbone, "embed_dim", None) or getattr(
            self.backbone, "num_features", None
        )
        assert self.feat_dim is not None, "Cannot infer DINOv3 feature dim."

    def _norm_01(self, x_01):
        mean = x_01.new_tensor(self.mean)[None, :, None, None]
        std = x_01.new_tensor(self.std)[None, :, None, None]
        return (x_01 - mean) / std

    def _embed(self, x_norm):
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x_norm)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
        y = self.backbone(x_norm)
        if torch.is_tensor(y) and y.dim() == 2:
            return y
        raise RuntimeError("Unexpected DINOv3 output shape.")

    @torch.no_grad()
    def forward(self, x_pm1):
        x_01 = (x_pm1.clamp(-1, 1) + 1.0) / 2.0
        x_norm = self._norm_01(x_01)
        feats = self._embed(x_norm)
        return feats  # [B, D]


@torch.no_grad()
def global_minibatch_pair_sample_plan(
    z0_local: torch.Tensor,
    z1_local: torch.Tensor,
    args,
):
    if not (dist.is_available() and dist.is_initialized()):
        z0_pi, z1_pi = minibatch_pair_sample_plan(
            z0_local,
            z1_local,
            pairing=args.pairing,
            ot_method=args.ot_method,
            eps=args.ot_eps,
            iters=args.ot_iters,
            replace=args.ot_replace,
            num_threads=args.ot_num_threads,
            mnn_min_mutual_frac=args.mnn_min_mutual_frac,
            softmax_tau=args.softmax_tau,
        )
        return z0_pi, z1_pi

    rank = dist.get_rank()
    B = z0_local.size(0)

    z0_all = _gather_cat(z0_local)
    z1_all = _gather_cat(z1_local)

    z0_pi_all, z1_pi_all = minibatch_pair_sample_plan(
        z0_all,
        z1_all,
        pairing=args.pairing,
        ot_method=args.ot_method,
        eps=args.ot_eps,
        iters=args.ot_iters,
        replace=args.ot_replace,
        num_threads=args.ot_num_threads,
        mnn_min_mutual_frac=args.mnn_min_mutual_frac,
        softmax_tau=args.softmax_tau,
    )

    start = rank * B
    end = start + B
    return z0_pi_all[start:end], z1_pi_all[start:end]


@torch.no_grad()
def global_pair_pixels_from_dino(
    x_src_local: torch.Tensor,  # (B,3,H,W) in [-1,1]
    x_tgt_local: torch.Tensor,  # (B,3,H,W) in [-1,1]
    dino: nn.Module,            # frozen, returns (B,D)
    args,
):
    if not (dist.is_available() and dist.is_initialized()):
        f0 = dino(x_src_local)
        f1 = dino(x_tgt_local)

        B = f0.shape[0]
        f0f = f0.view(B, -1)
        f1f = f1.view(B, -1)
        cost = ((f0f[:, None, :] - f1f[None, :, :]) ** 2).mean(dim=2)

        i, j = minibatch_pair_indices_from_cost(
            cost,
            pairing=args.pairing,
            ot_method=args.ot_method,
            eps=args.ot_eps,
            iters=args.ot_iters,
            replace=args.ot_replace,
            num_threads=args.ot_num_threads,
            mnn_min_mutual_frac=args.mnn_min_mutual_frac,
            softmax_tau=args.softmax_tau,
        )
        return x_src_local[i], x_tgt_local[j]

    rank = dist.get_rank()
    B_local = x_src_local.size(0)

    f0_local = dino(x_src_local)
    f1_local = dino(x_tgt_local)

    f0_all = _gather_cat(f0_local)
    f1_all = _gather_cat(f1_local)
    x0_all = _gather_cat(x_src_local)
    x1_all = _gather_cat(x_tgt_local)

    G = f0_all.shape[0]
    f0f = f0_all.view(G, -1)
    f1f = f1_all.view(G, -1)
    cost = ((f0f[:, None, :] - f1f[None, :, :]) ** 2).mean(dim=2)

    i, j = minibatch_pair_indices_from_cost(
        cost,
        pairing=args.pairing,
        ot_method=args.ot_method,
        eps=args.ot_eps,
        iters=args.ot_iters,
        replace=args.ot_replace,
        num_threads=args.ot_num_threads,
        mnn_min_mutual_frac=args.mnn_min_mutual_frac,
        softmax_tau=args.softmax_tau,
    )

    x0_pi_all = x0_all[i]
    x1_pi_all = x1_all[j]

    start = rank * B_local
    end = start + B_local
    return x0_pi_all[start:end], x1_pi_all[start:end]


def main(rank, world_size, args):
    local_rank = setup_ddp(rank, world_size)

    DATA_ROOT = args.data_root
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    rank0 = (rank == 0)

    loss_log_path = os.path.join(SAVE_DIR, "loss_scores_rectified.txt")
    best_loss = float("inf")
    best_fid = float("inf")

    if rank0 and (not os.path.exists(loss_log_path)):
        _append_line(loss_log_path, "epoch\tavg_loss")

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

    if args.mode == "latent":
        in_ch, out_ch = 4, 4
        model_channels = 192
        num_res_blocks = 3
        attention_resolutions = (2, 4, 8)
        channel_mult = (1, 2, 3, 4)
        num_heads = 4
    else:
        in_ch, out_ch = 3, 3
        model_channels = 128
        num_res_blocks = 2
        attention_resolutions = (4, 8)
        channel_mult = (1, 2, 4)
        num_heads = 2

    model = UNetModel(
        in_channels=in_ch,
        model_channels=model_channels,
        out_channels=out_ch,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=0.1,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        num_heads=num_heads,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        with_fourier_features=False,
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    ema = EMA(model.module, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    vae = None
    dino = None

    if args.mode == "latent":
        vae = AutoencoderKL.from_pretrained(args.vae_id).to(local_rank)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
    elif args.mode == "dino":
        dino = DINOv3FeatureExtractor(
            repo_dir=args.dino_repo_dir,
            weights=args.dino_weights,
            stats=args.dino_stats,
            freeze_backbone=True,
            arch=args.dino_arch,
        ).to(local_rank)
        dino.eval()

    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        if rank0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}] (mode={args.mode}, pairing={args.pairing})")

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=(not rank0))

        for batch in pbar:
            x_src = batch["sat"].to(local_rank, non_blocking=True)  # [-1,1]
            x_tgt = batch["map"].to(local_rank, non_blocking=True)  # [-1,1]
            B = x_src.size(0)

            t = torch.rand(B, device=local_rank)
            t_b = t.view(B, 1, 1, 1)

            with torch.no_grad():
                if args.mode == "latent":
                    z0 = vae.encode(x_src).latent_dist.sample() * args.scale_factor
                    z1 = vae.encode(x_tgt).latent_dist.sample() * args.scale_factor

                    z0_pi, z1_pi = global_minibatch_pair_sample_plan(
                        z0_local=z0,
                        z1_local=z1,
                        args=args,
                    )

                    z_t = (1.0 - t_b) * z0_pi + t_b * z1_pi
                    target_vel = (z1_pi - z0_pi)
                    inp = z_t

                elif args.mode == "dino":
                    x0_pi, x1_pi = global_pair_pixels_from_dino(
                        x_src_local=x_src,
                        x_tgt_local=x_tgt,
                        dino=dino,
                        args=args,
                    )
                    x_t = (1.0 - t_b) * x0_pi + t_b * x1_pi
                    target_vel = (x1_pi - x0_pi)
                    inp = x_t

                else:  # pixel
                    x0_pi, x1_pi = global_minibatch_pair_sample_plan(
                        z0_local=x_src,
                        z1_local=x_tgt,
                        args=args,
                    )
                    x_t = (1.0 - t_b) * x0_pi + t_b * x1_pi
                    target_vel = (x1_pi - x0_pi)
                    inp = x_t

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred_vel = model(inp, t, extra={})
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
            if rank0:
                pbar.set_postfix(loss=loss.item())

        dist.barrier()
        avg_loss = total_loss / len(dataloader)

        if rank0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
            _append_line(loss_log_path, f"{epoch+1}\t{avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, "best_loss.pth"))
                torch.save(ema.state_dict(), os.path.join(SAVE_DIR, "best_loss_ema.pth"))

        dist.barrier()

        if (not args.no_fid) and ((epoch + 1) % args.eval_interval == 0):
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
                mode=args.mode,
                scale_factor=args.scale_factor,
                vae_id=args.vae_id,
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

    parser.add_argument(
        "--mode",
        type=str,
        default="latent",
        choices=["latent", "dino", "pixel"],
        help="latent: SD-VAE latent rectified flow; dino: pixel rectified flow with pairing in DINOv3 feature space; pixel: pixel rectified flow with pairing in pixel space.",
    )

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

    parser.add_argument("--vae_id", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="HuggingFace id for SD-VAE (used in --mode latent).")
    parser.add_argument("--scale_factor", type=float, default=0.18215)

    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay for model parameters.")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps for learning rate.")

    parser.add_argument("--pairing", type=str, default="ot",
                        choices=["ot", "mnn", "hungarian", "softmax"],
                        help="Batch pairing strategy for unpaired coupling.")
    parser.add_argument("--softmax_tau", type=float, default=0.1,
                        help="Temperature for --pairing softmax (lower = sharper).")
    parser.add_argument("--mnn_min_mutual_frac", type=float, default=0.25,
                        help="MNN: if mutual matches >= this fraction of batch, refresh non-mutual pairs from mutual pool.")
    parser.add_argument("--ot_method", type=str, default="sinkhorn", choices=["exact", "sinkhorn"],
                        help="OT solver: exact (POT emd, CPU) or sinkhorn (entropic, GPU).")
    parser.add_argument("--ot_eps", type=float, default=0.05,
                        help="Entropic regularization epsilon for Sinkhorn (ignored for exact).")
    parser.add_argument("--ot_iters", type=int, default=50,
                        help="Sinkhorn iterations (ignored for exact).")
    parser.add_argument("--ot_replace", action="store_true",
                        help="Sample OT pairs with replacement (closest to official sample_plan).")
    parser.add_argument("--ot_num_threads", type=int, default=1,
                        help="Threads for exact OT (POT emd).")

    parser.add_argument("--dino_repo_dir", type=str,
                        default="path/to/github_repository/dinov3",
                        help="Local repo dir for torch.hub.load of DINOv3.")
    parser.add_argument("--dino_weights", type=str,
                        default="path/to/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                        help="Path or URL to DINOv3 weights.")
    parser.add_argument("--dino_stats", type=str, default="LVD",
                        help="Which normalization stats to use: 'LVD' or 'SAT'.")
    parser.add_argument("--dino_arch", type=str, default="dinov3_vits16",
                        help="torch.hub entry name (default: dinov3_vits16).")

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
