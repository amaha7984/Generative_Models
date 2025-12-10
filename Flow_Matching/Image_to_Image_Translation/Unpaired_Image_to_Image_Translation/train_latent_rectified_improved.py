# train_latent_rectified_improved.py
# Rectified Flow in latent space for unpaired I2I (sat -> map)
# with U-shaped timestep sampling and LPIPS-Huber loss,
# inspired by "Improving the Training of Rectified Flows".

import os
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from datasets import GenericI2IDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNetModel

from fid_eval_i2i_rectified_improved import eval_fid_i2i_rectified_rfpp

from diffusers.models import AutoencoderKL
from piq import LPIPS 

import copy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ------------------------ DDP utils ------------------------ #

def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# ------------------------ EMA wrapper ------------------------ #

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
        g['lr'] = base_lr * scale


# ------------------------ t sampling (U-shaped) ------------------------ #

def sample_t_u_shape(batch_size, device, t_min=0.02, t_max=0.98, alpha=0.5):
    """
    Sample timesteps t in (0,1) with a U-shaped distribution.
    We use a Beta(alpha, alpha) with alpha < 1 (U-shaped),
    then clip to [t_min, t_max].

    alpha=0.5 gives strong U-shape; alpha=0.7 is milder.
    """
    dist_beta = torch.distributions.Beta(alpha, alpha)
    t = dist_beta.sample((batch_size,)).to(device)
    t = t.clamp(min=t_min, max=t_max)
    return t


def sample_t_uniform(batch_size, device, t_min=0.02, t_max=0.98):
    t = torch.rand(batch_size, device=device) * (t_max - t_min) + t_min
    return t


# ------------------------ Main training ------------------------ #

def main(rank, world_size, args):
    local_rank = setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    # ------------------ Paths & Data ------------------ #
    DATA_ROOT = args.data_root
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

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
        drop_last=True
    )

    # ------------------ Model (latent rectified flow) ------------------ #
    # Latent from SD-VAE has 4 channels
    model = UNetModel(
        in_channels=4,     # z_t only
        model_channels=192,
        out_channels=4,    # predict velocity in latent space
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
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    ema = EMA(model.module, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ------------------ Frozen SD-VAE ------------------ #
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    scale_factor = args.scale_factor

    # ------------------ Losses (LPIPS-Huber etc.) ------------------ #
    # We'll support: mse, huber, lpips, lpips-huber
    lpips_loss = LPIPS(replace_pooling=True, reduction="none").to(device)

    def huber_loss(pred, target):
        # pred, target: (B,4,H,W) latent velocity
        data_dim = pred.shape[1] * pred.shape[2] * pred.shape[3]
        huber_c = 0.00054 * data_dim
        diff2 = torch.sum((pred - target) ** 2, dim=(1, 2, 3))  # (B,)
        loss = torch.sqrt(diff2 + huber_c ** 2) - huber_c
        return loss / data_dim  # (B,)

    def mse_loss(pred, target):
        return torch.mean((pred - target) ** 2, dim=(1, 2, 3))  # (B,)

    def lpips_image_loss(x_hat_pm1, x_src_pm1):
        """
        x_hat_pm1, x_src_pm1: [B,3,H,W] in [-1,1]
        We'll upsample to 224x224 and convert to [0,1] inside LPIPS.
        """
        x_hat = F.interpolate(x_hat_pm1, size=224, mode="bilinear", align_corners=False)
        x_src = F.interpolate(x_src_pm1, size=224, mode="bilinear", align_corners=False)
        # LPIPS expects [0,1] or [-1,1] depending on implementation; here we follow your repo:
        # x * 0.5 + 0.5 -> [0,1]
        return lpips_loss(x_hat * 0.5 + 0.5, x_src * 0.5 + 0.5).view(x_hat.shape[0])

    # ------------------ Training loop ------------------ #
    scaler = torch.cuda.amp.GradScaler()
    eps = 1e-3  # for safety, but we also clamp t

    global_step = 0

    if rank == 0:
        print(f"Using loss_type={args.loss_type}, lpips_divt={args.lpips_divt}, t_dist={args.t_dist}")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        if rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=(rank != 0))

        for batch in pbar:
            x_src = batch["sat"].to(device, non_blocking=True)  # [-1,1]
            x_tgt = batch["map"].to(device, non_blocking=True)  # [-1,1]
            B = x_src.shape[0]

            # Sample t with U-shaped distribution (or uniform)
            if args.t_dist == "u_shape":
                t = sample_t_u_shape(
                    batch_size=B,
                    device=device,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    alpha=args.t_alpha
                )
            else:
                t = sample_t_uniform(
                    batch_size=B,
                    device=device,
                    t_min=args.t_min,
                    t_max=args.t_max
                )

            t_b = t.view(B, 1, 1, 1)

            # ----------- VAE encode to latent domains π0, π1 ----------- #
            with torch.no_grad():
                z0 = vae.encode(x_src).latent_dist.sample() * scale_factor  # source latent
                z1 = vae.encode(x_tgt).latent_dist.sample() * scale_factor  # target latent

                # Linear interpolation: Z_t = (1-t) * X0 + t * X1
                z_t = (1.0 - t_b) * z0 + t_b * z1

                # Target velocity: X1 - X0
                target_vel = (z1 - z0)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                pred_vel = model(z_t, t, extra={})  # (B,4,H,W)

                # Reconstruct z0 (source latent) from predicted velocity:
                # z0_hat = z_t - t * v_\theta(z_t,t)
                z0_hat = z_t - t_b * pred_vel

                # Decode predicted source image
                x0_hat_pm1 = vae.decode(z0_hat / scale_factor).sample  # [-1,1]

                # Compute per-sample loss components
                if args.loss_type == "mse":
                    loss_vec = mse_loss(pred_vel, target_vel)  # (B,)
                elif args.loss_type == "huber":
                    loss_vec = huber_loss(pred_vel, target_vel)  # (B,)
                elif args.loss_type == "lpips":
                    lp = lpips_image_loss(x0_hat_pm1, x_src)
                    if args.lpips_divt:
                        loss_vec = lp / t
                    else:
                        loss_vec = lp
                elif args.loss_type == "lpips-huber":
                    hub = huber_loss(pred_vel, target_vel)      # (B,)
                    lp = lpips_image_loss(x0_hat_pm1, x_src)    # (B,)
                    if args.lpips_divt:
                        loss_vec = (1.0 - t) * hub + lp / t
                    else:
                        loss_vec = (1.0 - t) * hub + lp
                else:
                    raise ValueError(f"Unknown loss_type {args.loss_type}")

                loss = loss_vec.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            global_step += 1
            adjust_lr(optimizer, global_step, base_lr=args.lr, warmup_steps=args.warmup_steps)

            scaler.step(optimizer)
            scaler.update()

            # Update EMA after optimizer step
            ema.update(model.module)

            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix(loss=loss.item())

        dist.barrier()
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # --------------- Eval / Save --------------- #
        if (epoch + 1) % args.eval_interval == 0:
            if args.no_fid:
                if rank == 0:
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth")
                    )
                dist.barrier()
            else:
                ema_model = ema.ema_model.to(device)
                fid_val = eval_fid_i2i_rectified_rfpp(
                    ema_model,
                    device=device,
                    sat_dir=args.testA,
                    map_dir=args.testB,
                    out_dir=SAVE_DIR,
                    epoch=epoch + 1,
                    steps=args.fid_steps,
                    batch_size=args.fid_batch,
                    num_workers=args.num_workers,
                    save_samples=10,
                    scale_factor=scale_factor,
                )

                dist.barrier()

                if rank == 0:
                    print(f"[Epoch {epoch+1}] FID: {fid_val:.6f}")
                    torch.save(model.module.state_dict(),
                               os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth"))
                    torch.save(ema.state_dict(),
                               os.path.join(SAVE_DIR, f"model_epoch{epoch+1}_ema.pth"))

    cleanup_ddp()


# ------------------------ Argparse ------------------------ #

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
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay for model parameters.")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps for learning rate.")

    # t-distribution & loss
    parser.add_argument("--t_dist", type=str, default="u_shape",
                        choices=["uniform", "u_shape"],
                        help="t sampling distribution")
    parser.add_argument("--t_min", type=float, default=0.02,
                        help="Minimum t (for clipping)")
    parser.add_argument("--t_max", type=float, default=0.98,
                        help="Maximum t (for clipping)")
    parser.add_argument("--t_alpha", type=float, default=0.5,
                        help="Beta(alpha,alpha) for u_shape (alpha<1 => U-shaped)")
    parser.add_argument("--loss_type", type=str, default="lpips-huber",
                        choices=["mse", "huber", "lpips", "lpips-huber"])
    parser.add_argument("--lpips_divt", action="store_true",
                        help="Divide LPIPS term by t (LPIPS-Huber-1/t)")

    # FID sampling
    parser.add_argument("--fid_steps", type=int, default=200,
                        help="Number of RK4 steps for rectified flow during FID")
    parser.add_argument("--fid_batch", type=int, default=32,
                        help="FID batch size")

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
