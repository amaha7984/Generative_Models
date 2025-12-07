# train_latent_rectified.py
#motivation: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

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
        """
        self.decay = decay
        # Deep copy of the model for EMA weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA parameters from `model` parameters.
        """
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



def main(rank, world_size, args):
    local_rank = setup_ddp(rank, world_size)

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    DATA_ROOT = args.data_root
    SAVE_DIR  = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Data Loading (unpaired A/B as before)
    # ---------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),   # [-1,1]
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

    # ---------------------------------------------------------
    # Model Setup (latent-space rectified flow)
    # ---------------------------------------------------------
    # NOTE: in_channels = 4 now (z_t only), NOT [z_t, z_src]
    model = UNetModel(
        in_channels=4,     # latent channels only
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
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    ema = EMA(model.module, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
    eps = 1e-3   # smallest t (like in rectified flow code)

    global_step = 0 
    # ---------------------------------------------------------
    # Training Loop (Rectified Flow in latent space)
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        if rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=(rank != 0))

        for batch in pbar:
            # Unpaired domains as before:
            #   "sat" from trainA  (domain π0, source)
            #   "map" from trainB  (domain π1, target)
            x_src = batch["sat"].to(local_rank, non_blocking=True)  # [-1,1]
            x_tgt = batch["map"].to(local_rank, non_blocking=True)  # [-1,1]

            B = x_src.shape[0]

            # Sample t ~ Uniform(eps, 1)
            t = torch.rand(B, device=local_rank) * (1.0 - eps) + eps
            t_b = t.view(B, 1, 1, 1)  # for broadcasting

            # ------------ VAE Encoding (latent domains π0, π1) ------------
            with torch.no_grad():
                z0 = vae.encode(x_src).latent_dist.sample() * scale_factor  # X0: source latent (sat)
                z1 = vae.encode(x_tgt).latent_dist.sample() * scale_factor  # X1: target latent (map)

                # Linear interpolation: Z_t = (1-t) * X0 + t * X1
                z_t = (1.0 - t_b) * z0 + t_b * z1

                # Target velocity: X1 - X0
                target_vel = (z1 - z0)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred_vel = model(z_t, t, extra={})
                loss = criterion(pred_vel, target_vel)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Increment global step and apply LR warmup before stepping
            global_step += 1
            adjust_lr(optimizer, global_step, base_lr=args.lr, warmup_steps=args.warmup_steps)

            scaler.step(optimizer)
            scaler.update()

            # Update EMA after optimizer step (using unwrapped model)
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
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth")
                    )
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
                    steps=50,
                    batch_size=32,
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

        #lr_sched.step()

    cleanup_ddp()


# ---------------------------------------------------------
# Argparse
# ---------------------------------------------------------
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--no_fid", action="store_true")
    parser.add_argument("--scale_factor", type=float, default=0.18215)
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay for model parameters.")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps for learning rate.")


    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
