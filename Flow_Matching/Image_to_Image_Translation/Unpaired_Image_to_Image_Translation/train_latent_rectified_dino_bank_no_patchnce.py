# Rectified Flow + DINOv3 NN pairing (bank) 
# This version removes DINO loss 

import os
import time
import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from datasets import GenericI2IDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNetModel

from fid_eval_i2i_rectified import eval_fid_i2i_rectified

# Frozen SD-VAE (encode/decode)
from diffusers.models import AutoencoderKL


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------
# DDP
# ---------------------------------------------------------
def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# ---------------------------------------------------------
# EMA (same as yours)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# DINOv3 feature extractor
# ---------------------------------------------------------
class DINOv3FeatureExtractor(nn.Module):
    def __init__(self, repo_dir, weights, stats="LVD", freeze_backbone=True):
        super().__init__()
        self.backbone = torch.hub.load(
            repo_dir,
            "dinov3_vits16",
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

    def _norm_01(self, x_01):
        mean = x_01.new_tensor(self.mean)[None, :, None, None]
        std = x_01.new_tensor(self.std)[None, :, None, None]
        return (x_01 - mean) / std

    def _prep(self, x_pm1):
        x_01 = (x_pm1.clamp(-1, 1) + 1.0) / 2.0
        return self._norm_01(x_01)

    @torch.no_grad()
    def forward_cls(self, x_pm1):
        x = self._prep(x_pm1)
        out = self.backbone.forward_features(x)
        if not (isinstance(out, dict) and "x_norm_clstoken" in out):
            raise RuntimeError("DINO forward_features missing x_norm_clstoken")
        return out["x_norm_clstoken"]  # [B, D]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
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
    # Model Setup (latent-space rectified flow)
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
    # Frozen SD-VAE
    # ---------------------------------------------------------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(local_rank)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    scale_factor = args.scale_factor

    # ---------------------------------------------------------
    # DINOv3 (frozen) - used ONLY for NN pairing
    # ---------------------------------------------------------
    dino = DINOv3FeatureExtractor(
        repo_dir=args.dino_repo_dir,
        weights=args.dino_weights,
        stats=args.dino_stats,
        freeze_backbone=True,
    ).to(local_rank)
    dino.eval()

    # ---------------------------------------------------------
    # Load Map Bank (CLS feats + latents)
    # ---------------------------------------------------------
    if rank == 0:
        print(f"[Bank] Loading: {args.map_bank}")

    bank = torch.load(args.map_bank, map_location="cpu")
    map_feats = bank["feats"]      # [N, D]
    map_latents = bank["latents"]  # [N, 4, 32, 32]

    # Move to GPU for fast NN search
    map_feats = map_feats.to(local_rank, non_blocking=True)
    map_latents = map_latents.to(local_rank, non_blocking=True)

    map_feats = F.normalize(map_feats.float(), dim=1)

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
            # Source images only; target is selected via DINO NN bank
            x_src = batch["sat"].to(local_rank, non_blocking=True)  # [-1,1]
            B = x_src.shape[0]

            # Sample t ~ Uniform(eps, 1)
            t = torch.rand(B, device=local_rank) * (1.0 - eps) + eps
            t_b = t.view(B, 1, 1, 1)

            # ----------------------------
            # Build pseudo-paired target latent via DINO NN
            # ----------------------------
            with torch.no_grad():
                # source embedding (CLS)
                f_src = dino.forward_cls(x_src)                  # [B, D]
                f_src = F.normalize(f_src.float(), dim=1).to(f_src.dtype)

                # cosine sim = dot product of normalized feats
                sim = f_src @ map_feats.t()                      # [B, N]
                nn_idx = sim.argmax(dim=1)                       # [B]

                # fetch target latent from bank
                z_tgt = map_latents[nn_idx]                      # [B,4,32,32]

                # encode source latent
                z_src = vae.encode(x_src).latent_dist.sample() * scale_factor

                # rectified path
                z_t = (1.0 - t_b) * z_src + t_b * z_tgt
                target_vel = (z_tgt - z_src)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                pred_vel = model(z_t, t, extra={})
                loss = criterion(pred_vel, target_vel)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # warmup
            global_step += 1
            adjust_lr(optimizer, global_step, base_lr=args.lr, warmup_steps=args.warmup_steps)

            scaler.step(optimizer)
            scaler.update()

            # EMA update
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
                        os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth"),
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
                    steps=args.fid_steps,
                    batch_size=args.fid_batch_size,
                    num_workers=args.num_workers,
                    save_samples=args.save_samples,
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
                        help="Path to testA folder for FID (source).")
    parser.add_argument("--testB", type=str, required=True,
                        help="Path to testB folder for FID (target).")

    # Map bank (CLS feats + latents)
    parser.add_argument("--map_bank", type=str, required=True,
                        help="Path to saved map bank .pt (feats + latents).")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--no_fid", action="store_true")

    parser.add_argument("--scale_factor", type=float, default=0.18215)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=2000)

    # DINO (still required for NN pairing)
    parser.add_argument("--dino_repo_dir", type=str,
                        default="/path/to/repository/dinov3",
                        help="Local repo dir for torch.hub.load of DINOv3.")
    parser.add_argument("--dino_weights", type=str,
                        default="/path/to/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                        help="Path or URL to DINOv3 weights.")
    parser.add_argument("--dino_stats", type=str, default="LVD")

    parser.add_argument("--fid_steps", type=int, default=200)
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--save_samples", type=int, default=10)

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
