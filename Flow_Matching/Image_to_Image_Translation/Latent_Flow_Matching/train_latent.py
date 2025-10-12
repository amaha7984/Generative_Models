# unetflow_sat2map_updated_match_original_new.py (latent-space version)
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

from torchvision.utils import save_image  # (unused here)
import random
from fid_eval_i2i import eval_fid_i2i

# NEW: SD VAE for latent space
from diffusers.models import AutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_ddp(rank, world_size):
    """Initialize distributed process group (NCCL) and set device."""
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Destroy distributed process group cleanly."""
    dist.destroy_process_group()


def main(rank, world_size, args):
    local_rank = setup_ddp(rank, world_size)

    # -------- Paths --------
    DATA_ROOT = "/path/google_mapsdataset"
    SAVE_DIR = "/path/condpath_googlemap_400"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------- Data Loading --------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1] expected by SD-VAE
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

    # -------- Model Setup (latent-space I/O) --------
    # Latent z has 4 channels; we concat z_t (4) with z_src (4) -> in_channels = 8
    # We predict latent velocity u (4) -> out_channels = 4
    model = UNetModel(
        in_channels=8,
        model_channels=96,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=48,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=True,
        with_fourier_features=False,
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -------- VAE (frozen) --------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(local_rank)
    vae.eval()
    vae.train = False
    for p in vae.parameters():
        p.requires_grad = False
    scale_factor = args.scale_factor  # 0.18215 default

    # -------- Loss & AMP --------
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    eps = 1e-5  # tiny floor for noise mix as in the latent FM reference

    # -------- Training Loop --------
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=rank != 0)

        for batch in pbar:
            # Target (map) and source (sat) in [-1,1]
            x1 = batch["map"].to(local_rank, non_blocking=True)  # target
            x0 = batch["sat"].to(local_rank, non_blocking=True)  # source (conditioning)

            B = x0.shape[0]
            t = torch.rand(B, device=local_rank)  # (B,)
            t_b = t.view(B, 1, 1, 1)              # broadcast for mixing

            # Encode to latent space (fp32; do NOT autocast VAE)
            with torch.no_grad():
                z0 = vae.encode(x1).latent_dist.sample() * scale_factor   # (B,4,32,32)
                z_src = vae.encode(x0).latent_dist.sample() * scale_factor # (B,4,32,32)
                z1 = torch.randn_like(z0)

                # Linear path with epsilon floor on noise mix
                z_t = (1.0 - t_b) * z0 + (eps + (1.0 - eps) * t_b) * z1
                u   = (1.0 - eps) * z1 - z0

            x_input = torch.cat([z_t, z_src], dim=1)  # (B,8,32,32)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(x_input, t, extra={})    # (B,4,32,32)
                loss = criterion(pred, u)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix(loss=loss.item())

        # -------- End of epoch --------
        torch.cuda.synchronize()
        dist.barrier()  # ALL ranks sync

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # -------- Eval / Save --------
        if (epoch + 1) % args.eval_interval == 0:
            if args.no_fid:
                if rank == 0:
                    ckpt_path = os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth")
                    torch.save(model.module.state_dict(), ckpt_path)
                    print(f"[Epoch {epoch+1}] Saved checkpoint (no FID).")
                dist.barrier()
            else:
                sat_dir = "/path/testA_5000"
                map_dir = "/path/testB_5000"

                # FID will instantiate its own frozen VAE and decode for metrics
                fid_val = eval_fid_i2i(
                    model.module, device=local_rank,
                    sat_dir=sat_dir, map_dir=map_dir,
                    out_dir=SAVE_DIR, epoch=epoch+1,
                    steps=50, batch_size=32, num_workers=args.num_workers,
                    save_samples=10, scale_factor=scale_factor
                )

                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                if rank == 0:
                    print(f"[Epoch {epoch+1}] FID: {fid_val:.6f}")
                    ckpt_path = os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth")
                    torch.save(model.module.state_dict(), ckpt_path)

                    if not hasattr(main, "_best_fid"):
                        main._best_fid = float("inf")
                    if fid_val < main._best_fid:
                        main._best_fid = fid_val
                        best_path = os.path.join(SAVE_DIR, "model_best_fid.pth")
                        torch.save(model.module.state_dict(), best_path)
                        print(f"[Epoch {epoch+1}] New best FID ({fid_val:.6f}). Saved: {best_path}")

                dist.barrier()

        lr_sched.step()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate and save 10 samples every N epochs")
    parser.add_argument("--no_fid", action="store_true",
                        help="Skip FID evaluation during training (still saves checkpoints).")

    # NEW: scale factor for SD VAE latents
    parser.add_argument("--scale_factor", type=float, default=0.18215)

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
