# sat2map_multisample.py
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
from flow_matching.path.scheduler.scheduler import CondOTScheduler
from flow_matching.path.affine import AffineProbPath

from torchvision.utils import save_image
import random
from fid_eval_i2i import eval_fid_i2i

# ---- BatchOT (multisample) sampler ----
# This file must be created (you already planned this) by copying OTPlanSampler
# from the official repo. See msfm_optimal_transport.py you created.
from msfm_optimal_transport import OTPlanSampler


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
    DATA_ROOT = "/aul/homes/amaha038/Mapsgeneration/TerraFlySat_and_MapDatatset/TerraFly_Full_Map&Satellite_Dataset/Final_Sat_Map_Dataset"
    SAVE_DIR = "/aul/homes/amaha038/Generation/Generative_Models/Flow_Matching/Flow_Matching_Complete/I2I_Sat2Map_Multisample/weights/tests"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------- Data Loading --------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
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

    # -------- Model Setup --------
    model = UNetModel(
        in_channels=6,
        model_channels=96,
        out_channels=3,
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

    scheduler = CondOTScheduler()
    path = AffineProbPath(scheduler)

    criterion = nn.MSELoss()

    # -------- BatchOT Sampler (multisample coupling) --------
    # Mirrors the official OTPlanSampler usage. Method choices: exact | sinkhorn | unbalanced | partial
    ot_sampler = OTPlanSampler(
        method=args.ot_method,
        reg=args.ot_reg,
        reg_m=args.ot_reg_m,
        normalize_cost=args.ot_normalize_cost
    )

    # -------- Training Loop --------
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=rank != 0)

        for batch in pbar:
            x1 = batch["map"].to(local_rank, non_blocking=True)  # target
            x0 = batch["sat"].to(local_rank, non_blocking=True)  # source (conditioning)
            noise = torch.randn_like(x0)
            B = x0.shape[0]
            t = torch.rand(B, device=local_rank)

            # ---------- BatchOT coupling block ----------
            # Compute an OT plan between base samples (noise) and targets (x1), then
            # sample index pairs (i, j) ~ π and reorder tensors so each row matches.
            # IMPORTANT: also reorder the conditioner x0 to the chosen x1.
            pi = ot_sampler.get_map(noise.detach(), x1.detach())  # NumPy array
            ii_np, jj_np = ot_sampler.sample_map(pi, B)
            ii = torch.as_tensor(ii_np, device=local_rank, dtype=torch.long)
            jj = torch.as_tensor(jj_np, device=local_rank, dtype=torch.long)

            noise = noise.index_select(0, ii)
            x1    = x1.index_select(0, jj)
            x0    = x0.index_select(0, jj)
            # ---------- end BatchOT block ----------

            path_sample = path.sample(t=t, x_0=noise, x_1=x1)
            x_t = path_sample.x_t      # = (1-t)*noise + t*x1 for CondOT
            u_t = path_sample.dx_t     # = (-1)*noise + (1)*x1 for CondOT

            # Condition by concatenating source image
            x_input = torch.cat([x_t, x0], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(x_input, t, extra={})
                loss = (pred - u_t).pow(2).mean()

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
        dist.barrier()  # <-- make ALL ranks arrive here together

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # Eval interval: run FID on ALL ranks (distributed)
        if (epoch + 1) % args.eval_interval == 0:
            sat_dir = "/aul/homes/amaha038/Mapsgeneration/TerraFlySat_and_MapDatatset/TerraFly_Full_Map&Satellite_Dataset/Final_Sat_Map_Dataset/testA_5000"
            map_dir = "/aul/homes/amaha038/Mapsgeneration/TerraFlySat_and_MapDatatset/TerraFly_Full_Map&Satellite_Dataset/Final_Sat_Map_Dataset/testB_5000"

            fid_val = eval_fid_i2i(
                model.module, device=local_rank,
                sat_dir=sat_dir, map_dir=map_dir,
                out_dir=SAVE_DIR, epoch=epoch+1,
                steps=50, batch_size=32, num_workers=args.num_workers,
                save_samples=10
            )

            # after eval returns, free cached blocks on this rank
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # Rank 0 logs & saves; others just participate
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

            dist.barrier()  # <-- everyone waits until rank 0 finishes I/O

        lr_sched.step()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluate and save 10 samples every N epochs")

    # BatchOT options (match official OTPlanSampler)
    parser.add_argument("--ot_method", type=str, default="exact", choices=["exact", "sinkhorn", "unbalanced", "partial"])
    parser.add_argument("--ot_reg", type=float, default=0.05, help="Entropic regularization (used by sinkhorn/partial/unbalanced)")
    parser.add_argument("--ot_reg_m", type=float, default=1.0, help="Marginal relaxation (only for unbalanced)")
    parser.add_argument("--ot_normalize_cost", action="store_true", help="Divide cost matrix by its max before OT")

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
