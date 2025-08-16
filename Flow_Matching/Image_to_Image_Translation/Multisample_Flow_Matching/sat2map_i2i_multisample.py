import os
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

from eval_i2i_multisample import eval_fid_i2i
from ot_sampler import OTPlanSampler


def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    return local_rank

def cleanup_ddp():
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
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1] for training
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

    # -------- Model --------
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

    # -------- Flow Matching Path (CondOT) --------
    scheduler = CondOTScheduler()
    path = AffineProbPath(scheduler)

    # -------- Multisample OT Sampler (optional) --------
    # Use --ot_method exact for BatchOT as in the paper. For Sinkhorn, tune --ot_reg (e.g., 0.05 ~ 2*sigma^2).
    ot_sampler = OTPlanSampler(method=args.ot_method, reg=args.ot_reg) if args.ot_method else None

    scaler = torch.cuda.amp.GradScaler()

    # -------- Training Loop --------
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=rank != 0)
        for batch in pbar:
            # x0: source/conditioning (satellite) in [-1,1]
            # x1: target (map) in [-1,1]
            x1 = batch["map"].to(local_rank, non_blocking=True)
            x0 = batch["sat"].to(local_rank, non_blocking=True)

            # Gaussian base samples (independent of x0)
            z = torch.randn_like(x1)  # base noise ~ N(0, I)

            # Multisample: couple (z, x1) within the minibatch via OT if enabled
            if ot_sampler is not None:
                with torch.no_grad():
                    z, x1 = ot_sampler.sample_plan(z, x1)

            B = x0.shape[0]
            t = torch.rand(B, device=local_rank)

            # CondOT path: x_t = (1-t) * z + t * x1, u_t = x1 - z
            ps = path.sample(t=t, x_0=z, x_1=x1)
            x_t, u_t = ps.x_t, ps.dx_t

            # condition on source by concatenation
            x_in = torch.cat([x_t, x0], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(x_in, t, extra={})
                loss = (pred - u_t).pow(2).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix(loss=loss.item())

        torch.cuda.synchronize()
        dist.barrier()

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        # -------- Eval (distributed FID) --------
        if (epoch + 1) % args.eval_interval == 0:
            sat_dir = "/aul/homes/amaha038/Mapsgeneration/TerraFlySat_and_MapDatatset/TerraFly_Full_Map&Satellite_Dataset/Final_Sat_Map_Dataset/testA_5000"
            map_dir = "/aul/homes/amaha038/Mapsgeneration/TerraFlySat_and_MapDatatset/TerraFly_Full_Map&Satellite_Dataset/Final_Sat_Map_Dataset/testB_5000"

            fid_val = eval_fid_i2i(
                model.module, device=local_rank,
                sat_dir=sat_dir, map_dir=map_dir,
                out_dir=SAVE_DIR, epoch=epoch+1,
                steps=50, batch_size=32, num_workers=args.num_workers,
                save_samples=10,
                ot_sampler=ot_sampler  # same coupling choice as training
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
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--ot_method", type=str, default=None,
                        choices=[None, "exact", "sinkhorn"],
                        help="None (independent), 'exact' (BatchOT), 'sinkhorn' (BatchEOT)")
    parser.add_argument("--ot_reg", type=float, default=0.05,
                        help="Entropic reg for sinkhorn (try ~ 0.05 if you use sinkhorn)")

    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size, args)
