# train_latent_neural_ot.py
# motivated from paper: NEURAL OPTIMAL TRANSPORT

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from torch.nn.utils import spectral_norm

from datasets import GenericI2IDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNetModel

from fid_eval_i2i_neural_ot import eval_fid_i2i_not  # FID file

from diffusers.models import AutoencoderKL
import copy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------- DDP helpers ----------------

def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


# ---------------- EMA for generator ----------------

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
    """Linear LR warmup: 0 -> base_lr over `warmup_steps` steps."""
    if warmup_steps <= 0:
        return
    if step >= warmup_steps:
        return
    scale = float(step) / float(max(1, warmup_steps))
    for g in optimizer.param_groups:
        g['lr'] = base_lr * scale


# ---------------- Critic (Kantorovich potential) ----------------

class Critic(nn.Module):
    """
    Simple CNN critic on RGB images in [-1, 1], outputs scalar per image.
    This plays the role of f in NOT: E[f(T(X))] - E[f(Y)].
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, c, 4, 2, 1)),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(c, c * 2, 4, 2, 1)),        # 128 -> 64
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(c * 2, c * 4, 4, 2, 1)),    # 64 -> 32
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(c * 4, c * 8, 4, 2, 1)),    # 32 -> 16
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(c * 8, c * 8, 4, 2, 1)),    # 16 -> 8
            nn.LeakyReLU(0.2, inplace=False),

            spectral_norm(nn.Conv2d(c * 8, 1, 8, 1, 0)),        # 8 -> 1
        )

    def forward(self, x):
        out = self.net(x)  # (B,1,1,1)
        return out.view(x.size(0))


def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)
    module.train()


# ---------------- Main training ----------------

def main(rank, world_size, args):

    local_rank = setup_ddp(rank, world_size)

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    DATA_ROOT = args.data_root
    SAVE_DIR = args.save_dir
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
    # Generator T: latent-space NOT map (z_src -> z_hat)
    # ---------------------------------------------------------
    T_model = UNetModel(
        in_channels=4,     # z_src channels only
        model_channels=192,
        out_channels=4,    # latent target channels
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

    T_model = DDP(T_model, device_ids=[local_rank], output_device=local_rank)
    ema = EMA(T_model.module, decay=args.ema_decay)

    # ---------------------------------------------------------
    # Critic f on decoded images
    # ---------------------------------------------------------
    critic = Critic(in_channels=3, base_channels=64).to(local_rank)
    critic = DDP(critic, device_ids=[local_rank], output_device=local_rank)

    # ---------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------
    T_opt = torch.optim.AdamW(T_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    f_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, betas=(0.9, 0.999))

    # ---------------------------------------------------------
    # Frozen SD-VAE (encode/decode)
    # ---------------------------------------------------------
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(local_rank)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    scale_factor = args.scale_factor

    # AMP scaler for T only
    scaler_T = torch.cuda.amp.GradScaler()
    global_step = 0

    # ---------------------------------------------------------
    # Training Loop: Neural OT in latent space
    # ---------------------------------------------------------
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        T_model.train()
        critic.train()
        total_T_loss = 0.0
        total_f_loss = 0.0

        if rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=(rank != 0))

        for batch in pbar:
            x_src = batch["sat"].to(local_rank, non_blocking=True)  # [-1,1], domain X
            x_tgt = batch["map"].to(local_rank, non_blocking=True)  # [-1,1], domain Y

            B = x_src.shape[0]

            # ------------ Encode to latent (no grad) ------------
            with torch.no_grad():
                z_src = vae.encode(x_src).latent_dist.sample() * scale_factor

            # -------------------------------------------------
            # 1) Optimize T (transport map) for T_iters steps
            # -------------------------------------------------
            freeze(critic)
            unfreeze(T_model)
            for _ in range(args.T_iters):
                T_opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    # dummy time (UNetModel signature)
                    t_dummy = torch.ones(B, device=local_rank)
                    z_hat = T_model(z_src, t_dummy, extra={})  # latent prediction

                    # decode to pixel space
                    x_hat_pm1 = vae.decode(z_hat / scale_factor).sample  # [-1,1]
                    x_hat_pm1 = x_hat_pm1.clamp(-1, 1)

                    # data term: preserve source structure (like NOT's MSE(X, T(X)))
                    data_loss = F.mse_loss(x_src, x_hat_pm1)

                    # OT dual term: -E[f(T(X))]
                    f_fake = critic(x_hat_pm1)
                    ot_loss = -f_fake.mean()
                    ot_loss = torch.clamp(ot_loss, -10.0, 10.0) 

                    # Total T loss
                    T_loss = data_loss + args.lambda_ot * ot_loss

                scaler_T.scale(T_loss).backward()
                scaler_T.unscale_(T_opt)
                torch.nn.utils.clip_grad_norm_(T_model.parameters(), 1.0)

                # LR warmup on generator
                global_step += 1
                adjust_lr(T_opt, global_step, base_lr=args.lr, warmup_steps=args.warmup_steps)

                scaler_T.step(T_opt)
                scaler_T.update()

                ema.update(T_model.module)

            total_T_loss += T_loss.item()

            # -------------------------------------------------
            # 2) Optimize critic f (Kantorovich potential)
            #    Pure FP32 here; no AMP / no scaler
            # -------------------------------------------------
            freeze(T_model)
            unfreeze(critic)
            f_opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                t_dummy = torch.ones(B, device=local_rank)
                z_hat = T_model(z_src, t_dummy, extra={})
                x_hat_pm1 = vae.decode(z_hat / scale_factor).sample
                x_hat_pm1 = x_hat_pm1.clamp(-1, 1)

            # critic forward in plain FP32
            f_fake = critic(x_hat_pm1)
            f_real = critic(x_tgt)
            # minimize E[f_fake] - E[f_real] = -(E[f_real] - E[f_fake])
            f_loss = f_fake.mean() - f_real.mean()
            f_loss = torch.clamp(f_loss, -10.0, 10.0)

            f_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            f_opt.step()

            total_f_loss += f_loss.item()

            if rank == 0:
                pbar.set_postfix(
                    T_loss=T_loss.item(),
                    f_loss=f_loss.item()
                )

            del x_src, x_tgt, z_src, z_hat, x_hat_pm1, f_fake, f_real

        # End epoch
        dist.barrier()
        avg_T_loss = total_T_loss / len(dataloader)
        avg_f_loss = total_f_loss / len(dataloader)

        if rank == 0:
            print(f"[Epoch {epoch+1}] T_loss: {avg_T_loss:.4f} | f_loss: {avg_f_loss:.4f}")

        # ---------------------------------------------------------
        # Eval / Save
        # ---------------------------------------------------------
        if (epoch + 1) % args.eval_interval == 0:
            if args.no_fid:
                if rank == 0:
                    torch.save(
                        T_model.module.state_dict(),
                        os.path.join(SAVE_DIR, f"model_epoch{epoch+1}.pth")
                    )
                    torch.save(
                        ema.state_dict(),
                        os.path.join(SAVE_DIR, f"model_epoch{epoch+1}_ema.pth")
                    )
                dist.barrier()
            else:
                ema_model = ema.ema_model.to(local_rank)
                fid_val = eval_fid_i2i_not(
                    ema_model,
                    device=local_rank,
                    sat_dir=args.testA,
                    map_dir=args.testB,
                    out_dir=SAVE_DIR,
                    epoch=epoch + 1,
                    batch_size=32,
                    num_workers=args.num_workers,
                    save_samples=10,
                    scale_factor=scale_factor,
                )

                dist.barrier()

                if rank == 0:
                    print(f"[Epoch {epoch+1}] FID (NOT): {fid_val:.6f}")
                    torch.save(T_model.module.state_dict(),
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
                        help="Path to testA folder for FID (source domain, e.g., sat).")
    parser.add_argument("--testB", type=str, required=True,
                        help="Path to testB folder for FID (target domain, e.g., map).")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Generator (T) learning rate.")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help="Critic (f) learning rate.")

    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--no_fid", action="store_true")
    parser.add_argument("--scale_factor", type=float, default=0.18215)
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay for generator parameters.")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps for generator LR.")

    parser.add_argument("--T_iters", type=int, default=1,
                        help="Number of NOT T-updates per batch.")
    parser.add_argument("--lambda_ot", type=float, default=1.0,
                        help="Weight of -E[f(T(X))] term in T loss.")

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)
