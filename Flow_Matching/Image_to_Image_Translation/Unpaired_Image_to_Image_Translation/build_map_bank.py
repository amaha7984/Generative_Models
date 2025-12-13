#This file is used to generate a bank of all the target images embedded with DINOv3
#The saved memory bank is used by train_latent_rectified_dino_bank_patchnce.py training logic
import os
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from diffusers.models import AutoencoderKL


# -----------------------------
# Dataset
# -----------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root: str, size: int = 256):
        self.root = root
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith(exts)
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in: {root}")

        self.tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        path = os.path.join(self.root, name)
        img = Image.open(path).convert("RGB")
        x_pm1 = self.tf(img)
        return x_pm1, name


# -----------------------------
# DINOv3 (global CLS embedding)
# -----------------------------
class DINOv3FeatureExtractor(nn.Module):
    def __init__(
        self,
        repo_dir: str,
        weights: str,
        stats: str = "LVD",
        freeze_backbone: bool = True,
    ):
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

        self.feat_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)
        assert self.feat_dim is not None, "Cannot infer DINOv3 feature dim."

    def _norm_01(self, x_01: torch.Tensor) -> torch.Tensor:
        mean = x_01.new_tensor(self.mean)[None, :, None, None]
        std  = x_01.new_tensor(self.std)[None, :, None, None]
        return (x_01 - mean) / std

    @torch.no_grad()
    def forward_cls(self, x_pm1: torch.Tensor) -> torch.Tensor:
        """
        x_pm1: [-1,1], [B,3,H,W]
        returns: [B, D] CLS embedding (x_norm_clstoken) if available
        """
        x_01 = (x_pm1.clamp(-1, 1) + 1.0) / 2.0
        x_norm = self._norm_01(x_01)

        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x_norm)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]  # [B, D]

        y = self.backbone(x_norm)
        if torch.is_tensor(y) and y.dim() == 2:
            return y

        raise RuntimeError("Unexpected DINOv3 output; cannot obtain CLS embedding.")


# -----------------------------
# Helpers
# -----------------------------
def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    # Data
    ds = ImageFolderDataset(args.map_dir, size=args.size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Models (frozen)
    print("[Bank] Loading SD-VAE ...")
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("[Bank] Loading DINOv3 ...")
    dino = DINOv3FeatureExtractor(
        repo_dir=args.dino_repo_dir,
        weights=args.dino_weights,
        stats=args.dino_stats,
        freeze_backbone=True,
    ).to(device)
    dino.eval()

    # Output dtype (for storage)
    save_dtype = parse_dtype(args.save_dtype)

    feats_list: List[torch.Tensor] = []
    latents_list: List[torch.Tensor] = []
    paths_list: List[str] = []

    autocast_dtype = parse_dtype(args.autocast_dtype)

    print(f"[Bank] Building map bank from: {args.map_dir}")
    print(f"[Bank] N_map = {len(ds)} | save_dtype={save_dtype} | autocast_dtype={autocast_dtype}")
    print(f"[Bank] Output: {args.out}")

    with torch.no_grad():
        pbar = tqdm(total=len(ds), ncols=100)
        for x_pm1, names in dl:
            x_pm1 = x_pm1.to(device, non_blocking=True)

            # mixed precision compute if desired
            use_amp = (device.type == "cuda") and args.use_amp
            ctx = torch.cuda.amp.autocast(dtype=autocast_dtype) if use_amp else torch.autocast("cpu", enabled=False)

            with ctx:
                # DINO CLS embedding
                f = dino.forward_cls(x_pm1)  # [B,D]
                f = F.normalize(f, dim=1)   # cosine-ready

                # VAE latent
                z = vae.encode(x_pm1).latent_dist.sample() * args.scale_factor  # [B,4,h,w]

            # Move to CPU and cast for storage
            feats_list.append(f.detach().to("cpu", dtype=save_dtype))
            latents_list.append(z.detach().to("cpu", dtype=save_dtype))
            paths_list.extend(list(names))

            pbar.update(x_pm1.size(0))
        pbar.close()

    feats = torch.cat(feats_list, dim=0)       # [N,D]
    latents = torch.cat(latents_list, dim=0)   # [N,4,h,w]

    bank = {
        "feats": feats,         # L2-normalized
        "latents": latents,     # scaled by scale_factor
        "paths": paths_list,
        "meta": {
            "map_dir": args.map_dir,
            "size": args.size,
            "vae_name": args.vae_name,
            "scale_factor": args.scale_factor,
            "dino_repo_dir": args.dino_repo_dir,
            "dino_weights": args.dino_weights,
            "dino_stats": args.dino_stats,
            "save_dtype": args.save_dtype,
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(bank, args.out)

    print("[Bank] Done.")
    print(f"[Bank] feats:   {tuple(feats.shape)} dtype={feats.dtype}")
    print(f"[Bank] latents: {tuple(latents.shape)} dtype={latents.dtype}")
    print(f"[Bank] saved to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--map_dir", type=str, required=True,
                        help="Path to map images folder (e.g., trainB).")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .pt path (e.g., map_bank.pt).")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--vae_name", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--scale_factor", type=float, default=0.18215)

    parser.add_argument("--dino_repo_dir", type=str,
                        default="/path/to/original_implementation/dinov3",
                        help="Local repo dir for torch.hub.load of DINOv3.")
    parser.add_argument("--dino_weights", type=str,
                        default="/path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                        help="Path or URL to DINOv3 weights.")
    parser.add_argument("--dino_stats", type=str, default="LVD",
                        help="Normalization stats: LVD or SAT.")

    # Storage / precision
    parser.add_argument("--save_dtype", type=str, default="fp16",
                        help="Storage dtype for feats/latents: fp16, bf16, fp32.")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use autocast during feature/latent computation (GPU only).")
    parser.add_argument("--autocast_dtype", type=str, default="bf16",
                        help="Autocast dtype when --use_amp is set: bf16 or fp16.")

    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu (cuda recommended).")

    args = parser.parse_args()
    main(args)
