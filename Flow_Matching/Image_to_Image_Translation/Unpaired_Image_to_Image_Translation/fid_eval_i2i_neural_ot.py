# FID evaluation for latent Neural OT image-to-image translation

import os, random
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from diffusers.models import AutoencoderKL


class EvalPairDataset(Dataset):
    def __init__(self, sat_dir, map_dir, size=256):
        self.sat_dir = sat_dir
        self.map_dir = map_dir
        self.to_src = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)   # [-1,1]
        ])
        self.to_real = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()                    # [0,1]
        ])

        self.sat_files = sorted([
            n for n in os.listdir(sat_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.map_files = sorted([
            n for n in os.listdir(map_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.num_sat = len(self.sat_files)
        self.num_map = len(self.map_files)

        if self.num_sat == 0:
            raise RuntimeError(f"No images found in sat_dir: {sat_dir}")
        if self.num_map == 0:
            raise RuntimeError(f"No images found in map_dir: {map_dir}")

    def __len__(self):
        # unpaired: iterate over all sat images, sample map randomly
        return self.num_sat

    def __getitem__(self, i):
        # source image
        n = self.sat_files[i]
        x0 = Image.open(os.path.join(self.sat_dir, n)).convert("RGB")

        # target/real image: random sample from map_dir
        j = random.randint(0, self.num_map - 1)
        m = self.map_files[j]
        x1 = Image.open(os.path.join(self.map_dir, m)).convert("RGB")

        return {
            "sat_pm1": self.to_src(x0),   # [-1,1]
            "map_01":  self.to_real(x1),  # [0,1]
            "name_sat": n,
            "name_map": m,
        }


def _pm1_to_01(x):
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def eval_fid_i2i_not(
    T_model, device,
    sat_dir, map_dir,
    out_dir, epoch,
    batch_size=16, num_workers=4,
    save_samples=10, scale_factor=0.18215
):
    """
    Evaluate FID for NOT-based latent I2I:
    - Encode source x_src -> z_src
    - Apply T_model(z_src)
    - Decode to pixels
    - FID vs real target images from map_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    T_model.eval()

    # Dist info
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)

    # Frozen VAE for encode/decode
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    vae.train = False
    for p in vae.parameters():
        p.requires_grad = False

    ds = EvalPairDataset(sat_dir, map_dir, size=256)
    sampler = DistributedSampler(
        ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    ) if is_dist else None

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Save qualitative samples
    if is_rank0 and save_samples > 0:
        sample_dir = os.path.join(out_dir, f"epoch_{epoch}_samples_not")
        os.makedirs(sample_dir, exist_ok=True)
        idxs = random.sample(range(len(ds)), min(save_samples, len(ds)))
        for i, idx in enumerate(idxs):
            b = ds[idx]
            x0 = b["sat_pm1"].unsqueeze(0).to(device)  # [-1,1], source
            x1 = b["map_01"].unsqueeze(0).to(device)   # [0,1], random real target

            # encode source latent
            z0 = vae.encode(x0).latent_dist.sample() * scale_factor

            # apply NOT map (latent)
            t_dummy = torch.ones(1, device=device)
            z_hat = T_model(z0, t_dummy, extra={})

            # decode to pixel
            gen_pm1 = vae.decode(z_hat / scale_factor).sample
            gen01 = _pm1_to_01(gen_pm1)

            vis = torch.cat([_pm1_to_01(x0), x1, gen01], dim=0)
            save_image(vis, os.path.join(sample_dir, f"{i:02d}.png"), nrow=3)

            del x0, x1, z0, z_hat, gen_pm1, gen01, vis

    # Progress bar (rank 0 shows its shard progress)
    total_local = len(sampler) if sampler is not None else len(ds)
    pbar = tqdm(
        total=total_local,
        desc=f"FID NOT (epoch {epoch}) [rank {rank}/{world}]",
        ncols=100,
        disable=not is_rank0
    )

    # Full FID pass
    if is_dist and sampler is not None:
        sampler.set_epoch(epoch)

    for batch in dl:
        real01 = batch["map_01"].to(device, non_blocking=True)   # [0,1], real target
        x_src = batch["sat_pm1"].to(device, non_blocking=True)   # [-1,1], source

        fid.update(real01, real=True)

        # encode source latent
        z0 = vae.encode(x_src).latent_dist.sample() * scale_factor

        # apply NOT transport map
        B = z0.size(0)
        t_dummy = torch.ones(B, device=device)
        z_hat = T_model(z0, t_dummy, extra={})

        # decode to pixel
        gen_pm1 = vae.decode(z_hat / scale_factor).sample
        gen01 = _pm1_to_01(gen_pm1)

        fid.update(gen01, real=False)

        if is_rank0:
            pbar.update(real01.size(0))

        del real01, x_src, z0, z_hat, gen_pm1, gen01

    if is_rank0:
        pbar.close()

    if is_dist:
        dist.barrier()

    fid_val = float(fid.compute().detach().cpu())

    if is_rank0:
        log_path = os.path.join(out_dir, "fid_scores_not.txt")
        need_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if need_header:
                f.write("epoch\tfid\n")
            f.write(f"{epoch}\t{fid_val:.6f}\n")

    if is_dist:
        dist.barrier()
    fid.reset()

    torch.cuda.synchronize()

    del dl, ds, fid, vae
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return fid_val
