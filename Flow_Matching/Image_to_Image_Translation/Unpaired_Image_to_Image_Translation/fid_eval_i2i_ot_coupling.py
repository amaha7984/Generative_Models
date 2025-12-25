# fid_eval_i2i_ot_coupling.py


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
        self.to_src  = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x*2 - 1)   # [-1,1]
        ])
        self.to_real = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()                  # [0,1]
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
        return self.num_sat

    def __getitem__(self, i):
        n = self.sat_files[i]
        x0 = Image.open(os.path.join(self.sat_dir, n)).convert("RGB")

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
def _rk4_generate_latent_rectified(model, z0, steps=50):
    device = z0.device
    z = z0.clone()
    B = z.shape[0]

    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t0, t1 = ts[i].item(), ts[i + 1].item()
        h = t1 - t0

        def f_scalar(t_s, z_s):
            tb = torch.full((B,), t_s, device=device, dtype=z_s.dtype)
            return model(z_s, tb, extra={})

        k1 = f_scalar(t0, z)
        k2 = f_scalar(t0 + 0.5*h, z + 0.5*h*k1)
        k3 = f_scalar(t0 + 0.5*h, z + 0.5*h*k2)
        k4 = f_scalar(t0 + h,     z + h*k3)
        z  = z + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return z


@torch.no_grad()
def _rk4_generate_pixel_rectified(model, x0_pm1, steps=50):
    device = x0_pm1.device
    x = x0_pm1.clone()
    B = x.shape[0]

    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t0, t1 = ts[i].item(), ts[i + 1].item()
        h = t1 - t0

        def f_scalar(t_s, x_s):
            tb = torch.full((B,), t_s, device=device, dtype=x_s.dtype)
            return model(x_s, tb, extra={})

        k1 = f_scalar(t0, x)
        k2 = f_scalar(t0 + 0.5*h, x + 0.5*h*k1)
        k3 = f_scalar(t0 + 0.5*h, x + 0.5*h*k2)
        k4 = f_scalar(t0 + h,     x + h*k3)
        x  = x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x


@torch.no_grad()
def eval_fid_i2i_rectified(
    model, device,
    sat_dir, map_dir,
    out_dir, epoch, steps=200, batch_size=16, num_workers=4,
    save_samples=10,
    mode="latent",
    scale_factor=0.18215,
    vae_id="stabilityai/sd-vae-ft-mse",
):
    """
    mode:
      - "latent": integrate in latent space, decode with VAE
      - "dino"  : integrate in pixel space
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    is_dist  = dist.is_available() and dist.is_initialized()
    rank     = dist.get_rank() if is_dist else 0
    world    = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)

    vae = None
    if mode == "latent":
        vae = AutoencoderKL.from_pretrained(vae_id).to(device)
        vae.eval()
        vae.train = False
        for p in vae.parameters():
            p.requires_grad = False

    ds = EvalPairDataset(sat_dir, map_dir, size=256)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if is_dist else None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Save a few samples
    if is_rank0 and save_samples > 0:
        sample_dir = os.path.join(out_dir, f"epoch_{epoch}_samples_rectified")
        os.makedirs(sample_dir, exist_ok=True)
        idxs = random.sample(range(len(ds)), min(save_samples, len(ds)))
        for i, idx in enumerate(idxs):
            b  = ds[idx]
            x0 = b["sat_pm1"].unsqueeze(0).to(device)  # [-1,1]
            x1 = b["map_01"].unsqueeze(0).to(device)   # [0,1]

            if mode == "latent":
                z0 = vae.encode(x0).latent_dist.sample() * scale_factor
                z1_hat = _rk4_generate_latent_rectified(model, z0, steps=steps)
                gen_pm1 = vae.decode(z1_hat / scale_factor).sample
                gen01 = _pm1_to_01(gen_pm1)
            else:
                x1_hat_pm1 = _rk4_generate_pixel_rectified(model, x0, steps=steps)
                gen01 = _pm1_to_01(x1_hat_pm1)

            vis = torch.cat([_pm1_to_01(x0), x1, gen01], dim=0)
            save_image(vis, os.path.join(sample_dir, f"{i:02d}.png"), nrow=3)

            del x0, x1, vis, gen01
            if mode == "latent":
                del z0, z1_hat, gen_pm1
            else:
                del x1_hat_pm1

    total_local = len(sampler) if sampler is not None else len(ds)
    pbar = tqdm(total=total_local,
                desc=f"FID Rectified (epoch {epoch}) [rank {rank}/{world}]",
                ncols=100, disable=not is_rank0)

    if is_dist and sampler is not None:
        sampler.set_epoch(epoch)

    for batch in dl:
        real01 = batch["map_01"].to(device, non_blocking=True)   # [0,1]
        x_src  = batch["sat_pm1"].to(device, non_blocking=True)  # [-1,1]

        fid.update(real01, real=True)

        if mode == "latent":
            z0 = vae.encode(x_src).latent_dist.sample() * scale_factor
            z1_hat = _rk4_generate_latent_rectified(model, z0, steps=steps)
            gen_pm1 = vae.decode(z1_hat / scale_factor).sample
            gen01 = _pm1_to_01(gen_pm1)
            del z0, z1_hat, gen_pm1
        else:
            gen_pm1 = _rk4_generate_pixel_rectified(model, x_src, steps=steps)
            gen01 = _pm1_to_01(gen_pm1)
            del gen_pm1

        fid.update(gen01, real=False)

        if is_rank0:
            pbar.update(real01.size(0))

        del real01, x_src, gen01

    if is_rank0:
        pbar.close()

    if is_dist:
        dist.barrier()

    fid_val = float(fid.compute().detach().cpu())

    if is_rank0:
        log_path = os.path.join(out_dir, "fid_scores_rectified.txt")
        need_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if need_header:
                f.write("epoch\tfid\n")
            f.write(f"{epoch}\t{fid_val:.6f}\n")

    if is_dist:
        dist.barrier()
    fid.reset()

    torch.cuda.synchronize()

    del dl, ds, fid
    if vae is not None:
        del vae
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return fid_val
