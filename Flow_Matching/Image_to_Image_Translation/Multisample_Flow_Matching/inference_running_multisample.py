import os, math, json, argparse, random, fcntl
import numpy as np
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from datetime import timedelta


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet import UNetModel



class EvalPairDataset(Dataset):
    def __init__(self, sat_dir: str, map_dir: str, size: int = 256):
        self.sat_dir = sat_dir
        self.map_dir = map_dir
        self.to_src  = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Lambda(lambda x: x*2-1)])  # [-1,1]
        self.to_real = T.Compose([T.Resize((size, size)), T.ToTensor()])                              # [0,1]

        a_names = sorted([n for n in os.listdir(sat_dir)
                          if n.endswith((".jpg", ".png")) and "_A." in n])
        self.pairs = []
        for n in a_names:
            base, ext = os.path.splitext(n)
            m = base.replace("_A", "_B") + ext
            if os.path.exists(os.path.join(map_dir, m)):
                self.pairs.append((n, m))

        if len(self.pairs) == 0:
            raise RuntimeError("No A→B pairs found. Expected *_A.ext in A and *_B.ext in B.")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i: int):
        n, m = self.pairs[i]
        x0 = Image.open(os.path.join(self.sat_dir, n)).convert("RGB")
        x1 = Image.open(os.path.join(self.map_dir, m)).convert("RGB")
        return {"sat_pm1": self.to_src(x0), "map_01": self.to_real(x1), "name": n}



def make_t_grid(nfe: int, kind: str = "cosine") -> np.ndarray:
    """
    Returns a strictly increasing np.array of length nfe+1 with endpoints 0 and 1.
    kind in {"uniform", "cosine", "front2", "back2"}.
    """
    assert nfe >= 1
    if kind == "uniform":
        return np.linspace(0.0, 1.0, nfe+1, dtype=np.float64)

    # cosine (bi-end dense): t_i = 0.5*(1 - cos(pi*i/N))
    if kind == "cosine":
        i = np.arange(nfe+1, dtype=np.float64)
        return 0.5 * (1.0 - np.cos(np.pi * i / nfe))

    # front-loaded (more resolution near 0): power-2
    if kind == "front2":
        u = np.linspace(0.0, 1.0, nfe+1, dtype=np.float64)
        return u**2

    # back-loaded (more near 1): power-2
    if kind == "back2":
        u = np.linspace(0.0, 1.0, nfe+1, dtype=np.float64)
        return 1.0 - (1.0 - u)**2

    raise ValueError(f"Unknown schedule kind: {kind}")


def load_t_grid_from_file(path: str) -> np.ndarray:
    """
    Load a custom time grid from:
      - .npy: a 1D array
      - .json: list of floats
      - .txt: whitespace- or comma-separated floats
    Must include 0 and 1 and be strictly increasing.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".json":
        with open(path, "r") as f:
            arr = np.array(json.load(f), dtype=np.float64)
    else:
        with open(path, "r") as f:
            txt = f.read().replace(",", " ")
        vals = [float(tok) for tok in txt.split()]
        arr = np.array(vals, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError("t_grid file must be a 1D array/list of floats.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("t_grid must be strictly increasing.")
    if abs(arr[0]) > 1e-12 or abs(arr[-1] - 1.0) > 1e-12:
        raise ValueError("t_grid must start at 0.0 and end at 1.0.")
    return arr.astype(np.float64)


# ---------------------------
# RK4 with variable time grid  [keep from File 1]
# ---------------------------
@torch.no_grad()
def rk4_generate_with_grid(model, x_src_pm1: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """
    x' = v_theta(x, t | src), RK4 over a *given* nonuniform time grid.
    x_src_pm1 in [-1,1]; returns x in [-1,1].
    t_grid: shape [N+1], strictly increasing from 0 to 1 (on the same device).
    """
    device = x_src_pm1.device
    x = torch.randn_like(x_src_pm1)

    # ensure on device & dtype (from File 1)
    t_grid = t_grid.to(device=device, dtype=x.dtype)
    assert t_grid.ndim == 1 and t_grid.numel() >= 2
    assert torch.all(t_grid[1:] > t_grid[:-1])
    assert abs(float(t_grid[0].item())) < 1e-12 and abs(float(t_grid[-1].item()) - 1.0) < 1e-12

    for i in range(t_grid.numel() - 1):
        t0 = float(t_grid[i].item())
        t1 = float(t_grid[i+1].item())
        h  = t1 - t0

        def f(t_s: float, x_s: torch.Tensor):
            tb = torch.full((x_s.size(0),), t_s, device=device, dtype=x.dtype)
            xin = torch.cat([x_s, x_src_pm1], dim=1)  # condition by concat
            return model(xin, tb, extra={})

        k1 = f(t0, x)
        k2 = f(t0 + 0.5*h, x + 0.5*h*k1)
        k3 = f(t0 + 0.5*h, x + 0.5*h*k2)
        k4 = f(t1,         x + h*k3)
        x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return x


def _pm1_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0

def _add_done(done_file: str, inc: int) -> int:
    with open(done_file, "a+b") as f:
        f.seek(0)
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        cur = int(raw.decode() or "0")
        new = cur + inc
        f.seek(0); f.truncate(0); f.write(str(new).encode()); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return new

# ---------------------------
# Dynamic work-queue saving (copied/adapted from File 2)
# ---------------------------
def _claim_next_chunk(cursor_file: str, chunk_size: int, n_total: int):
    """
    Atomically claim [start, end) from a shared counter in cursor_file.
    Returns (start, end) or None if no work remains.
    """
    with open(cursor_file, "a+b") as f:
        f.seek(0)
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        raw = f.read()
        cur = int(raw.decode() or "0")
        if cur >= n_total:
            fcntl.flock(f, fcntl.LOCK_UN)
            return None
        start = cur
        end = min(cur + chunk_size, n_total)
        f.seek(0); f.truncate(0); f.write(str(end).encode()); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return (start, end)


@torch.no_grad()
def generate_all_dynamic(
    model: torch.nn.Module,
    device: torch.device,
    ds: EvalPairDataset,
    gen_dir: str,
    t_grid: torch.Tensor,
    batch_size: int,
    cursor_file: str,
    chunk_size: int,
):
    os.makedirs(gen_dir, exist_ok=True)
    model.eval()

    is_dist  = dist.is_available() and dist.is_initialized()
    rank     = dist.get_rank() if is_dist else 0
    world    = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)

    n_total = len(ds)
    done_file = cursor_file + ".done"

    if is_rank0:
        with open(cursor_file, "wb") as f:
            f.write(b"0")
        with open(done_file, "wb") as f:
            f.write(b"0")
    if is_dist: dist.barrier()

    pbar = tqdm(total=n_total, desc=f"Generate (dynamic) [rank {rank}/{world}]", ncols=100, disable=not is_rank0)

    while True:
        claim = _claim_next_chunk(cursor_file, chunk_size, n_total)
        if claim is None:
            break
        start, end = claim

        idx = start
        while idx < end:
            jend = min(idx + batch_size, end)

            xs, names = [], []
            for i in range(idx, jend):
                item = ds[i]
                xs.append(item["sat_pm1"])
                names.append(item["name"])
            x_src = torch.stack(xs, dim=0).to(device, non_blocking=True)

            gen_pm1 = rk4_generate_with_grid(model, x_src, t_grid)
            gen01   = _pm1_to_01(gen_pm1)

            gen01_cpu = gen01.detach().cpu()
            for j in range(gen01_cpu.size(0)):
                nm = names[j]
                base, _ = os.path.splitext(nm)
                out_name = (base.replace("_A", "_B_pred") if "_A" in base else base + "_pred") + ".png"
                save_image(gen01_cpu[j], os.path.join(gen_dir, out_name))

            del x_src, gen_pm1, gen01, gen01_cpu
            n_inc = jend - idx
            new_total = _add_done(done_file, n_inc)
            if is_rank0:
                pbar.n = new_total
                pbar.refresh()

            idx = jend

    if is_rank0:
        pbar.close()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ---------------------------
# DDP helpers (keep File 1 style)
# ---------------------------
def setup_ddp(rank, world_size):
    """Initialize distributed process group (NCCL) and set device (same as working script)."""
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=2),
    )
    return local_rank


def cleanup_ddp():
    """Destroy distributed process group cleanly."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def load_state_dict_strict(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
    new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    model.load_state_dict(new_sd, strict=True)



# ---------------------------
# Main
# ---------------------------
def main(rank, world_size):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights .pth (trained multisample model).")
    parser.add_argument("--sat_dir", type=str, required=True,
                        help="Directory with *_A.{jpg,png} source images.")
    parser.add_argument("--map_dir", type=str, required=True,
                        help="Directory with *_B.{jpg,png} target images (only used to keep pairing logic).")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Where to write generated images & cursor file.")

    parser.add_argument("--nfe", type=int, default=50, help="Number of RK4 steps (ignored if --t_grid_file is set).")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["uniform", "cosine", "front2", "back2"],
                        help="Time grid type if not using --t_grid_file.")
    parser.add_argument("--t_grid_file", type=str, default=None,
                        help="Optional path to a custom time grid file (.npy/.json/.txt). Must be strictly increasing from 0 to 1.")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)  # retained from File 1 (unused)
    parser.add_argument("--epoch_tag", type=int, default=0, help="Used in output folder naming.")
    parser.add_argument("--save_samples", type=int, default=10)  # retained from File 1 (unused)
    # New (copied from File 2) for dynamic work queue & output dir
    parser.add_argument("--gen_dir", type=str, default=None, help="Optional explicit output image dir.")
    parser.add_argument("--chunk_size", type=int, default=256, help="Images claimed per chunk by a rank.")
    args = parser.parse_args()

    local_rank = setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Build the same model architecture as in training (keep from File 1)
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
    ).to(device)

    # Load weights (keep File 1's strict load to avoid changing behavior)
    load_state_dict_strict(model, args.weights, device)
    model.eval()

    # Build the time grid (keep from File 1)
    if args.t_grid_file is not None:
        t_np = load_t_grid_from_file(args.t_grid_file)
    else:
        t_np = make_t_grid(args.nfe, args.schedule)
    t_grid = torch.from_numpy(t_np).to(device=device, dtype=torch.float32)  # rk4 will move to device/dtype

    # Data (keep File 1 pairing to preserve names like *_A)
    ds = EvalPairDataset(args.sat_dir, args.map_dir, size=256)

    # Output + shared cursor (copied from File 2)
    os.makedirs(args.out_dir, exist_ok=True)
    gen_dir = args.gen_dir or os.path.join(args.out_dir, f"epoch_{args.epoch_tag}_gen")
    cursor_file = os.path.join(args.out_dir, f".cursor_e{args.epoch_tag}.txt")

    # Generate with dynamic work sharing (copied from File 2)
    generate_all_dynamic(
        model=model,
        device=device,
        ds=ds,
        gen_dir=gen_dir,
        t_grid=t_grid,
        batch_size=args.batch_size,
        cursor_file=cursor_file,
        chunk_size=args.chunk_size,
    )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size)
