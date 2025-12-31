import os, random
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class EvalDataset(Dataset):
    def __init__(self, src_dir, tgt_dir, size=256):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        
        self.transform_src = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2 - 1)  # [-1, 1] for generator input
        ])
        
        self.transform_tgt = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()  # [0, 1] for FID
        ])
        
        self.src_files = sorted([
            n for n in os.listdir(src_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.tgt_files = sorted([
            n for n in os.listdir(tgt_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        
        if len(self.src_files) == 0:
            raise RuntimeError(f"No images in {src_dir}")
        if len(self.tgt_files) == 0:
            raise RuntimeError(f"No images in {tgt_dir}")
    
    def __len__(self):
        return len(self.src_files)
    
    def __getitem__(self, idx):
        # Source
        src_path = os.path.join(self.src_dir, self.src_files[idx])
        src_img = Image.open(src_path).convert("RGB")
        src_tensor = self.transform_src(src_img)
        
        # Target (random for unpaired)
        tgt_idx = random.randint(0, len(self.tgt_files) - 1)
        tgt_path = os.path.join(self.tgt_dir, self.tgt_files[tgt_idx])
        tgt_img = Image.open(tgt_path).convert("RGB")
        tgt_tensor = self.transform_tgt(tgt_img)
        
        return {
            "src": src_tensor,  # [-1, 1]
            "tgt": tgt_tensor,  # [0, 1]
            "name": self.src_files[idx]
        }


def to_01(x):
    """Convert [-1, 1] to [0, 1]"""
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def eval_fid_pixel(
    generator,
    device,
    src_dir,
    tgt_dir,
    out_dir,
    epoch,
    batch_size=32,
    num_workers=4,
    save_samples=10
):
    """
    Evaluate FID for pixel-space generator.
    
    Args:
        generator: G(x_src) -> x_tgt (outputs in [-1, 1])
        device: cuda device
        src_dir: source test images
        tgt_dir: target test images
        out_dir: output directory
        epoch: current epoch
        batch_size: batch size
        num_workers: dataloader workers
        save_samples: number of samples to save
    """
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    
    # DDP info
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world = dist.get_world_size() if is_dist else 1
    is_rank0 = (rank == 0)
    
    # Dataset
    dataset = EvalDataset(src_dir, tgt_dir, size=256)
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    ) if is_dist else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # FID metric
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Save visual samples
    if is_rank0 and save_samples > 0:
        sample_dir = os.path.join(out_dir, f"epoch_{epoch}_samples")
        os.makedirs(sample_dir, exist_ok=True)
        
        indices = random.sample(range(len(dataset)), min(save_samples, len(dataset)))
        for i, idx in enumerate(indices):
            batch = dataset[idx]
            src = batch["src"].unsqueeze(0).to(device)  # [1, 3, 256, 256] in [-1, 1]
            tgt = batch["tgt"].unsqueeze(0).to(device)  # [1, 3, 256, 256] in [0, 1]
            
            # Generate
            gen = generator(src)  # [1, 3, 256, 256] in [-1, 1]
            gen_01 = to_01(gen)
            
            # Visualize: [source, real_target, generated]
            vis = torch.cat([to_01(src), tgt, gen_01], dim=0)
            save_image(vis, os.path.join(sample_dir, f"{i:02d}.png"), nrow=3)
    
    # FID computation
    if is_dist and sampler is not None:
        sampler.set_epoch(epoch)
    
    pbar = tqdm(
        dataloader,
        desc=f"FID Eval Epoch {epoch} [rank {rank}/{world}]",
        disable=not is_rank0
    )
    
    for batch in pbar:
        src = batch["src"].to(device, non_blocking=True)  # [-1, 1]
        tgt = batch["tgt"].to(device, non_blocking=True)  # [0, 1]
        
        # Update FID with real targets
        fid.update(tgt, real=True)
        
        # Generate and update FID with fakes
        gen = generator(src)  # [-1, 1]
        gen_01 = to_01(gen)   # [0, 1]
        fid.update(gen_01, real=False)
    
    if is_rank0:
        pbar.close()
    
    if is_dist:
        dist.barrier()
    
    # Compute FID
    fid_val = float(fid.compute().detach().cpu())
    
    # Log
    if is_rank0:
        log_path = os.path.join(out_dir, "fid_scores.txt")
        need_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if need_header:
                f.write("epoch\tfid\n")
            f.write(f"{epoch}\t{fid_val:.6f}\n")
    
    if is_dist:
        dist.barrier()
    
    fid.reset()
    torch.cuda.empty_cache()
    
    return fid_val