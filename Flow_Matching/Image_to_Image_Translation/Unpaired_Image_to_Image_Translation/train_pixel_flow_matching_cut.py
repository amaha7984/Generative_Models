import os
import argparse
import copy
import math

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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.unet import UNetModel
from fid_eval_pixel_flow_matching_cut import eval_fid_pixel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# UTILITIES
# ============================================================================

def setup_ddp(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


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

    def state_dict(self):
        return self.ema_model.state_dict()


def init_weights(m):
    """Proper weight initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ============================================================================
# GENERATOR (UNet-based)
# ============================================================================

class Generator(nn.Module):
    """
    Generator: x_source → x_target_pred
    Uses UNet architecture
    """
    def __init__(self):
        super().__init__()
        self.model = UNetModel(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(4, 8),
            dropout=0.0,
            channel_mult=(1, 2, 4, 8),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_head_channels=32,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
        )
    
    def forward(self, x):
        # UNet expects (x, t, extra) but we don't use time for generator
        # Pass dummy time = 0
        t = torch.zeros(x.size(0), device=x.device)
        out = self.model(x, t, extra={})
        return torch.tanh(out)  # [-1, 1]


# ============================================================================
# FEATURE ENCODER (for pairing)
# ============================================================================

class FeatureEncoder(nn.Module):
    """
    Extracts features for computing semantic similarity.
    Used to find which target should pair with which source.
    """
    def __init__(self, ndf=64):
        super().__init__()
        # Lightweight encoder
        self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1)  # 128x128
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)  # 64x64
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)  # 32x32
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)  # 16x16
        self.conv5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)  # 8x8
        
        self.norm2 = nn.InstanceNorm2d(ndf*2)
        self.norm3 = nn.InstanceNorm2d(ndf*4)
        self.norm4 = nn.InstanceNorm2d(ndf*8)
        self.norm5 = nn.InstanceNorm2d(ndf*8)
        
        self.act = nn.LeakyReLU(0.2)
        
        # Global pooling to get feature vector
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, layer_ids=None):
        """
        Args:
            x: input image [B, 3, H, W]
            layer_ids: which layers to return features from (for multi-scale)
        
        Returns:
            if layer_ids is None: feature vector [B, 512]
            else: list of feature maps for contrastive learning
        """
        feats = []
        
        h1 = self.act(self.conv1(x))
        if layer_ids is not None and 0 in layer_ids:
            feats.append(h1)
        
        h2 = self.act(self.norm2(self.conv2(h1)))
        if layer_ids is not None and 1 in layer_ids:
            feats.append(h2)
        
        h3 = self.act(self.norm3(self.conv3(h2)))
        if layer_ids is not None and 2 in layer_ids:
            feats.append(h3)
        
        h4 = self.act(self.norm4(self.conv4(h3)))
        if layer_ids is not None and 3 in layer_ids:
            feats.append(h4)
        
        h5 = self.act(self.norm5(self.conv5(h4)))
        if layer_ids is not None and 4 in layer_ids:
            feats.append(h5)
        
        if layer_ids is not None:
            return feats
        
        # Return global feature vector
        feat = self.pool(h5).view(x.size(0), -1)  # [B, 512]
        return feat


# ============================================================================
# DISCRIMINATOR (PatchGAN)
# ============================================================================

class Discriminator(nn.Module):
    """
    PatchGAN discriminator (70x70 receptive field)
    """
    def __init__(self, ndf=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: [B, 3, 256, 256]
            nn.Conv2d(3, ndf, 4, 2, 1),  # [B, 64, 128, 128]
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),  # [B, 128, 64, 64]
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),  # [B, 256, 32, 32]
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 1),  # [B, 512, 31, 31]
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf*8, 1, 4, 1, 1),  # [B, 1, 30, 30]
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# PAIRING STRATEGY
# ============================================================================

@torch.no_grad()
def stochastic_pairing(feat_src, feat_tgt, temperature=0.07):
    """
    Stochastic feature-based pairing to avoid repetition.
    
    Key idea: Use learned features to find semantic similarity,
    then sample probabilistically (not deterministically).
    
    Args:
        feat_src: source features [B, D]
        feat_tgt: target features [B, D]
        temperature: controls diversity (lower = more peaked)
    
    Returns:
        indices: which target to pair with each source [B]
    """
    B = feat_src.size(0)
    
    # Normalize features
    feat_src = F.normalize(feat_src, dim=1)
    feat_tgt = F.normalize(feat_tgt, dim=1)
    
    # Compute similarity matrix
    sim = feat_src @ feat_tgt.T / temperature  # [B, B]
    
    # Convert to probabilities
    probs = F.softmax(sim, dim=1)  # [B, B]
    
    # Sample indices (stochastic → diversity!)
    # Each source samples from its probability distribution over targets
    indices = torch.multinomial(probs, 1).squeeze(1)  # [B]
    
    return indices


# ============================================================================
# CONTRASTIVE LOSS (from CUT)
# ============================================================================

class PatchNCELoss(nn.Module):
    """
    Patch-wise contrastive loss (from CUT paper).
    Encourages corresponding patches to be similar in feature space.
    """
    def __init__(self, temperature=0.07, num_patches=256):
        super().__init__()
        self.temperature = temperature
        self.num_patches = num_patches
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, feat_src, feat_tgt):
        """
        Args:
            feat_src: source features [B, C, H, W]
            feat_tgt: target features [B, C, H, W] (from paired target)
        """
        B, C, H, W = feat_src.shape
        
        # Reshape to patches
        feat_src = feat_src.view(B, C, -1)  # [B, C, H*W]
        feat_tgt = feat_tgt.view(B, C, -1)  # [B, C, H*W]
        
        # Sample random patches
        num_patches = min(self.num_patches, H * W)
        patch_ids = torch.randperm(H * W)[:num_patches]
        
        feat_src = feat_src[:, :, patch_ids]  # [B, C, num_patches]
        feat_tgt = feat_tgt[:, :, patch_ids]  # [B, C, num_patches]
        
        # Normalize
        feat_src = F.normalize(feat_src, dim=1)
        feat_tgt = F.normalize(feat_tgt, dim=1)
        
        # Compute similarity
        # For each source patch, compute similarity with all target patches
        feat_src = feat_src.permute(0, 2, 1)  # [B, num_patches, C]
        feat_tgt = feat_tgt.permute(0, 2, 1)  # [B, num_patches, C]
        
        total_loss = 0.0
        for i in range(B):
            # [num_patches, C] @ [C, num_patches] = [num_patches, num_patches]
            sim = feat_src[i] @ feat_tgt[i].T / self.temperature
            
            # Diagonal should be highest (corresponding patches)
            labels = torch.arange(num_patches, device=sim.device)
            loss = self.cross_entropy(sim, labels)
            total_loss += loss
        
        return total_loss / B


# ============================================================================
# FLOW MATCHING MODEL
# ============================================================================

class FlowModel(nn.Module):
    """
    Flow model for refinement.
    Takes (x_t, t) and predicts velocity field.
    """
    def __init__(self):
        super().__init__()
        self.model = UNetModel(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(4, 8),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_head_channels=32,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,
        )
    
    def forward(self, x, t):
        return self.model(x, t, extra={})


# ============================================================================
# TRAINING
# ============================================================================

def main(rank, world_size, args):
    local_rank = setup_ddp(rank, world_size)
    rank0 = (rank == 0)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Logging
    loss_log_path = os.path.join(args.save_dir, "loss_clean.txt")
    if rank0 and not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as f:
            f.write("epoch\tavg_loss\tavg_fm\tavg_gan_g\tavg_gan_d\tavg_nce\n")
    
    # Dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
    ])
    
    dataset = GenericI2IDataset(args.data_root, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Models
    generator = Generator().to(local_rank)
    generator.apply(init_weights)
    
    encoder = FeatureEncoder(ndf=64).to(local_rank)
    encoder.apply(init_weights)
    
    discriminator = Discriminator(ndf=64).to(local_rank)
    discriminator.apply(init_weights)
    
    flow_model = FlowModel().to(local_rank)
    flow_model.apply(init_weights)
    
    # DDP wrap
    generator = DDP(generator, device_ids=[local_rank])
    encoder = DDP(encoder, device_ids=[local_rank])
    discriminator = DDP(discriminator, device_ids=[local_rank])
    flow_model = DDP(flow_model, device_ids=[local_rank])
    
    # EMA
    ema_gen = EMA(generator.module, decay=args.ema_decay)
    ema_flow = EMA(flow_model.module, decay=args.ema_decay)
    
    # Optimizers
    opt_G = torch.optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()) + list(flow_model.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Loss functions
    criterion_gan = nn.MSELoss()  # LSGAN
    criterion_fm = nn.MSELoss()
    criterion_nce = PatchNCELoss(temperature=0.07, num_patches=256)
    
    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        generator.train()
        encoder.train()
        discriminator.train()
        flow_model.train()
        
        total_loss = 0.0
        total_fm = 0.0
        total_gan_g = 0.0
        total_gan_d = 0.0
        total_nce = 0.0
        
        if rank0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        pbar = tqdm(dataloader, desc=f"[GPU {local_rank}] Epoch {epoch+1}", disable=not rank0)
        
        for batch in pbar:
            x_src = batch["sat"].to(local_rank, non_blocking=True)
            x_tgt = batch["map"].to(local_rank, non_blocking=True)
            B = x_src.size(0)
            
            # ================================================================
            # STEP 1: Stochastic Pairing (changes every iteration!)
            # ================================================================
            with torch.no_grad():
                feat_src = encoder(x_src)  # [B, 512]
                feat_tgt = encoder(x_tgt)  # [B, 512]
                
                # Stochastic pairing based on feature similarity
                pair_indices = stochastic_pairing(feat_src, feat_tgt, temperature=args.pair_temp)
                x_tgt_paired = x_tgt[pair_indices]  # [B, 3, 256, 256]
            
            # ================================================================
            # STEP 2: Generate Target
            # ================================================================
            x_tgt_pred = generator(x_src)  # [B, 3, 256, 256]
            
            # ================================================================
            # STEP 3: Update Discriminator
            # ================================================================
            opt_D.zero_grad()
            
            # Real
            pred_real = discriminator(x_tgt)
            loss_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # Fake
            pred_fake = discriminator(x_tgt_pred.detach())
            loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()
            
            # ================================================================
            # STEP 4: Update Generator + Encoder + Flow Model
            # ================================================================
            opt_G.zero_grad()
            
            # 4a. GAN loss (fool discriminator)
            pred_fake = discriminator(x_tgt_pred)
            loss_gan_g = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            
            # 4b. Contrastive loss (preserve structure using PAIRED target)
            # Extract multi-scale features for NCE
            feat_src_layers = encoder(x_src, layer_ids=[0, 2, 4])
            feat_paired_layers = encoder(x_tgt_paired, layer_ids=[0, 2, 4])
            
            loss_nce = 0.0
            for feat_s, feat_p in zip(feat_src_layers, feat_paired_layers):
                loss_nce += criterion_nce(feat_s, feat_p)
            loss_nce = loss_nce / len(feat_src_layers)
            
            # 4c. Flow matching loss (refine generation)
            t = torch.rand(B, device=local_rank)
            t_expanded = t.view(B, 1, 1, 1)
            
            # Flow from source to PREDICTED target (not paired!)
            x_t = (1.0 - t_expanded) * x_src + t_expanded * x_tgt_pred
            if args.sigma > 0:
                x_t = x_t + args.sigma * torch.randn_like(x_t)
            
            v_pred = flow_model(x_t, t)
            v_target = x_tgt_pred - x_src
            loss_fm = criterion_fm(v_pred, v_target)
            
            # 4d. Total generator loss
            loss_G = (args.lambda_gan * loss_gan_g + 
                     args.lambda_nce * loss_nce + 
                     args.lambda_fm * loss_fm)
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters()) + list(encoder.parameters()) + list(flow_model.parameters()),
                1.0
            )
            opt_G.step()
            
            # Update EMA
            ema_gen.update(generator.module)
            ema_flow.update(flow_model.module)
            
            # Logging
            total_loss += loss_G.item()
            total_fm += loss_fm.item()
            total_gan_g += loss_gan_g.item()
            total_gan_d += loss_D.item()
            total_nce += loss_nce.item()
            
            if rank0:
                pbar.set_postfix(
                    fm=loss_fm.item(),
                    gan_g=loss_gan_g.item(),
                    nce=loss_nce.item()
                )
        
        dist.barrier()
        
        # Epoch logging
        n = len(dataloader)
        avg_loss = total_loss / n
        avg_fm = total_fm / n
        avg_gan_g = total_gan_g / n
        avg_gan_d = total_gan_d / n
        avg_nce = total_nce / n
        
        if rank0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, FM: {avg_fm:.4f}, "
                  f"GAN_G: {avg_gan_g:.4f}, GAN_D: {avg_gan_d:.4f}, NCE: {avg_nce:.4f}")
            
            with open(loss_log_path, "a") as f:
                f.write(f"{epoch+1}\t{avg_loss:.6f}\t{avg_fm:.6f}\t"
                       f"{avg_gan_g:.6f}\t{avg_gan_d:.6f}\t{avg_nce:.6f}\n")
            
            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                torch.save({
                    'generator': generator.module.state_dict(),
                    'encoder': encoder.module.state_dict(),
                    'flow_model': flow_model.module.state_dict(),
                    'ema_gen': ema_gen.state_dict(),
                    'ema_flow': ema_flow.state_dict(),
                    'epoch': epoch + 1,
                }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pth"))
        
        dist.barrier()
        
        # FID Evaluation
        if (not args.no_fid) and args.testA and args.testB and ((epoch + 1) % args.eval_interval == 0):
            
            fid_val = eval_fid_pixel(
                ema_gen.ema_model,
                device=local_rank,
                src_dir=args.testA,
                tgt_dir=args.testB,
                out_dir=args.save_dir,
                epoch=epoch + 1,
                batch_size=32,
                num_workers=args.num_workers,
                save_samples=10,
            )
            
            dist.barrier()
            
            if rank0:
                print(f"[Epoch {epoch+1}] FID: {fid_val:.6f}")
                
                # Save best FID checkpoint
                if fid_val < float('inf'):
                    torch.save({
                        'generator': generator.module.state_dict(),
                        'ema_gen': ema_gen.state_dict(),
                        'epoch': epoch + 1,
                        'fid': fid_val,
                    }, os.path.join(args.save_dir, "best_fid.pth"))
            
            dist.barrier()
    
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--save_interval", type=int, default=10)
    
    # FID evaluation
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--no_fid", action="store_true")
    parser.add_argument("--testA", type=str, default=None,
                       help="Path to test source images for FID")
    parser.add_argument("--testB", type=str, default=None,
                       help="Path to test target images for FID")
    
    # Loss weights
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_nce", type=float, default=1.0)
    parser.add_argument("--lambda_fm", type=float, default=1.0)
    
    # Pairing
    parser.add_argument("--pair_temp", type=float, default=0.07,
                       help="Temperature for stochastic pairing (lower = more peaked)")
    
    # Flow matching
    parser.add_argument("--sigma", type=float, default=0.0,
                       help="Noise level for flow matching (0 = deterministic)")
    
    args = parser.parse_args()
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size, args)