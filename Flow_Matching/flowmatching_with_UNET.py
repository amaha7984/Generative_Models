import os
import math
import time
import json
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.utils import save_image, make_grid
from torch import nn, optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Tuple, Optional
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------- Model Components ---------
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def conv2d(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def norm(channels):
    return nn.GroupNorm(32, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.in_layers = nn.Sequential(norm(in_ch), SiLU(), conv2d(in_ch, out_ch, 3, padding=1))
        self.emb_layers = nn.Sequential(SiLU(), linear(emb_ch, out_ch))
        self.out_layers = nn.Sequential(norm(out_ch), SiLU(), conv2d(out_ch, out_ch, 3, padding=1))
        self.skip_connection = conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).view(x.size(0), -1, 1, 1)
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class UNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=128, out_channels=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            linear(model_channels, model_channels * 4),
            SiLU(),
            linear(model_channels * 4, model_channels * 4)
        )
        self.input_conv = conv2d(in_channels, model_channels, 3, padding=1)
        self.res1 = ResBlock(model_channels, model_channels * 2, model_channels * 4)
        self.res2 = ResBlock(model_channels * 2, model_channels * 4, model_channels * 4)
        self.res3 = ResBlock(model_channels * 4, model_channels * 2, model_channels * 4)
        self.res4 = ResBlock(model_channels * 2, model_channels, model_channels * 4)
        self.output_conv = nn.Sequential(norm(model_channels), SiLU(), conv2d(model_channels, out_channels, 3, padding=1))

    def forward(self, x, t):
        emb = timestep_embedding(t, 128)
        emb = self.time_mlp(emb)
        x = self.input_conv(x)
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        x = self.res3(x, emb)
        x = self.res4(x, emb)
        return self.output_conv(x)

# --------- Training Setup ---------
def get_dataloader(data_path, image_size=256, batch_size=16):
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2. - 1.)
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --------- Main Training Loop ---------
def train(model, dataloader, epochs=100, save_dir="./flowmatching_output"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        
        model.train()
        losses = []
        for x1, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), device=device)
            xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
            target = x1 - x0

            pred = model(xt, t)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"✅ Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

        # Save samples every 10 epochs
        if epoch % 10 == 0:
            generate_and_save_samples(model, epoch, save_dir)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"💾 Best model saved at epoch {epoch} with loss {avg_loss:.4f}")


def generate_and_save_samples(model, epoch, save_dir):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.randn(6, 3, 256, 256).to(device)
        steps = 500
        for i, t_val in enumerate(torch.linspace(0, 1, steps)):
            t_vec = t_val.expand(xt.size(0)).to(device)
            xt = xt + (1.0 / steps) * model(xt, t_vec)

        imgs = xt.clamp(-1, 1) * 0.5 + 0.5  # scale to [0,1]
        grid = make_grid(imgs, nrow=3)
        save_path = os.path.join(save_dir, f"samples_epoch_{epoch}.png")
        save_image(grid, save_path)
        print(f"🖼️ Saved image grid at: {save_path}")

# --------- Entry ---------
if __name__ == "__main__":
    data_path = "/aul/homes/amaha038/Mapsgeneration/TerraFlySat_and_MapDatatset/TerraFly_Full_Satellite_Dataset/Philadelphia_Washington_Newyork_Train"  # 👈 Set your dataset path here (with subfolders per class)
    dataloader = get_dataloader(data_path)
    model = UNet()
    train(model, dataloader)
