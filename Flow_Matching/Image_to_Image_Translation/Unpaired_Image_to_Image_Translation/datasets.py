# datasets.py
import os
import random  # <-- ADD THIS
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GenericI2IDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.sat_dir = os.path.join(root_dir, "trainA")
        self.map_dir = os.path.join(root_dir, "trainB")
        self.transform = transform

        # filter only image files (optional but safer)
        self.sat_files = sorted([
            f for f in os.listdir(self.sat_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.map_files = sorted([
            f for f in os.listdir(self.map_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.num_sat = len(self.sat_files)
        self.num_map = len(self.map_files)

        assert self.num_sat > 0, "No images found in trainA (sat_dir)."
        assert self.num_map > 0, "No images found in trainB (map_dir)."

    def __len__(self):
        # unpaired: iterate over all sat images, sample map randomly
        return self.num_sat

    def __getitem__(self, idx):
        sat_path = os.path.join(self.sat_dir, self.sat_files[idx])

        # unpaired: pick a random map image each time
        j = random.randint(0, self.num_map - 1)
        map_path = os.path.join(self.map_dir, self.map_files[j])

        sat_img = Image.open(sat_path).convert("RGB")
        map_img = Image.open(map_path).convert("RGB")

        if self.transform:
            sat_img = self.transform(sat_img)
            map_img = self.transform(map_img)

        return {"sat": sat_img, "map": map_img}
