import torch
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage


def get_train_transform():
    transform_list = [
        ToImage(),
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
    ]
    return Compose(transform_list)