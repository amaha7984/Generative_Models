import os
import torch
import argparse
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.utils import save_image
from models.ema import EMA
from model_wrapper import ModelWrapper
from flow_matching.solver.ode_solver import ODESolver
from train import instantiate_model, load_model
from edm_time_discretization import get_time_discretization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CFGScaledModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(self, x, t, cfg_scale=0.0, label=None):
        t = torch.zeros(x.shape[0], device=x.device) + t
        with torch.cuda.amp.autocast(), torch.no_grad():
            if cfg_scale != 0.0 and label is not None:
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
                result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
            else:
                result = self.model(x, t, extra={"label": label} if label is not None else {})
        self.nfe_counter += 1
        return result.to(dtype=torch.float32)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config
    model = instantiate_model(
        architecture=args.dataset,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )
    model.to(device)

    # Load trained weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=None, lr_schedule=None)

    # EMA model wrapper
    cfg_model = CFGScaledModel(model)
    cfg_model.train(False)

    # Set up ODE solver
    solver = ODESolver(velocity_model=cfg_model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generation loop
    total = args.num_images
    batch_size = args.batch_size
    image_size = args.image_size
    num_generated = 0


    while num_generated < total:
        x_0 = torch.randn((batch_size, 3, image_size, image_size), device=device)

        if args.edm_schedule:
            time_grid = get_time_discretization(nfes=args.nfe)
        else:
            time_grid = torch.tensor([0.0, 1.0], device=device)

        samples = solver.sample(
            time_grid=time_grid,
            x_init=x_0,
            method=args.ode_method,
            return_intermediates=False,
            atol=args.atol,
            rtol=args.rtol,
            step_size=args.step_size,
            label=None,
            cfg_scale=args.cfg_scale,
        )

        samples = torch.clamp(samples * 0.5 + 0.5, min=0.0, max=1.0)
        samples = torch.floor(samples * 255)
        samples = samples.to(torch.float32) / 255.0

        # Convert to uint8 and save each image individually
        images_np = (
            (samples * 255.0)
            .clip(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        
        for i, image_np in enumerate(images_np):
            image_path = output_dir / f"gen_{num_generated + i:05d}.png"
            Image.fromarray(image_np, "RGB").save(image_path)
            logger.info(f"Saved {image_path}")

        num_generated += samples.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flow Matching Image Generation")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--discrete_flow_matching", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--edm_schedule", action="store_true")
    parser.add_argument("--nfe", type=int, default=30)
    parser.add_argument("--ode_method", type=str, default="rk4")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint dir")
    args = parser.parse_args()

    main(args)
