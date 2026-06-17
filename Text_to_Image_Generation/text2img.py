import argparse
import os
import time

import torch


# Model registry: friendly name -> (HF repo id, pipeline class name)
MODELS = {
    "flux-schnell": ("black-forest-labs/FLUX.1-schnell", "FluxPipeline"),
    "flux-dev":     ("black-forest-labs/FLUX.1-dev",     "FluxPipeline"),
    "sdxl":         ("stabilityai/stable-diffusion-xl-base-1.0", "StableDiffusionXLPipeline"),
    "sd15":         ("runwayml/stable-diffusion-v1-5",   "StableDiffusionPipeline"),
}

# Sensible default sampling params per model family.
DEFAULTS = {
    "flux-schnell": dict(steps=4,  guidance=0.0, height=1024, width=1024),
    "flux-dev":     dict(steps=28, guidance=3.5, height=1024, width=1024),
    "sdxl":         dict(steps=30, guidance=5.0, height=1024, width=1024),
    "sd15":         dict(steps=50, guidance=7.5, height=512,  width=512),
}


def load_pipeline(model_key: str, device: str, dtype: torch.dtype):
    repo_id, pipe_cls_name = MODELS[model_key]
    import diffusers

    pipe_cls = getattr(diffusers, pipe_cls_name)
    print(f"[load] {model_key} -> {repo_id} ({pipe_cls_name}, {dtype})")

    pipe = pipe_cls.from_pretrained(repo_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Speed: faster attention/matmul on Hopper. Safe no-ops if unsupported.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return pipe


def generate(pipe, model_key, prompt, args, generator):
    p = DEFAULTS[model_key]
    steps = args.steps if args.steps is not None else p["steps"]
    guidance = args.guidance if args.guidance is not None else p["guidance"]
    height = args.height if args.height is not None else p["height"]
    width = args.width if args.width is not None else p["width"]

    call = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=generator,
    )

    # FLUX pipelines don't accept negative_prompt the same way; we are only passing it for SD.
    if model_key.startswith("sd") and args.negative:
        call["negative_prompt"] = args.negative
    if model_key == "flux-dev" or model_key == "flux-schnell":
        call["max_sequence_length"] = 512 if model_key == "flux-dev" else 256

    t0 = time.time()
    image = pipe(**call).images[0]
    print(f"[gen] steps={steps} guidance={guidance} {width}x{height} "
          f"in {time.time() - t0:.1f}s")
    return image


def main():
    ap = argparse.ArgumentParser(description="Text-to-Image (SD / FLUX)")
    ap.add_argument("--model", choices=list(MODELS), default="flux-schnell")
    ap.add_argument("--prompt", type=str, help="single prompt")
    ap.add_argument("--prompt-file", type=str, help="file with one prompt per line")
    ap.add_argument("--negative", type=str, default="", help="negative prompt (SD only)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="outputs/image.png",
                    help="output path for a single prompt")
    ap.add_argument("--out-dir", type=str, default="outputs",
                    help="output directory for batch / prompt-file")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    if not args.prompt and not args.prompt_file:
        ap.error("provide --prompt or --prompt-file")

    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    pipe = load_pipeline(args.model, args.device, dtype)

    # Collect prompts
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    else:
        prompts = [args.prompt]

    if len(prompts) == 1 and not args.prompt_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        img = generate(pipe, args.model, prompts[0], args, gen)
        img.save(args.out)
        print(f"[save] {args.out}")
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        for i, prompt in enumerate(prompts):
            gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
            img = generate(pipe, args.model, prompt, args, gen)
            out = os.path.join(args.out_dir, f"image_{i:03d}.png")
            img.save(out)
            print(f"[save] {out}  ::  {prompt[:60]}")


if __name__ == "__main__":
    main()
