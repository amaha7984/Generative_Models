import argparse
import os
import time

import torch
from PIL import Image


# Model registry: friendly name -> (HF repo id, pipeline class name, mode)
# mode: "img2img" uses `strength`; "edit" is instruction-based (no strength).
MODELS = {
    "flux-img2img":    ("black-forest-labs/FLUX.1-dev",      "FluxImg2ImgPipeline",                 "img2img"),
    "flux-kontext":    ("black-forest-labs/FLUX.1-Kontext-dev", "FluxKontextPipeline",              "edit"),
    "sdxl-img2img":    ("stabilityai/stable-diffusion-xl-base-1.0", "StableDiffusionXLImg2ImgPipeline", "img2img"),
    "sd15-img2img":    ("runwayml/stable-diffusion-v1-5",     "StableDiffusionImg2ImgPipeline",      "img2img"),
    "instruct-pix2pix":("timbrooks/instruct-pix2pix",         "StableDiffusionInstructPix2PixPipeline", "edit"),
}

# Sensible default sampling params per model.
DEFAULTS = {
    "flux-img2img":    dict(steps=28, guidance=3.5, strength=0.7),
    "flux-kontext":    dict(steps=28, guidance=2.5, strength=None),
    "sdxl-img2img":    dict(steps=30, guidance=5.0, strength=0.6),
    "sd15-img2img":    dict(steps=50, guidance=7.5, strength=0.6),
    "instruct-pix2pix":dict(steps=30, guidance=7.5, strength=None),
}


def load_pipeline(model_key: str, device: str, dtype: torch.dtype):
    repo_id, pipe_cls_name, _ = MODELS[model_key]
    import diffusers

    pipe_cls = getattr(diffusers, pipe_cls_name)
    print(f"[load] {model_key} -> {repo_id} ({pipe_cls_name}, {dtype})")

    pipe = pipe_cls.from_pretrained(repo_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return pipe


def load_image(path: str, max_side: int | None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_side:
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
    return img


def generate(pipe, model_key, prompt, image, args, generator):
    p = DEFAULTS[model_key]
    mode = MODELS[model_key][2]
    steps = args.steps if args.steps is not None else p["steps"]
    guidance = args.guidance if args.guidance is not None else p["guidance"]

    call = dict(
        prompt=prompt,
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )

    if mode == "img2img":
        strength = args.strength if args.strength is not None else p["strength"]
        call["strength"] = strength

    # SD pipelines accept negative_prompt; FLUX ones do not.
    if model_key.startswith("sd") or model_key == "instruct-pix2pix":
        if args.negative:
            call["negative_prompt"] = args.negative
    if model_key == "instruct-pix2pix":
        call["image_guidance_scale"] = args.image_guidance

    t0 = time.time()
    out = pipe(**call).images[0]
    print(f"[gen] {model_key} steps={steps} guidance={guidance} "
          f"in {time.time() - t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser(description="Image-to-Image / text-guided editing (SD / FLUX)")
    ap.add_argument("--model", choices=list(MODELS), default="flux-kontext")
    ap.add_argument("--image", type=str, required=True, help="input image path")
    ap.add_argument("--prompt", type=str, required=True,
                    help="target description (img2img) or edit instruction (edit)")
    ap.add_argument("--negative", type=str, default="", help="negative prompt (SD only)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--strength", type=float, default=None,
                    help="img2img only: 0=keep input, 1=ignore input")
    ap.add_argument("--image-guidance", type=float, default=1.5,
                    help="instruct-pix2pix only: how much to preserve the input image")
    ap.add_argument("--max-side", type=int, default=1024, help="downscale input so longest side <= this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="outputs/edit.png")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    pipe = load_pipeline(args.model, args.device, dtype)

    image = load_image(args.image, args.max_side)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    out = generate(pipe, args.model, args.prompt, image, args, gen)
    out.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
