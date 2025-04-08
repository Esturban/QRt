# qr_art.py  ── drop this in the same folder as your notebook
import torch, qrcode
from PIL import Image
from functools import lru_cache
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)


# ---------------------------- helpers ---------------------------- #
def _resize(img: Image.Image, res: int = 768) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    k = res / min(h, w)
    w, h = int(round(w * k / 64)) * 64, int(round(h * k / 64)) * 64
    return img.resize((w, h), Image.LANCZOS)

_SAMPLERS = {
    "DPM++ Karras SDE": lambda cfg: DPMSolverMultistepScheduler.from_config(
        cfg, use_karras=True, algorithm_type="sde-dpmsolver++"
    ),
    "DPM++ Karras":      lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, use_karras=True),
    "Heun":              lambda cfg: HeunDiscreteScheduler.from_config(cfg),
    "Euler":             lambda cfg: EulerDiscreteScheduler.from_config(cfg),
    "DDIM":              lambda cfg: DDIMScheduler.from_config(cfg),
    "DEIS":              lambda cfg: DEISMultistepScheduler.from_config(cfg),
}

# ------------------------ pipeline loader ------------------------ #
@lru_cache(maxsize=1)
def _load_pipe(device: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe.to(device)
    # uncomment if xformers is available
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

# Add this function to handle .webp conversion
def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.format == 'WEBP':
        img = img.convert('RGBA')  # Convert to RGBA or any other format you need
    return img

# Resize function for condition images
def resize_for_condition_image(input_image: Image, resolution: int) -> Image.Image:
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

# ----------------------- public entry‑point ---------------------- #
def generate_qr_art(
    qr_code_content: str | None = None,
    prompt: str = "",
    negative_prompt: str = "ugly, disfigured, low quality, blurry, nsfw",
    *,
    init_image: Image.Image | None = None,
    qr_code_image: Image.Image | None = None,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    sampler: str = "DPM++ Karras SDE",
    seed: int | None = None,
    width: int = 768,
    height: int = 768,
    device: str | None = None,
    num_inference_steps: int = 40,
) -> Image.Image:
    """
    Generate a stylised QR‑code image with Stable Diffusion + ControlNet.

    Either *qr_code_content* **or** *qr_code_image* must be supplied.
    """
    if not prompt:
        raise ValueError("prompt is required")
    if not qr_code_content and qr_code_image is None:
        raise ValueError("Either qr_code_content or qr_code_image must be supplied")

    # pick device automatically if not given
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = _load_pipe(device)

    # Resize init_image if provided
    if init_image is not None:
        init_image = load_image(init_image)
        init_image = resize_for_condition_image(init_image, 768)

    # ---------------------------------------------------------------- QR code
    if qr_code_image is None:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)
        qr_code_image = qr.make_image(fill_color="black", back_color="white")
    qr_code_image = _resize(qr_code_image, max(width, height))

    # ---------------------------------------------------------------- Sampler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.scheduler = _SAMPLERS[sampler](pipe.scheduler.config)

    # ---------------------------------------------------------------- Diffuse
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    result = pipe(
        prompt           = prompt,
        negative_prompt  = negative_prompt,
        image            = init_image,
        control_image    = qr_code_image,
        width            = width,
        height           = height,
        guidance_scale   = guidance_scale,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        strength         = strength,
        num_inference_steps = num_inference_steps,
        generator        = g,
    )
    return result.images[0]
