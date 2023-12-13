from diffusers import StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial

model_ckpt = "runwayml/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")

prompts = [
    "a photo of an astronaut riding a horse on mars"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device).manual_seed(1024)
images = sd_pipeline(prompts, num_images_per_prompt=1, output_type="np", generator=generator).images
print(images.shape)
# (6, 512, 512, 3)

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score: {sd_clip_score}")
# CLIP score: 35.7038


