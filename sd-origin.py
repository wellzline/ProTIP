from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt_ = "a photograph of an astronaut riding a horse on the moon with a flower in his hands."
prompt_ = "A blue bench under a tree in the park surrounded by red leaves."
prompt_ = "A red ? ball on . green grass ! under a blue ! ! sky?"
prompt_ = "a photograph of an astronaut riding a horse on the moon with a flower in his hands."
# image = pipe(prompt).images[0]  
# image.save("astronaut_rides_horse.png")
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

num_images = 6
prompt = [prompt_] * num_images
generator = torch.Generator("cuda").manual_seed(1024)
images = pipe(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator).images
grid = image_grid(images, rows=2, cols=3)
grid.save(f"output/astronaut_rides_horse.png")



