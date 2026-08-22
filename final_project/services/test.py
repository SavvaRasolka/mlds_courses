import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel

model_id = "THUDM/CogVideoX-5b-I2V"

transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)


pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
)

# pipe.to("cuda")
pipe.enable_sequential_cpu_offload()

prompt = "White lines swirl. Camera moves toward person and animal. Dont change the shape of human and animal face. Keep the style of picture"

image = load_image("final_project/services/beliberda2.jpg")

video = pipe(image=image, prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]

export_to_video(video, "output.mp4", fps=8)
