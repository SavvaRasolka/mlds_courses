import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel

_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        model_id = "THUDM/CogVideoX-5b-I2V"

        transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
        text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
        vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)

        _model_instance = CogVideoXImageToVideoPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=torch.float16,
        )
        _model_instance.enable_sequential_cpu_offload()
    return _model_instance