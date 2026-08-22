import uuid
import time
from celery import shared_task
from diffusers.utils import export_to_video, load_image

from services.model import get_model

@shared_task(bind=True)
def generate_video_task(self, image_path, text_prompt):
    model = get_model()
    image = load_image(image_path)
    video_path = f"C:/Users/Sava/work/final/final_project/media/temp/{uuid.uuid4()}.mp4"
    relative_path = video_path[44:]
    video = model(image=image, prompt=text_prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]
    export_to_video(video, video_path, fps=8)
    return relative_path

def dummy_model():
    video_path = f"C:/Users/Sava/work/final/final_project/media/temp/walk.mp4"
    relative_path = video_path[44:]
    seconds = 10
    for i in range(seconds):
        time.sleep(1)
    return relative_path
