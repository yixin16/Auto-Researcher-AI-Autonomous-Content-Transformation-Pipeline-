from diffusers import AutoPipelineForText2Image
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
IMG_DIR = Path("outputs/generated_images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

class ImageGenerator:
    def __init__(self):
        self.pipe = None
        self.model_id = "stabilityai/sdxl-turbo"

    def load_model(self):
        if self.pipe is None:
            logger.info("🎨 Loading SDXL-Turbo...")
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id, torch_dtype=torch.float16, variant="fp16"
            )
            if torch.cuda.is_available(): self.pipe.to("cuda")

    def unload_model(self):
        if self.pipe:
            del self.pipe
            torch.cuda.empty_cache()
            self.pipe = None

    def generate(self, prompt, filename_suffix):
        self.load_model()
        full_prompt = f"cinematic, professional presentation, {prompt}, 4k, highly detailed"
        try:
            image = self.pipe(prompt=full_prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
            path = IMG_DIR / f"gen_{filename_suffix}.png"
            image.save(path)
            return str(path)
        except Exception as e:
            logger.error(f"Image Gen Failed: {e}")
            return None