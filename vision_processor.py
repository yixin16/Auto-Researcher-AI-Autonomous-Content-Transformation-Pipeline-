# vision_processor.py
import cv2
import os
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

logger = logging.getLogger(__name__)
FRAME_DIR = Path("outputs/frames")
FRAME_DIR.mkdir(parents=True, exist_ok=True)

class VisionProcessor:
    def __init__(self):
        # We use Moondream because it's tiny (1.6B params) and runs fast alongside LLMs
        self.model_id = "vikhyatk/moondream2" 
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None:
            logger.info("👁️ Loading Vision Model (Moondream)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def unload_model(self):
        """Free up VRAM for the Image Generator"""
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None

    def extract_frames(self, video_path, interval=45):
        """Extracts 1 frame every 45 seconds."""
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frames = []
        count = 0
        
        while True:
            success, image = vidcap.read()
            if not success: break
            
            # Save frame every 'interval' seconds
            if count % (int(fps) * interval) == 0:
                path = FRAME_DIR / f"frame_{count}.jpg"
                cv2.imwrite(str(path), image)
                frames.append(str(path))
            count += 1
        vidcap.release()
        return frames

    def analyze_frame(self, image_path):
        self.load_model()
        image = Image.open(image_path)
        enc_image = self.model.encode_image(image)
        # Moondream specific prompting
        desc = self.model.answer_question(enc_image, "Describe the diagrams, charts, or text in this image detailedly.", self.tokenizer)
        return f"[VISUAL CONTEXT]: {desc}"