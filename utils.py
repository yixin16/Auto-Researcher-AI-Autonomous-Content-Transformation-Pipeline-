# utils.py
import os
from pathlib import Path
import whisper
import subprocess
import logging

logger = logging.getLogger(__name__)

# Define Directories
AUDIO_DIR = Path("outputs/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def download_audio(youtube_url: str) -> str:
    """Downloads audio only (m4a) from YouTube."""
    # Uses hash of URL to prevent duplicate downloads of same video
    import hashlib
    url_hash = hashlib.md5(youtube_url.encode()).hexdigest()
    filename = f"{url_hash}.m4a"
    output_path = AUDIO_DIR / filename

    if output_path.exists():
        logger.info(f"Audio found in cache: {output_path}")
        return str(output_path)

    # Command to download audio
    cmd = [
        "yt-dlp", "--quiet", 
        "--extract-audio", 
        "--audio-format", "m4a", 
        "-o", str(output_path), 
        youtube_url
    ]
    
    try:
        logger.info(f"Downloading audio from {youtube_url}...")
        subprocess.run(cmd, check=True)
        return str(output_path)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise e

def transcribe_audio_whisper(audio_path: str, model_size="base") -> str:
    logger.info(f"Transcribing {audio_path} using {model_size} model...")
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""

def chunk_text(text: str, max_len=2500) -> list:
    """Splits text into manageable chunks for the LLM."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(text_len, start + max_len)
        # Try to cut at the last period to keep sentences whole
        cut = text.rfind(".", start, end)
        if cut <= start: 
            cut = end
            
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut + 1
        
    return chunks