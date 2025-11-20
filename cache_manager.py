# cache_manager.py
import hashlib
import json
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

CACHE_DIR = Path("outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_path(url: str, step_name: str, identifier_hash: str) -> Path:
    """Creates a unique, short filename using hashes."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    # Clean step name to prevent filesystem errors
    safe_step = re.sub(r'[\\/*?:"<>|]', "", step_name)[:20]
    filename = f"{url_hash}_{safe_step}_{identifier_hash}.json"
    return CACHE_DIR / filename

def save_to_cache(path: Path, data: dict):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def load_from_cache(path: Path) -> dict | None:
    if path.exists():
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return None
    return None

def get_or_create(url: str, step_name: str, generation_function, *args, **kwargs):
    """
    Smart caching with Type Validation.
    """
    # Create hash from arguments
    arg_str = str(args) + str(kwargs)
    input_hash = hashlib.md5(arg_str.encode('utf-8')).hexdigest()
    
    cache_file = get_cache_path(url, step_name, input_hash)
    
    # Try to load from cache
    cached_data = load_from_cache(cache_file)
    
    if cached_data and 'content' in cached_data:
        content = cached_data['content']
        
        # --- CRITICAL FIX: Validate Data Type ---
        # If we are doing SWOT or Charts, we expect a Dict. If cache gives a String, it's stale.
        if "swot" in step_name.lower() or "chart" in step_name.lower():
            if isinstance(content, str):
                logger.warning(f"⚠️ Detected stale/corrupt cache for {step_name}. Regenerating...")
                # Fall through to generation below
            else:
                logger.info(f"Loaded '{step_name}' from cache.")
                return content
        else:
            logger.info(f"Loaded '{step_name}' from cache.")
            return content
    
    # Generate fresh
    logger.info(f"Generating '{step_name}'...")
    content = generation_function(*args, **kwargs)
    
    # Save
    save_to_cache(cache_file, {'content': content})
    return content