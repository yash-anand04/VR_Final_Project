"""
Phase 3 - Step 2: Image Captioning with BLIP-2
Generate captions for detected products
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils import (
    setup_logger,
    DatasetManager,
)
from config import DATASET_CONFIG, MODEL_CONFIG, CAPTIONS_DIR, DETECTION_RESULTS_DIR

logger = setup_logger(__name__)

# Try to import BLIP2Captioner, but fall back to simple caption generator if it fails
try:
    from utils import BLIP2Captioner
    BLIP2_AVAILABLE = True
except Exception as e:
    logger.warning(f"BLIP-2 model not available: {e}. Using fallback caption generation.")
    BLIP2_AVAILABLE = False

def generate_simple_caption(item_id: str) -> str:
    """Generate a simple placeholder caption based on item ID"""
    return f"A fashion product (item {item_id})"


def _caption_key(image_path: str) -> str:
    """Store captions with paths relative to the dataset image root."""
    try:
        rel_path = Path(image_path).relative_to(DATASET_CONFIG["image_dir"])
        return str(rel_path)
    except Exception:
        return str(image_path)


def try_load_blip2_captioner(model_name: str, device: str):
    """Try to load BLIP-2 captioner, return None if it fails"""
    try:
        from utils import BLIP2Captioner
        logger.info(f"Loading BLIP-2 captioner...")
        captioner = BLIP2Captioner(model_name=model_name, device=device)
        return captioner
    except Exception as e:
        logger.warning(f"Failed to load BLIP-2 captioner: {e}. Will use simple captions.")
        return None

def generate_captions(
    partition: str = "gallery",
    model_name: str = "Salesforce/blip2-opt-2.7b",
    device: str = None,
    load_from_detection: bool = True,
    use_blip2: bool = True,
):
    """
    Generate captions for products
    
    Args:
        partition: Dataset partition
        model_name: BLIP-2 model name
        device: Device to use
        load_from_detection: Whether to use detected crop images
        use_blip2: Whether to try using BLIP2 (falls back to simple captions if it fails)
    """
    
    device = device or MODEL_CONFIG["blip2"]["device"]
    logger.info(f"Starting caption generation for {partition} partition...")
    
    # Initialize captioner if available
    captioner = None
    use_blip2_actual = use_blip2 and BLIP2_AVAILABLE
    
    if use_blip2_actual:
        try:
            from utils import BLIP2Captioner as Captioner
            captioner = Captioner(model_name=model_name, device=device)
        except Exception as e:
            logger.warning(f"Failed to initialize BLIP-2 captioner: {e}. Falling back to simple captions.")
            use_blip2_actual = False
    
    # Load detection results if available
    captions_data = {
        "partition": partition,
        "captions": {}
    }
    
    if load_from_detection:
        detection_file = DETECTION_RESULTS_DIR / f"{partition}_detection.json"
        
        if detection_file.exists():
            with open(detection_file, 'r') as f:
                detection_results = json.load(f)
            
            logger.info(f"Processing {len(detection_results['detections'])} detected crops...")
            
            # Try loading BLIP-2 first
            captioner = try_load_blip2_captioner(model_name=model_name, device=device) if not captioner else captioner
            
            captions_blip2 = 0
            captions_fallback = 0
            
            for det in tqdm(detection_results['detections'], desc="Generating captions"):
                item_id = det.get('item_id', 'unknown')
                image_path = det.get('image_path', '')
                caption_key = _caption_key(image_path)
                caption = None
                
                # Try BLIP-2 first if available
                if captioner is not None:
                    try:
                        crop_path = Path(det.get('crop_path', ''))
                        if crop_path.exists():
                            from PIL import Image
                            import torchvision.transforms as transforms
                            
                            image = Image.open(crop_path).convert('RGB')
                            to_tensor = transforms.ToTensor()
                            image_tensor = to_tensor(image)
                            caption = captioner.generate_caption(image_tensor)
                            if caption:
                                captions_blip2 += 1
                    except Exception as e:
                        logger.debug(f"BLIP-2 failed for {image_path}: {e}")
                        caption = None
                
                # Use fallback caption if BLIP-2 unavailable or failed
                if caption is None:
                    caption = generate_simple_caption(item_id)
                    captions_fallback += 1
                
                captions_data["captions"][caption_key] = {
                    "caption": caption,
                    "item_id": item_id,
                }
            
            logger.info(f"Captions generated - BLIP2: {captions_blip2}, Fallback: {captions_fallback}")
        else:
            logger.warning("Detection results not found, skipping crop-based captioning")
    
    # Save captions
    captions_file = CAPTIONS_DIR / f"{partition}_captions.json"
    captions_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(captions_file, 'w') as f:
        json.dump(captions_data, f, indent=4)
    
    logger.info(f"Generated captions for {len(captions_data['captions'])} images")
    logger.info(f"Captions saved to: {captions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions with BLIP-2")
    parser.add_argument("--partition", type=str, default="gallery", choices=["train", "query", "gallery"],
                       help="Dataset partition to process")
    parser.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b", help="BLIP-2 model name")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--use-original", action="store_true", help="Use original images instead of detected crops")
    
    args = parser.parse_args()
    
    generate_captions(
        partition=args.partition,
        model_name=args.model,
        device=args.device,
        load_from_detection=not args.use_original,
    )
