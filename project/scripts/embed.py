"""
Phase 3 - Step 3: Generate CLIP Embeddings
Extract image and text embeddings using CLIP
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils import (
    setup_logger,
    get_device,
    CLIPEmbedder,
    EmbeddingFusion,
    DatasetManager,
    save_embeddings,
)
from config import DATASET_CONFIG, MODEL_CONFIG, EMBEDDING_CONFIG, EMBEDDING_DIR
from config import CAPTIONS_DIR, DETECTION_RESULTS_DIR

logger = setup_logger(__name__)


def _normalize_caption_key(path_str: str) -> str:
    """Normalize caption keys to current dataset image paths."""
    if not path_str:
        return ""

    path = Path(path_str)
    if not path.is_absolute():
        return str(DATASET_CONFIG["image_dir"] / path)

    parts = path.parts
    for idx, part in enumerate(parts):
        if part.lower() == "img":
            rel = Path(*parts[idx + 1:])
            return str(DATASET_CONFIG["image_dir"] / rel)

    return str(path)


def generate_embeddings(
    partition: str = "gallery",
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = None,
    batch_size: int = 32,
    use_text: bool = True,
    alpha: float = 0.5,
    use_crops: bool = True,
    use_finetuned: bool = True,
):
    """
    Generate embeddings for dataset images
    
    Args:
        partition: Dataset partition
        model_name: CLIP model name
        device: Device to use
        batch_size: Batch size for processing
        use_text: Whether to fuse with text embeddings
        alpha: Weight for image embedding (1-alpha for text)
    """
    
    device = device or MODEL_CONFIG["clip"]["device"]
    crop_conf_threshold = MODEL_CONFIG["yolo"].get(
        "crop_conf_threshold",
        MODEL_CONFIG["yolo"]["conf_threshold"],
    )
    logger.info(f"Generating embeddings for {partition} partition...")
    
    # Initialize models
    embedder = CLIPEmbedder(model_name=model_name, device=device, use_finetuned=use_finetuned)
    
    # Load captions if needed
    captions_data = {}
    if use_text:
        captions_file = CAPTIONS_DIR / f"{partition}_captions.json"
        if captions_file.exists():
            with open(captions_file, 'r') as f:
                captions_data = json.load(f)
            captions_index = captions_data.get("captions", {})
            if captions_index:
                normalized_captions = {}
                for key, value in captions_index.items():
                    normalized_key = _normalize_caption_key(key)
                    if normalized_key:
                        normalized_captions.setdefault(normalized_key, value)
                captions_data["captions"] = normalized_captions
            logger.info(f"Loaded {len(captions_data.get('captions', {}))} captions")
    
    # Load dataset
    manager = DatasetManager()
    dataset = manager.get_dataset(partition=partition)

    # Load detection results for crop mapping (optional)
    crop_map = {}
    if use_crops:
        detection_file = DETECTION_RESULTS_DIR / f"{partition}_detection.json"
        if detection_file.exists():
            with open(detection_file, "r") as f:
                detection_results = json.load(f)

            for det in detection_results.get("detections", []):
                image_path = det.get("image_path")
                crop_path = det.get("crop_path")
                confidence = det.get("confidence")
                use_crop_flag = det.get("use_crop")
                if use_crop_flag is None:
                    detected_flag = det.get("detected")
                    if detected_flag is False:
                        use_crop_flag = False
                    elif confidence is not None:
                        use_crop_flag = float(confidence) >= crop_conf_threshold
                    else:
                        use_crop_flag = bool(crop_path)

                if image_path:
                    crop_map[image_path] = {
                        "crop_path": crop_path,
                        "confidence": confidence,
                        "use_crop": bool(use_crop_flag),
                    }
    
    # Create output directory
    embedding_dir = EMBEDDING_DIR / partition
    embedding_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect embeddings and metadata
    all_embeddings = []
    metadata_list = []
    
    logger.info(f"Processing {len(dataset)} images...")
    
    for idx in tqdm(range(0, len(dataset), batch_size), desc="Generating embeddings"):
        batch_end = min(idx + batch_size, len(dataset))
        batch_idx = list(range(idx, batch_end))
        
        # Load batch
        batch_images = []
        batch_metadata = []
        
        for i in batch_idx:
            sample = dataset[i]
            caption = None
            image_key = _normalize_caption_key(sample['image_path'])
            if use_text and captions_data:
                caption_info = captions_data.get('captions', {}).get(image_key)
                if caption_info:
                    caption = caption_info.get('caption')

            image_path = sample['image_path']
            crop_info = crop_map.get(image_path) if use_crops else None
            crop_path = crop_info.get("crop_path") if crop_info else None
            crop_confidence = crop_info.get("confidence") if crop_info else None
            use_crop_for_image = bool(crop_info and crop_info.get("use_crop"))

            if use_crop_for_image and crop_path and Path(crop_path).exists():
                image_to_open = crop_path
                used_crop = True
            else:
                image_to_open = image_path
                used_crop = False

            from PIL import Image
            image = Image.open(image_to_open).convert("RGB")
            batch_images.append(image)
            
            batch_metadata.append({
                'image_path': sample['image_path'],
                'crop_path': crop_path if crop_path and Path(crop_path).exists() else None,
                'used_crop': used_crop,
                'crop_confidence': crop_confidence,
                'item_id': sample['item_id'],
                'description': sample['description'],
                'caption': caption,
            })
        
        # Get image embeddings
        image_embeddings = embedder.get_batch_image_embeddings(batch_images)
        
        # Get text embeddings if needed
        if use_text and captions_data:
            text_embeddings = []
            for i, metadata in enumerate(batch_metadata):
                image_key = _normalize_caption_key(metadata['image_path'])
                caption_info = captions_data.get('captions', {}).get(image_key)
                
                if caption_info and caption_info.get('caption'):
                    text_emb = embedder.get_text_embedding(caption_info['caption'])
                    text_embeddings.append(text_emb[0])
                else:
                    # Use description as fallback
                    desc = metadata['description'] or "fashion clothing item"
                    text_emb = embedder.get_text_embedding(desc)
                    text_embeddings.append(text_emb[0])
            
            text_embeddings = np.vstack(text_embeddings)
            
            # Fuse embeddings
            fused_embeddings = EmbeddingFusion.fuse_embeddings(
                image_embeddings,
                text_embeddings,
                alpha=alpha
            )
            all_embeddings.append(fused_embeddings)
        else:
            all_embeddings.append(image_embeddings)
        
        metadata_list.extend(batch_metadata)
    
    # Combine all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Save embeddings
    embedding_file = embedding_dir / "embeddings.npy"
    metadata_file = embedding_dir / "metadata.json"
    
    save_embeddings(all_embeddings, embedding_file)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    
    logger.info(f"Generated {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
    logger.info(f"Embeddings saved to: {embedding_file}")
    logger.info(f"Metadata saved to: {metadata_file}")
    
    return all_embeddings, metadata_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings")
    parser.add_argument("--partition", type=str, default="gallery", choices=["train", "query", "gallery"],
                       help="Dataset partition to process")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-text", action="store_true", help="Don't use text embeddings")
    parser.add_argument("--alpha", type=float, default=0.5, help="Image embedding weight (0-1)")
    parser.add_argument("--no-crops", action="store_true", help="Don't use detection crops for embeddings")
    parser.add_argument("--no-finetune", action="store_true", help="Don't load fine-tuned CLIP weights")
    
    args = parser.parse_args()
    
    generate_embeddings(
        partition=args.partition,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        use_text=not args.no_text,
        alpha=args.alpha,
        use_crops=not args.no_crops,
        use_finetuned=not args.no_finetune,
    )
