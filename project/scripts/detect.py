"""
Phase 3 - Step 1: Object Detection with YOLO
Detect products in images and create cropped datasets
"""

import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import torch
from utils import (
    setup_logger,
    get_device,
    YOLODetector,
    DatasetManager,
)
from config import DATASET_CONFIG, MODEL_CONFIG, CROPS_DIR, DETECTION_RESULTS_DIR

logger = setup_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def detect_products(
    partition: str = "gallery",
    model_name: str = "yolov8m.pt",
    device: str = None,
    save_crops: bool = True,
):
    """
    Detect products in dataset images
    
    Args:
        partition: Dataset partition ("train", "query", "gallery")
        model_name: YOLO model name
        device: Device to use
        save_crops: Whether to save cropped images
    """
    
    device = device or MODEL_CONFIG["yolo"]["device"]
    crop_conf_threshold = MODEL_CONFIG["yolo"].get(
        "crop_conf_threshold",
        MODEL_CONFIG["yolo"]["conf_threshold"],
    )
    logger.info(f"Starting product detection for {partition} partition...")
    
    # Initialize detector
    detector = YOLODetector(model_name=model_name, device=device)
    
    # Load dataset
    manager = DatasetManager()
    dataset = manager.get_dataset(partition=partition)
    
    # Create output directory
    crop_dir = CROPS_DIR / partition
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    # Detection results
    detection_results = {
        "total_images": len(dataset),
        "detected": 0,
        "not_detected": 0,
        "low_confidence": 0,
        "errors": 0,
        "fallback_full_image": 0,
        "detections": []
    }
    
    # Process each image
    for idx, image_path in enumerate(dataset.image_paths):
        try:
            # Detect
            cropped, bbox, detected, confidence = detector.detect_and_crop(
                image_path,
                return_bbox=True,
                return_detection=True,
                return_confidence=True,
            )

            use_crop = bool(detected and confidence >= crop_conf_threshold)
            fallback_reason = None
            if not detected:
                fallback_reason = "no_detection"
            elif not use_crop:
                fallback_reason = "low_confidence"
            
            # Save cropped image if requested
            crop_path = None
            if save_crops:
                if detected:
                    # Convert tensor to image and save
                    import torchvision.transforms as transforms
                    from PIL import Image

                    to_pil = transforms.ToPILImage()
                    crop_image = to_pil(cropped)

                    # Create relative path for crop
                    rel_path = Path(image_path).relative_to(DATASET_CONFIG['image_dir'])
                    crop_path = crop_dir / rel_path.parent / (rel_path.stem + "_crop.jpg")
                    crop_path.parent.mkdir(parents=True, exist_ok=True)

                    crop_image.save(crop_path)

            detection_results["detections"].append({
                "image_path": image_path,
                "crop_path": str(crop_path) if crop_path else None,
                "bbox": bbox,
                "item_id": dataset.item_ids[idx],
                "detected": bool(detected),
                "confidence": float(confidence),
                "use_crop": bool(use_crop),
                "fallback_reason": fallback_reason,
            })

            if detected:
                detection_results["detected"] += 1
            else:
                detection_results["not_detected"] += 1

            if not use_crop:
                detection_results["fallback_full_image"] += 1
                if detected:
                    detection_results["low_confidence"] += 1
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} images")
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            detection_results["errors"] += 1
    
    # Save results
    results_path = DETECTION_RESULTS_DIR / f"{partition}_detection.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(results_path, 'w') as f:
            json.dump(detection_results, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Results successfully saved to: {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {results_path}: {e}")
        raise
    
    logger.info(f"\nDetection Results for {partition}:")
    logger.info(f"  Total: {detection_results['total_images']}")
    logger.info(f"  Detected: {detection_results['detected']}")
    logger.info(f"  Not Detected: {detection_results['not_detected']}")
    logger.info(f"  Fallback Full Image: {detection_results['fallback_full_image']}")
    logger.info(f"  Errors: {detection_results['errors']}")
    logger.info(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect products using YOLO")
    parser.add_argument("--partition", type=str, default="gallery", choices=["train", "query", "gallery"],
                       help="Dataset partition to process")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLO model name")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--no-save-crops", action="store_true", help="Don't save cropped images")
    
    args = parser.parse_args()
    
    detect_products(
        partition=args.partition,
        model_name=args.model,
        device=args.device,
        save_crops=not args.no_save_crops,
    )
