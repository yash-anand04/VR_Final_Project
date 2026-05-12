"""
Data loading and preprocessing module for DeepFashion dataset
Phase 2: Data Preparation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from config import DATASET_CONFIG, DATA_PROCESSING, MODEL_CONFIG
from utils import setup_logger

logger = setup_logger(__name__)

class DeepFashionDataset(Dataset):
    """
    DeepFashion In-Shop Clothes Retrieval dataset loader
    """
    
    def __init__(
        self,
        image_dir: Path,
        eval_partition_file: Path,
        bbox_file: Optional[Path] = None,
        description_file: Optional[Path] = None,
        partition: str = "gallery",  # "train", "query", "gallery"
        transform=None,
    ):
        """
        Args:
            image_dir: Path to image directory
            eval_partition_file: Path to evaluation partition file
            bbox_file: Path to bounding box annotations
            description_file: Path to descriptions JSON
            partition: Which partition to load ("train", "query", "gallery")
            transform: Image transforms to apply
        """
        self.image_dir = Path(image_dir)
        self.partition = partition
        # Keep transform optional so CLIP preprocessing can be centralized.
        self.transform = transform
        
        # Load evaluation partition
        self.image_paths = []
        self.item_ids = []
        self._load_eval_partition(eval_partition_file)
        
        # Load bounding boxes if provided
        self.bboxes = {}
        if bbox_file:
            self._load_bboxes(bbox_file)
        
        # Load descriptions if provided
        self.descriptions = {}
        if description_file:
            self._load_descriptions(description_file)
        
        logger.info(f"Loaded {len(self.image_paths)} images for partition: {partition}")
    
    def _get_default_transform(self):
        """Default image transformation pipeline"""
        clip_size = MODEL_CONFIG["clip"]["image_size"]
        return transforms.Compose([
            transforms.Resize((clip_size, clip_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_eval_partition(self, eval_partition_file: Path) -> None:
        """Load evaluation partition file"""
        eval_partition_file = Path(eval_partition_file)
        
        with open(eval_partition_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Skip header lines
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            image_name = parts[0]
            image_rel = Path(image_name)
            if image_rel.parts and image_rel.parts[0].lower() == "img":
                image_rel = Path(*image_rel.parts[1:])
            item_id = parts[1]
            partition = parts[2]
            
            if partition == self.partition:
                image_path = self.image_dir / image_rel
                if image_path.exists():
                    self.image_paths.append(str(image_path))
                    self.item_ids.append(item_id)
    
    def _load_bboxes(self, bbox_file: Path) -> None:
        """Load bounding box annotations"""
        bbox_file = Path(bbox_file)
        
        with open(bbox_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Skip header lines
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 6:
                continue
            
            image_name = parts[0]
            # Format: image_name clothes_type pose_type x1 y1 x2 y2
            bbox = {
                'clothes_type': int(parts[1]),
                'pose_type': int(parts[2]),
                'x1': int(parts[3]),
                'y1': int(parts[4]),
                'x2': int(parts[5]),
                'y2': int(parts[6]) if len(parts) > 6 else int(parts[5]),
            }
            self.bboxes[image_name] = bbox
    
    def _load_descriptions(self, description_file: Path) -> None:
        """Load item descriptions from JSON"""
        description_file = Path(description_file)
        
        with open(description_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_descriptions = json.load(f)

        normalized = {}

        if isinstance(raw_descriptions, dict):
            for key, value in raw_descriptions.items():
                if isinstance(value, dict):
                    item_id = str(value.get("item_id", key))
                    desc = value.get("description") or value.get("item_description") or ""
                else:
                    item_id = str(key)
                    desc = str(value)
                normalized[item_id] = desc

        elif isinstance(raw_descriptions, list):
            for entry in raw_descriptions:
                if isinstance(entry, dict):
                    item_id = str(entry.get("item_id") or entry.get("item") or "")
                    desc = entry.get("description") or entry.get("item_description") or ""
                    if isinstance(desc, list):
                        desc = " ".join(str(chunk).strip() for chunk in desc if chunk)
                    if item_id:
                        normalized[item_id] = str(desc)

        self.descriptions = normalized
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item by index
        Returns dict with:
            - image: transformed image tensor
            - image_path: original image path
            - item_id: item ID from dataset
            - description: item description if available
            - bbox: bounding box if available
        """
        image_path = self.image_paths[idx]
        item_id = self.item_ids[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Get description if available
        description = self.descriptions.get(str(item_id), "")
        
        # Get bbox if available
        image_name = Path(image_path).name
        bbox = self.bboxes.get(image_name, None)
        
        return {
            'image': image,
            'image_path': image_path,
            'item_id': item_id,
            'description': description,
            'bbox': bbox,
        }


class DatasetManager:
    """Manager for dataset preparation and splitting"""
    
    def __init__(self, dataset_config: Dict = None):
        """
        Args:
            dataset_config: Dataset configuration dictionary
        """
        self.config = dataset_config or DATASET_CONFIG
        self.image_dir = Path(self.config['image_dir'])
        self.eval_partition_file = Path(self.config['eval_partition_file'])
        self.bbox_file = Path(self.config['bbox_file']) if self.config.get('bbox_file') else None
        self.description_file = Path(self.config.get('description_file')) if self.config.get('description_file') else None
    
    def get_dataset(
        self,
        partition: str = "gallery",
        transform=None,
    ) -> DeepFashionDataset:
        """
        Get dataset for a specific partition
        
        Args:
            partition: "train", "query", or "gallery"
            transform: Optional image transforms
        
        Returns:
            DeepFashionDataset instance
        """
        return DeepFashionDataset(
            image_dir=self.image_dir,
            eval_partition_file=self.eval_partition_file,
            bbox_file=self.bbox_file,
            description_file=self.description_file,
            partition=partition,
            transform=transform,
        )
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        transform=None,
    ) -> Dict[str, DataLoader]:
        """
        Get dataloaders for all partitions
        
        Args:
            batch_size: Batch size for dataloader
            num_workers: Number of workers for data loading
            shuffle_train: Whether to shuffle training data
            transform: Optional image transforms
        
        Returns:
            Dictionary with dataloaders for "train", "query", "gallery"
        """
        dataloaders = {}
        
        for partition in ["train", "query", "gallery"]:
            dataset = self.get_dataset(partition=partition, transform=transform)
            
            dataloaders[partition] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=(shuffle_train and partition == "train"),
                pin_memory=True,
            )
            
            logger.info(f"Created {partition} dataloader with {len(dataset)} samples")
        
        return dataloaders
    
    def get_item_groups(self) -> Dict[int, List[str]]:
        """
        Get mapping of item_id to image paths for positive pair identification
        
        Returns:
            Dictionary mapping item_id -> list of image paths
        """
        item_groups = {}
        
        with open(self.eval_partition_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            image_name = parts[0]
            image_rel = Path(image_name)
            if image_rel.parts and image_rel.parts[0].lower() == "img":
                image_rel = Path(*image_rel.parts[1:])
            item_id = parts[1]
            
            image_path = str(self.image_dir / image_rel)
            
            if image_path.exists():
                if item_id not in item_groups:
                    item_groups[item_id] = []
                item_groups[item_id].append(image_path)
        
        logger.info(f"Found {len(item_groups)} unique items")
        return item_groups
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {}
        
        for partition in ["train", "query", "gallery"]:
            dataset = self.get_dataset(partition=partition)
            stats[partition] = {
                "num_samples": len(dataset),
                "num_unique_items": len(set(dataset.item_ids)),
            }
        
        return stats


def create_dataloaders(batch_size: int = 32, num_workers: int = 4):
    """Convenience function to create all dataloaders"""
    manager = DatasetManager()
    return manager.get_dataloaders(batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    # Test dataset loading
    logger.info("Testing DeepFashion dataset loading...")
    
    manager = DatasetManager()
    stats = manager.get_statistics()
    logger.info(f"Dataset statistics: {stats}")
    
    # Test dataloader
    dataloaders = manager.get_dataloaders(batch_size=4, num_workers=0)
    
    for partition, dataloader in dataloaders.items():
        batch = next(iter(dataloader))
        logger.info(f"{partition} batch - Image: {batch['image'].shape}, Items: {batch['item_id']}")
