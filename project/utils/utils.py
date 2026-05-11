"""
Utility functions and helpers for the project
"""

import os
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
from config import LOGGING_CONFIG, DATA_PROCESSING

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """Set up logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_CONFIG["level"])
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(LOGGING_CONFIG["log_file"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Device management
def get_device(device_name: Optional[str] = None) -> torch.device:
    """Get appropriate device (cuda/cpu)"""
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device_name)
    logger = setup_logger(__name__)
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return device

# Path utilities
def ensure_dir_exists(path: Path) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_image_paths(directory: Path, extensions: Tuple[str, ...] = ('.jpg', '.png', '.jpeg')) -> List[Path]:
    """Recursively get all image paths from directory"""
    image_paths = []
    directory = Path(directory)
    
    for ext in extensions:
        image_paths.extend(directory.rglob(f"*{ext}"))
        image_paths.extend(directory.rglob(f"*{ext.upper()}"))
    
    return sorted(list(set(image_paths)))  # Remove duplicates and sort

# Normalization utilities
def normalize_embeddings(embeddings: np.ndarray, norm: str = "l2") -> np.ndarray:
    """Normalize embeddings using specified norm"""
    if norm == "l2":
        return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
    elif norm == "l1":
        return embeddings / (np.sum(np.abs(embeddings), axis=1, keepdims=True) + 1e-6)
    else:
        return embeddings

def normalize_embeddings_torch(embeddings: torch.Tensor, norm: str = "l2") -> torch.Tensor:
    """Normalize embeddings using PyTorch"""
    if norm == "l2":
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    elif norm == "l1":
        return torch.nn.functional.normalize(embeddings, p=1, dim=1)
    else:
        return embeddings

# Image utilities
def create_image_grid(images: List[np.ndarray], cols: int = 5) -> np.ndarray:
    """Create a grid of images for visualization"""
    try:
        import cv2
    except ImportError:
        raise ImportError("cv2 required for image grid creation")
    
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        row = idx // cols
        col = idx % cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = cv2.resize(img, (w, h))
    
    return grid

# Metrics utilities
def compute_recall_at_k(retrieved_ids: np.ndarray, ground_truth_id: int, k: int) -> float:
    """Compute recall@k: 1 if ground_truth in top-k, else 0"""
    return 1.0 if ground_truth_id in retrieved_ids[:k] else 0.0

def compute_ndcg_at_k(retrieved_ids: np.ndarray, ground_truth_id: int, k: int) -> float:
    """Compute NDCG@k"""
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id == ground_truth_id:
            dcg = 1.0 / np.log2(i + 2)  # Position starts at 1
            break
    
    idcg = 1.0 / np.log2(2)  # Ideal DCG when relevant item at position 1
    ndcg = dcg / idcg
    
    return ndcg

def compute_map_at_k(retrieved_ids: np.ndarray, relevant_ids: set, k: int) -> float:
    """Compute MAP@k"""
    num_relevant = 0
    ap = 0.0
    
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in relevant_ids:
            num_relevant += 1
            ap += num_relevant / (i + 1)
    
    if len(relevant_ids) == 0:
        return 0.0
    
    return ap / min(len(relevant_ids), k)

# Tensor utilities
def move_to_device(data, device: torch.device):
    """Recursively move tensors to device"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data

# File I/O utilities
def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    """Save embeddings to disk"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

def load_embeddings(input_path: Path) -> np.ndarray:
    """Load embeddings from disk"""
    return np.load(input_path)

def save_index_metadata(metadata: dict, output_path: Path) -> None:
    """Save index metadata to JSON"""
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_index_metadata(input_path: Path) -> dict:
    """Load index metadata from JSON"""
    import json
    with open(input_path, 'r') as f:
        return json.load(f)
