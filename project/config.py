"""
Configuration file for the Visual Product Search Engine project
Defines all parameters and paths used across the system
"""

from pathlib import Path

# Workspace and project roots
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = Path(__file__).resolve().parent

# Canonical artifact paths (single source of truth)
DATA_DIR = WORKSPACE_ROOT / "Dataset"
MODEL_DIR = WORKSPACE_ROOT / "models"
EMBEDDING_DIR = WORKSPACE_ROOT / "embeddings"
INDEX_DIR = WORKSPACE_ROOT / "index"
RESULTS_DIR = WORKSPACE_ROOT / "results"
LOGS_DIR = WORKSPACE_ROOT / "logs"
CROPS_DIR = WORKSPACE_ROOT / "crops"
CAPTIONS_DIR = WORKSPACE_ROOT / "captions"
DETECTION_RESULTS_DIR = WORKSPACE_ROOT / "detection_results"

# Create directories if they don't exist
for directory in [
    MODEL_DIR,
    EMBEDDING_DIR,
    INDEX_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    CROPS_DIR,
    CAPTIONS_DIR,
    DETECTION_RESULTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "name": "DeepFashion_InShop",
    "image_dir": DATA_DIR / "Img",
    "annotation_dir": DATA_DIR / "Anno",
    "eval_dir": DATA_DIR / "Eval",
    "bbox_file": DATA_DIR / "Anno" / "list_bbox_inshop.txt",
    "item_file": DATA_DIR / "Anno" / "list_item_inshop.txt",
    "eval_partition_file": DATA_DIR / "Eval" / "list_eval_partition.txt",
    "description_file": DATA_DIR / "Anno" / "list_description_inshop.json",
}

# Model configuration
MODEL_CONFIG = {
    "yolo": {
        "model_name": "yolov8m.pt",
        "device": "cuda",  # or "cpu"
        "conf_threshold": 0.5,
        "crop_conf_threshold": 0.5,
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "device": "cuda",  # or "cpu"
        "image_size": 224,
        "finetuned_weights": MODEL_DIR / "clip_finetuned.pt",
    },
    "blip2": {
        "model_name": "Salesforce/blip2-opt-2.7b",
        "device": "cuda",  # or "cpu"
    },
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "alpha": 0.5,  # Fusion weight: image_emb weight (1-alpha for text_emb)
    "embedding_dim": 512,  # CLIP base32 embedding dimension
    "normalize": True,  # L2 normalization
}

# FAISS Index configuration
INDEX_CONFIG = {
    "index_type": "IVF",  # Options: "IVF", "Flat"
    "metric": "IP",  # Options: "L2", "IP" (inner product)
    "n_clusters": 100,  # For IVF index
    "training_size": 10000,  # Samples for index training
}

# Evaluation configuration
EVAL_CONFIG = {
    "k_values": [5, 10, 15],  # Top-K values for evaluation
    "metrics": ["recall", "ndcg", "map"],
}

# Data processing configuration
DATA_PROCESSING = {
    "image_size": (256, 256),
    "batch_size": 32,
    "num_workers": 4,
    "random_seed": 42,
}

# Training configuration
TRAINING_CONFIG = {
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "warmup_steps": 500,
    "gradient_accumulation_steps": 1,
    "pairs_per_epoch": 10000,
    "pos_ratio": 0.5,
    "margin": 0.2,
    "train_last_n_layers": 2,
    "log_every": 50,
    "checkpoint_path": MODEL_DIR / "clip_finetuned.pt",
}

# Experiment configuration
EXPERIMENTS = {
    "A": {
        "name": "CLIP_Only",
        "use_text": False,
        "alpha": 1.0,
        "fine_tune": False,
    },
    "B": {
        "name": "CLIP_BLIP2_No_Finetune",
        "use_text": True,
        "alpha": 0.5,
        "fine_tune": False,
    },
    "C": {
        "name": "Fine_tuned_CLIP_BLIP2",
        "use_text": True,
        "alpha": 0.5,
        "fine_tune": True,
    },
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "project.log",
}
