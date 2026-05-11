"""Utils module"""
from .utils import (
    setup_logger,
    get_device,
    ensure_dir_exists,
    get_image_paths,
    normalize_embeddings,
    normalize_embeddings_torch,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_map_at_k,
    save_embeddings,
    load_embeddings,
    save_index_metadata,
    load_index_metadata,
)

from .data_loader import (
    DeepFashionDataset,
    DatasetManager,
    create_dataloaders,
)

from .models import (
    YOLODetector,
    CLIPEmbedder,
    BLIP2Captioner,
    BLIP2ITMScorer,
    EmbeddingFusion,
)

from .index import (
    FAISSIndex,
    IndexBuilder,
    compute_similarity,
)

__all__ = [
    "setup_logger",
    "get_device",
    "ensure_dir_exists",
    "get_image_paths",
    "normalize_embeddings",
    "normalize_embeddings_torch",
    "compute_recall_at_k",
    "compute_ndcg_at_k",
    "compute_map_at_k",
    "save_embeddings",
    "load_embeddings",
    "save_index_metadata",
    "load_index_metadata",
    "DeepFashionDataset",
    "DatasetManager",
    "create_dataloaders",
    "YOLODetector",
    "CLIPEmbedder",
    "BLIP2Captioner",
    "BLIP2ITMScorer",
    "EmbeddingFusion",
    "FAISSIndex",
    "IndexBuilder",
    "compute_similarity",
]
