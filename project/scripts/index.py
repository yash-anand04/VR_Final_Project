"""
Phase 3 - Step 4: Build FAISS Index
Create searchable index from embeddings
"""

import argparse
import json
from pathlib import Path
import numpy as np
from utils import (
    setup_logger,
    load_embeddings,
    IndexBuilder,
)
from config import EMBEDDING_DIR, INDEX_DIR, INDEX_CONFIG

logger = setup_logger(__name__)


def build_index(
    partition: str = "gallery",
    index_type: str = None,
    metric: str = None,
    n_clusters: int = None,
):
    """
    Build FAISS index from embeddings
    
    Args:
        partition: Dataset partition
        index_type: Type of index ("IVF", "Flat")
        metric: Distance metric ("L2", "IP")
        n_clusters: Number of clusters for IVF
    """
    
    index_type = index_type or INDEX_CONFIG["index_type"]
    metric = metric or INDEX_CONFIG["metric"]
    n_clusters = n_clusters or INDEX_CONFIG["n_clusters"]
    
    logger.info(f"Building {index_type} index for {partition} partition...")
    logger.info(f"Metric: {metric}, Clusters: {n_clusters}")
    
    # Load embeddings
    embedding_file = EMBEDDING_DIR / partition / "embeddings.npy"
    metadata_file = EMBEDDING_DIR / partition / "metadata.json"
    
    if not embedding_file.exists() or not metadata_file.exists():
        logger.error(f"Embeddings not found for {partition}")
        logger.error(f"Expected: {embedding_file} and {metadata_file}")
        return False
    
    embeddings = load_embeddings(embedding_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Build index
    builder = IndexBuilder(
        embedding_dim=embeddings.shape[1],
        index_type=index_type,
        metric=metric,
        n_clusters=n_clusters,
    )
    
    builder.build_from_embeddings(
        embeddings,
        metadata,
        training_size=INDEX_CONFIG.get("training_size", 10000),
    )
    
    # Save index
    index_path = INDEX_DIR / f"{partition}_index.faiss"
    metadata_path = INDEX_DIR / f"{partition}_metadata.pkl"
    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    builder.get_index().save(index_path, metadata_path)
    
    # Save config
    config = {
        "partition": partition,
        "index_type": index_type,
        "metric": metric,
        "n_clusters": n_clusters,
        "embedding_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
    }
    
    config_path = INDEX_DIR / f"{partition}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("[OK] Index created successfully")
    logger.info(f"  Index saved to: {index_path}")
    logger.info(f"  Metadata saved to: {metadata_path}")
    logger.info(f"  Config saved to: {config_path}")
    logger.info(f"  Total samples: {embeddings.shape[0]}")
    
    return True


def build_all_indices(
    index_type: str = None,
    metric: str = None,
    n_clusters: int = None,
):
    """Build indices for all partitions"""
    
    for partition in ["gallery", "query"]:
        logger.info(f"\n{'='*60}")
        success = build_index(
            partition=partition,
            index_type=index_type,
            metric=metric,
            n_clusters=n_clusters,
        )
        
        if not success:
            logger.warning(f"Failed to build index for {partition}")
    
    logger.info(f"\n{'='*60}")
    logger.info("All indices built!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--partition", type=str, default=None, choices=["train", "query", "gallery"],
                       help="Dataset partition to index (if None, build all)")
    parser.add_argument("--index-type", type=str, default=None, choices=["IVF", "Flat"],
                       help="Type of index")
    parser.add_argument("--metric", type=str, default=None, choices=["L2", "IP"],
                       help="Distance metric")
    parser.add_argument("--clusters", type=int, default=None, help="Number of clusters for IVF")
    
    args = parser.parse_args()
    
    if args.partition:
        build_index(
            partition=args.partition,
            index_type=args.index_type,
            metric=args.metric,
            n_clusters=args.clusters,
        )
    else:
        build_all_indices(
            index_type=args.index_type,
            metric=args.metric,
            n_clusters=args.clusters,
        )
