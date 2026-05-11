"""
FAISS/HNSW Index creation and management
Phase 3: Offline Pipeline - Step 4
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from utils import setup_logger

logger = setup_logger(__name__)


class FAISSIndex:
    """FAISS-based vector index for similarity search"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "IVF",
        metric: str = "L2",
        n_clusters: int = 100,
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ("IVF", "Flat", "HNSW")
            metric: Distance metric ("L2" or "IP" for inner product)
            n_clusters: Number of clusters for IVF
        """
        import faiss
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.n_clusters = n_clusters
        
        # Create index
        if index_type == "IVF":
            quantizer = faiss.IndexFlatL2(embedding_dim) if metric == "L2" else faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
            self.index.nprobe = 10  # Number of clusters to search
        
        elif index_type == "Flat":
            self.index = faiss.IndexFlatL2(embedding_dim) if metric == "L2" else faiss.IndexFlatIP(embedding_dim)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.metadata = []
        self.is_trained = False
        
        logger.info(f"Created {index_type} index with metric {metric}")
    
    def train(self, embeddings: np.ndarray) -> None:
        """
        Train the index (required for IVF)
        
        Args:
            embeddings: Training embeddings (N, D)
        """
        if self.index_type == "IVF":
            if not self.index.is_trained:
                logger.info(f"Training index with {len(embeddings)} samples...")
                embeddings = embeddings.astype(np.float32)
                self.index.train(embeddings)
                self.is_trained = True
                logger.info("Index training complete")
    
    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Add embeddings to index
        
        Args:
            embeddings: Embeddings to add (N, D)
            metadata: List of metadata dicts for each embedding
        """
        embeddings = embeddings.astype(np.float32)
        
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[Dict], List[float]]:
        """
        Search for nearest neighbors
        
        Args:
            query_embedding: Query embedding (D,) or (1, D)
            k: Number of neighbors to return
        
        Returns:
            (list of metadata dicts, list of distances)
        """
        if query_embedding.dim() == 1 or (isinstance(query_embedding, np.ndarray) and query_embedding.ndim == 1):
            query_embedding = np.expand_dims(query_embedding, axis=0)
        
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results_metadata = [self.metadata[idx] for idx in indices[0]]
        results_distances = distances[0].tolist()
        
        return results_metadata, results_distances
    
    def search_batch(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[List[List[Dict]], List[List[float]]]:
        """Search for batch of queries"""
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        
        distances, indices = self.index.search(query_embeddings, k)
        
        all_metadata = []
        all_distances = []
        
        for i in range(len(query_embeddings)):
            metadata = [self.metadata[idx] for idx in indices[i]]
            dist = distances[i].tolist()
            
            all_metadata.append(metadata)
            all_distances.append(dist)
        
        return all_metadata, all_distances
    
    def save(self, index_path: Path, metadata_path: Optional[Path] = None) -> None:
        """Save index to disk"""
        import faiss
        
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved index to {index_path}")
        
        if metadata_path:
            metadata_path = Path(metadata_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self, index_path: Path, metadata_path: Optional[Path] = None) -> None:
        """Load index from disk"""
        import faiss
        
        index_path = Path(index_path)
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded index from {index_path}")
        
        if metadata_path:
            metadata_path = Path(metadata_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata entries")


class IndexBuilder:
    """Build and manage the embedding index"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "IVF",
        metric: str = "L2",
        n_clusters: int = 100,
    ):
        """Initialize index builder"""
        self.index = FAISSIndex(
            embedding_dim=embedding_dim,
            index_type=index_type,
            metric=metric,
            n_clusters=n_clusters,
        )
    
    def build_from_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        training_size: int = 10000,
    ) -> None:
        """
        Build index from embeddings
        
        Args:
            embeddings: All embeddings (N, D)
            metadata: Metadata for each embedding
            training_size: Number of samples to use for training
        """
        # Train index
        if self.index.index_type == "IVF":
            train_indices = np.random.choice(len(embeddings), min(training_size, len(embeddings)), replace=False)
            train_embeddings = embeddings[train_indices]
            self.index.train(train_embeddings)
        
        # Add all embeddings
        self.index.add(embeddings, metadata)
    
    def build_from_generator(
        self,
        embedding_generator,
        total_samples: int,
        batch_size: int = 1000,
        training_size: int = 10000,
    ) -> None:
        """
        Build index from a generator (memory-efficient for large datasets)
        
        Args:
            embedding_generator: Generator yielding (embeddings, metadata) tuples
            total_samples: Total number of embeddings
            batch_size: Batch size for processing
            training_size: Number of samples to use for training
        """
        # Collect training data
        if self.index.index_type == "IVF":
            train_embeddings = []
            train_count = 0
            
            for embeddings, metadata in embedding_generator():
                if train_count < training_size:
                    remaining = training_size - train_count
                    add_count = min(len(embeddings), remaining)
                    train_embeddings.append(embeddings[:add_count])
                    train_count += add_count
                
                if train_count >= training_size:
                    break
            
            if train_embeddings:
                train_embeddings = np.vstack(train_embeddings)
                self.index.train(train_embeddings)
        
        # Add all embeddings
        for embeddings, metadata in embedding_generator():
            self.index.add(embeddings, metadata)
    
    def get_index(self) -> FAISSIndex:
        """Get the built index"""
        return self.index


# Index utilities
def compute_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute similarity between two sets of embeddings
    
    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (M, D)
        metric: Similarity metric ("cosine", "l2", "dot")
    
    Returns:
        Similarity matrix (N, M)
    """
    if metric == "cosine":
        # L2 normalize
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        normalized1 = embeddings1 / (norm1 + 1e-6)
        normalized2 = embeddings2 / (norm2 + 1e-6)
        
        similarity = np.dot(normalized1, normalized2.T)
    
    elif metric == "l2":
        # Euclidean distance
        similarity = -np.sqrt(np.sum((embeddings1[:, None, :] - embeddings2[None, :, :]) ** 2, axis=2))
    
    elif metric == "dot":
        similarity = np.dot(embeddings1, embeddings2.T)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity


if __name__ == "__main__":
    # Test index creation
    logger.info("Testing index creation...")
    
    # Create dummy embeddings
    n_samples = 1000
    embedding_dim = 512
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Create dummy metadata
    metadata = [{"image_id": i, "item_id": i % 100} for i in range(n_samples)]
    
    # Build index
    builder = IndexBuilder(embedding_dim=embedding_dim)
    builder.build_from_embeddings(embeddings, metadata)
    
    logger.info("[OK] Index creation completed")
    
    # Test search
    query = embeddings[0:1]
    results_meta, results_dist = builder.get_index().search(query, k=10)
    logger.info(f"[OK] Search returned {len(results_meta)} results")
