"""
Phase 4: Online Search
Retrieve similar products for a query image
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

from utils import setup_logger, CLIPEmbedder, YOLODetector, EmbeddingFusion, BLIP2ITMScorer
from config import EMBEDDING_DIR, INDEX_DIR, MODEL_CONFIG, EMBEDDING_CONFIG, INDEX_CONFIG

logger = setup_logger(__name__)


class ProductSearchEngine:
    """Online retrieval engine for query-by-image search."""

    def __init__(self, partition: str = "gallery", device: Optional[str] = None, use_finetuned: bool = True):
        self.partition = partition
        self.device = device or MODEL_CONFIG["clip"]["device"]
        self.embedder = CLIPEmbedder(device=self.device, use_finetuned=use_finetuned)
        self.detector = YOLODetector(device=self.device)
        self.itm_scorer = None
        self.itm_model_name = MODEL_CONFIG["blip2"]["model_name"]
        self.crop_conf_threshold = MODEL_CONFIG["yolo"].get(
            "crop_conf_threshold",
            MODEL_CONFIG["yolo"]["conf_threshold"],
        )
        self.index, self.metadata, self.metric = self._load_index(partition)

    def _load_index(self, partition: str):
        import faiss
        index_path = INDEX_DIR / f"{partition}_index.faiss"
        metadata_path = INDEX_DIR / f"{partition}_metadata.pkl"
        config_path = INDEX_DIR / f"{partition}_config.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        index = faiss.read_index(str(index_path))
        import pickle
        with open(metadata_path, "rb") as handle:
            metadata = pickle.load(handle)

        metric = INDEX_CONFIG.get("metric", "IP")
        if config_path.exists():
            try:
                with open(config_path, "r") as handle:
                    config = json.load(handle)
                metric = config.get("metric", metric)
            except Exception as exc:
                logger.warning("Failed to read index config: %s", exc)

        logger.info("Loaded index with %d items", index.ntotal)
        return index, metadata, metric

    def _prepare_query_image(self, image_path: str, use_crop: bool = True):
        if use_crop:
            cropped, _, detected, confidence = self.detector.detect_and_crop(
                image_path,
                return_bbox=True,
                return_pil=True,
                return_detection=True,
                return_confidence=True,
            )
            if detected and confidence >= self.crop_conf_threshold:
                return cropped

        from PIL import Image
        return Image.open(image_path).convert("RGB")

    def search(
        self,
        image_path: str,
        k: int = 10,
        alpha: float = 1.0,
        query_text: Optional[str] = None,
        use_crop: bool = True,
    ) -> List[Dict]:
        """Search for top-k matches using a query image."""
        query_image = self._prepare_query_image(image_path, use_crop=use_crop)
        image_embedding = self.embedder.get_image_embedding(query_image)

        if query_text:
            text_embedding = self.embedder.get_text_embedding(query_text)
            query_embedding = EmbeddingFusion.fuse_embeddings(image_embedding, text_embedding, alpha=alpha)
        else:
            query_embedding = image_embedding

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]

        distances, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            if self.metric == "L2":
                item["score"] = float(-score)
            else:
                item["score"] = float(score)
            results.append(item)

        return results

    def rerank(
        self,
        query_image_path: str,
        candidates: List[Dict],
        top_n: int = 10,
        use_crop: bool = True,
    ) -> List[Dict]:
        """Re-rank candidates using BLIP-2 ITM scoring."""
        if not candidates:
            return []

        if self.itm_scorer is None:
            try:
                self.itm_scorer = BLIP2ITMScorer(model_name=self.itm_model_name, device=self.device)
            except Exception as exc:
                logger.warning("BLIP-2 ITM scorer unavailable: %s", exc)
                return candidates[:top_n]

        query_image = self._prepare_query_image(query_image_path, use_crop=use_crop)

        for item in candidates:
            text = item.get("caption") or item.get("description") or "fashion clothing item"
            item["itm_score"] = self.itm_scorer.score(query_image, text)

        reranked = sorted(candidates, key=lambda item: item.get("itm_score", float("-inf")), reverse=True)
        return reranked[:top_n]


def run_search(
    image_path: str,
    partition: str = "gallery",
    k: int = 10,
    alpha: float = 1.0,
    device: Optional[str] = None,
    rerank: bool = True,
    use_crop: bool = True,
    use_finetuned: bool = True,
):
    engine = ProductSearchEngine(partition=partition, device=device, use_finetuned=use_finetuned)
    results = engine.search(image_path=image_path, k=k, alpha=alpha, use_crop=use_crop)
    if rerank:
        return engine.rerank(image_path, results, top_n=k, use_crop=use_crop)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query-by-image product search")
    parser.add_argument("--query-image", required=True, help="Path to query image")
    parser.add_argument("--partition", default="gallery", choices=["gallery", "query"], help="Index partition to search")
    parser.add_argument("--k", type=int, default=10, help="Top-K results")
    parser.add_argument("--alpha", type=float, default=1.0, help="Image/text fusion weight")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--no-rerank", action="store_true", help="Disable BLIP-2 ITM reranking")
    parser.add_argument("--no-crop", action="store_true", help="Use original image (skip YOLO crop)")
    parser.add_argument("--no-finetune", action="store_true", help="Don't load fine-tuned CLIP weights")
    args = parser.parse_args()

    results = run_search(
        args.query_image,
        partition=args.partition,
        k=args.k,
        alpha=args.alpha,
        device=args.device,
        rerank=not args.no_rerank,
        use_crop=not args.no_crop,
        use_finetuned=not args.no_finetune,
    )
    print(json.dumps(results, indent=2))
