"""
Phase 6: Evaluation
Compute Recall@K, NDCG@K, and mAP@K for retrieval results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from utils import setup_logger, DatasetManager, compute_recall_at_k, compute_ndcg_at_k, compute_map_at_k
from scripts.search import ProductSearchEngine
from config import EVAL_CONFIG, RESULTS_DIR

logger = setup_logger(__name__)


def evaluate_partition(
    partition: str = "query",
    search_partition: str = "gallery",
    k_values: List[int] = None,
    device: str = None,
    alpha: float = 1.0,
    use_text: bool = False,
    use_rerank: bool = False,
    use_crop: bool = True,
    use_finetuned: bool = True,
    output_tag: str = None,
) -> Dict:
    k_values = k_values or EVAL_CONFIG["k_values"]
    manager = DatasetManager()
    dataset = manager.get_dataset(partition=partition)
    engine = ProductSearchEngine(partition=search_partition, device=device, use_finetuned=use_finetuned)

    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics.update({f"map@{k}": [] for k in k_values})

    for sample in tqdm(dataset, desc=f"Evaluating {partition}"):
        query_image = sample["image_path"]
        ground_truth_id = sample["item_id"]
        results = engine.search(
            query_image,
            k=max(k_values),
            alpha=alpha,
            query_text=None if not use_text else sample.get("description") or "fashion clothing item",
            use_crop=use_crop,
        )
        if use_rerank:
            results = engine.rerank(query_image, results, top_n=max(k_values), use_crop=use_crop)
        retrieved_ids = [item.get("item_id") for item in results]
        relevant_ids = {ground_truth_id}

        for k in k_values:
            metrics[f"recall@{k}"].append(compute_recall_at_k(retrieved_ids, ground_truth_id, k))
            metrics[f"ndcg@{k}"].append(compute_ndcg_at_k(retrieved_ids, ground_truth_id, k))
            metrics[f"map@{k}"].append(compute_map_at_k(retrieved_ids, relevant_ids, k))

    summary = {}
    for key, values in metrics.items():
        summary[key] = float(sum(values) / max(len(values), 1))

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_tag or partition
    output_file = output_dir / f"metrics_{suffix}.json"
    with open(output_file, "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Saved metrics to %s", output_file)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--partition", default="query", choices=["query", "gallery"], help="Evaluation query partition")
    parser.add_argument("--search-partition", default="gallery", choices=["gallery", "query"], help="Index partition")
    parser.add_argument("--k", type=int, nargs="*", default=None, help="K values to evaluate")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--alpha", type=float, default=1.0, help="Image/text fusion weight")
    parser.add_argument("--use-text", action="store_true", help="Use text fusion during evaluation")
    parser.add_argument("--use-rerank", action="store_true", help="Use BLIP-2 ITM reranking")
    parser.add_argument("--no-crop", action="store_true", help="Use original query image (skip YOLO crop)")
    parser.add_argument("--no-finetune", action="store_true", help="Don't load fine-tuned CLIP weights")
    args = parser.parse_args()

    results = evaluate_partition(
        args.partition,
        search_partition=args.search_partition,
        k_values=args.k,
        device=args.device,
        alpha=args.alpha,
        use_text=args.use_text,
        use_rerank=args.use_rerank,
        use_crop=not args.no_crop,
        use_finetuned=not args.no_finetune,
    )
    print(json.dumps(results, indent=2))
