"""
Phase 8: Experiment runner
Compare CLIP-only, CLIP+BLIP-2, and fine-tuned setups.
"""

import argparse
import json
from pathlib import Path

from utils import setup_logger
from scripts.evaluate import evaluate_partition
from scripts.embed import generate_embeddings
from scripts.index import build_index
from scripts.train import train_clip
from config import TRAINING_CONFIG, MODEL_CONFIG

logger = setup_logger(__name__)


EXPERIMENTS = {
    "A": {"alpha": 1.0, "use_text": False, "fine_tune": False, "rerank": False},
    "B": {"alpha": 0.5, "use_text": True, "fine_tune": False, "rerank": True},
    "C": {"alpha": 0.5, "use_text": True, "fine_tune": True, "rerank": True},
}


def run_experiments(partition: str = "query", search_partition: str = "gallery", device: str = None):
    results = {}
    for name, settings in EXPERIMENTS.items():
        logger.info("Running experiment %s", name)

        if settings["fine_tune"]:
            checkpoint_path = Path(TRAINING_CONFIG["checkpoint_path"])
            if not checkpoint_path.exists():
                logger.info("Fine-tuned weights missing; starting training for experiment %s", name)
                train_clip(epochs=TRAINING_CONFIG["num_epochs"], device=device)

        generate_embeddings(
            partition="gallery",
            alpha=settings["alpha"],
            use_text=settings["use_text"],
            use_finetuned=settings["fine_tune"],
            device=device,
        )
        generate_embeddings(
            partition="query",
            alpha=settings["alpha"],
            use_text=settings["use_text"],
            use_finetuned=settings["fine_tune"],
            device=device,
        )

        build_index(partition=search_partition)

        results[name] = evaluate_partition(
            partition,
            search_partition=search_partition,
            device=device,
            alpha=settings["alpha"],
            use_text=settings["use_text"],
            use_rerank=settings["rerank"],
            use_finetuned=settings["fine_tune"],
            output_tag=name,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval experiments")
    parser.add_argument("--partition", default="query", choices=["query", "gallery"])
    parser.add_argument("--search-partition", default="gallery", choices=["gallery", "query"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output = run_experiments(args.partition, search_partition=args.search_partition, device=args.device)
    print(json.dumps(output, indent=2))
