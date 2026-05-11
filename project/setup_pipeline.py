"""
Complete pipeline orchestration script
Runs all phases sequentially or selectively
"""

import argparse
import sys
from pathlib import Path
import torch
from utils import setup_logger
from config import EMBEDDING_DIR, INDEX_DIR

logger = setup_logger(__name__)


def resolve_device(requested: str) -> str:
    """Resolve runtime device with CUDA availability fallback."""
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return requested


def run_phase_3_offline_pipeline(
    partitions: list = None,
    use_text: bool = True,
    alpha: float = 0.5,
    batch_size: int = 32,
    device: str = "cuda",
    skip_detection: bool = False,
    skip_caption: bool = False,
):
    """
    Run complete offline pipeline (Phase 3)
    
    Args:
        partitions: List of partitions to process
        use_text: Whether to use text embeddings
        alpha: Image embedding weight
        batch_size: Batch size
        device: Device to use
        skip_detection: Skip YOLO detection
        skip_caption: Skip caption generation
    """
    
    partitions = partitions or ["gallery", "query"]
    device = resolve_device(device)
    
    logger.info("="*70)
    logger.info("VISUAL PRODUCT SEARCH ENGINE - PHASE 3: OFFLINE PIPELINE")
    logger.info("="*70)
    
    for partition in partitions:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing partition: {partition.upper()}")
        logger.info(f"{'='*70}")
        
        # Step 1: Detection
        if not skip_detection:
            logger.info(f"\n[Step 1/4] Running YOLO detection...")
            try:
                from scripts.detect import detect_products
                detect_products(partition=partition, device=device, save_crops=True)
                logger.info("[OK] Detection completed")
            except Exception as e:
                logger.error(f"[ERROR] Detection failed: {e}")
                sys.exit(1)
        
        # Step 2: Captioning
        if not skip_caption and use_text:
            logger.info(f"\n[Step 2/4] Generating captions...")
            try:
                from scripts.caption import generate_captions
                generate_captions(partition=partition, device=device, load_from_detection=(not skip_detection))
                logger.info("[OK] Captioning completed")
            except Exception as e:
                logger.error(f"[ERROR] Captioning failed: {e}")
                sys.exit(1)
        
        # Step 3: Embeddings
        logger.info(f"\n[Step 3/4] Generating embeddings...")
        try:
            from scripts.embed import generate_embeddings
            generate_embeddings(
                partition=partition,
                batch_size=batch_size,
                use_text=use_text,
                alpha=alpha,
                device=device,
            )
            logger.info("[OK] Embedding generation completed")
        except Exception as e:
            logger.error(f"[ERROR] Embedding generation failed: {e}")
            sys.exit(1)
        
        # Step 4: Indexing
        logger.info(f"\n[Step 4/4] Building FAISS index...")
        try:
            from scripts.index import build_index
            build_index(partition=partition)
            logger.info("[OK] Index building completed")
        except Exception as e:
            logger.error(f"[ERROR] Index building failed: {e}")
            sys.exit(1)
    
    logger.info(f"\n{'='*70}")
    logger.info("[OK] PHASE 3 OFFLINE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"{'='*70}")
    
    # Print summary
    logger.info("\nGenerated files:")
    for partition in partitions:
        emb_file = EMBEDDING_DIR / partition / "embeddings.npy"
        idx_file = INDEX_DIR / f"{partition}_index.faiss"
        
        if emb_file.exists():
            logger.info(f"  [{partition}] Embeddings: {emb_file}")
        if idx_file.exists():
            logger.info(f"  [{partition}] Index: {idx_file}")


def run_phase_4_search(query_image: str, partition: str = "gallery", k: int = 10, alpha: float = 1.0, device: str = "cuda"):
    """Run the online retrieval pipeline for a single query image."""
    from scripts.search import run_search
    device = resolve_device(device)

    results = run_search(query_image, partition=partition, k=k, alpha=alpha, device=device)
    logger.info("Top-%d results:", k)
    for rank, item in enumerate(results, start=1):
        logger.info("%d. %s (score=%.4f)", rank, item.get("image_path", "unknown"), item.get("score", 0.0))


def run_phase_5_training(epochs: int = 10, device: str = "cuda"):
    """Run the CLIP fine-tuning scaffold."""
    from scripts.train import train_clip
    device = resolve_device(device)

    train_clip(epochs=epochs, device=device)


def run_phase_6_evaluation(partition: str = "query", search_partition: str = "gallery", device: str = "cuda"):
    """Run retrieval evaluation and persist summary metrics."""
    from scripts.evaluate import evaluate_partition
    device = resolve_device(device)

    metrics = evaluate_partition(partition=partition, search_partition=search_partition, device=device)
    logger.info("Evaluation metrics: %s", metrics)


def run_phase_8_experiments(partition: str = "query", search_partition: str = "gallery", device: str = "cuda"):
    """Run the experiment comparison suite."""
    from scripts.run_experiments import run_experiments
    device = resolve_device(device)

    results = run_experiments(partition=partition, search_partition=search_partition, device=device)
    logger.info("Experiment results: %s", results)


def main():
    parser = argparse.ArgumentParser(
        description="Visual Product Search Engine - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for all partitions
  python setup_pipeline.py --phase 3
  
  # Run only for gallery partition
  python setup_pipeline.py --phase 3 --partition gallery
  
  # Run without text embeddings (CLIP only)
  python setup_pipeline.py --phase 3 --no-text --alpha 1.0
  
  # Use CPU
  python setup_pipeline.py --phase 3 --device cpu
        """
    )
    
    parser.add_argument(
        "--phase",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Phase to run (default: 3)",
    )
    
    parser.add_argument(
        "--partition",
        type=str,
        nargs="+",
        default=["gallery", "query"],
        choices=["train", "query", "gallery"],
        help="Dataset partitions to process",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Image embedding weight (0-1, default: 0.5)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Don't use text embeddings (CLIP only)",
    )
    
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip YOLO detection",
    )
    
    parser.add_argument(
        "--skip-caption",
        action="store_true",
        help="Skip caption generation",
    )

    parser.add_argument(
        "--query-image",
        type=str,
        default=None,
        help="Query image path for phase 4 search",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for phase 5 training",
    )

    parser.add_argument(
        "--search-partition",
        type=str,
        default="gallery",
        choices=["gallery", "query"],
        help="Index partition to search/evaluate",
    )
    
    args = parser.parse_args()
    
    if args.phase == 3:
        run_phase_3_offline_pipeline(
            partitions=args.partition,
            use_text=not args.no_text,
            alpha=args.alpha,
            batch_size=args.batch_size,
            device=args.device,
            skip_detection=args.skip_detection,
            skip_caption=args.skip_caption,
        )
    elif args.phase == 4:
        if not args.query_image:
            logger.error("Phase 4 requires --query-image")
            sys.exit(1)
        run_phase_4_search(
            query_image=args.query_image,
            partition=args.search_partition,
            k=10,
            alpha=args.alpha,
            device=args.device,
        )
    elif args.phase == 5:
        run_phase_5_training(epochs=args.epochs, device=args.device)
    elif args.phase == 6:
        run_phase_6_evaluation(partition="query", search_partition=args.search_partition, device=args.device)
    elif args.phase == 7:
        logger.info("Launch the Streamlit app with: streamlit run app/streamlit_app.py")
    elif args.phase == 8:
        run_phase_8_experiments(partition="query", search_partition=args.search_partition, device=args.device)
    else:
        logger.info(f"Phase {args.phase} not yet implemented")


if __name__ == "__main__":
    main()
