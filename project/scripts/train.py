"""
Phase 5: CLIP fine-tuning with contrastive image-pair training.
"""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup

from utils import setup_logger, DatasetManager
from config import MODEL_CONFIG, TRAINING_CONFIG

logger = setup_logger(__name__)


class PairDataset(Dataset):
    """On-the-fly sampler that returns positive/negative image pairs."""

    def __init__(self, item_to_images: Dict[str, List[str]], pairs_per_epoch: int, pos_ratio: float):
        self.item_to_images = {k: v for k, v in item_to_images.items() if v}
        self.item_ids = list(self.item_to_images.keys())
        self.pos_item_ids = [k for k, v in self.item_to_images.items() if len(v) >= 2]
        self.pairs_per_epoch = pairs_per_epoch
        self.pos_ratio = pos_ratio

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _sample_positive(self):
        item_id = random.choice(self.pos_item_ids)
        img1, img2 = random.sample(self.item_to_images[item_id], 2)
        return img1, img2, 1

    def _sample_negative(self):
        item_a, item_b = random.sample(self.item_ids, 2)
        img1 = random.choice(self.item_to_images[item_a])
        img2 = random.choice(self.item_to_images[item_b])
        return img1, img2, 0

    def __getitem__(self, idx: int):
        if self.pos_item_ids and random.random() < self.pos_ratio:
            img1, img2, label = self._sample_positive()
        else:
            img1, img2, label = self._sample_negative()

        from PIL import Image
        image_a = Image.open(img1).convert("RGB")
        image_b = Image.open(img2).convert("RGB")

        return {"image_a": image_a, "image_b": image_b, "label": label}


def collate_pairs(batch):
    images_a = [item["image_a"] for item in batch]
    images_b = [item["image_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return images_a, images_b, labels


def resolve_device(requested: str) -> str:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return requested


def set_trainable_vision_layers(model: CLIPModel, train_last_n_layers: int) -> None:
    for param in model.text_model.parameters():
        param.requires_grad = False

    for param in model.vision_model.parameters():
        param.requires_grad = False

    if train_last_n_layers is None or train_last_n_layers <= 0:
        return

    layers = model.vision_model.encoder.layers
    train_last_n_layers = min(train_last_n_layers, len(layers))
    for layer in layers[-train_last_n_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    if hasattr(model, "visual_projection"):
        for param in model.visual_projection.parameters():
            param.requires_grad = True
    if hasattr(model.vision_model, "post_layernorm"):
        for param in model.vision_model.post_layernorm.parameters():
            param.requires_grad = True


def contrastive_loss(similarity: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    pos_loss = (1.0 - similarity) * labels
    neg_loss = torch.relu(similarity - margin) * (1.0 - labels)
    return (pos_loss + neg_loss).mean()


def train_clip(
    epochs: int = None,
    device: str = "cuda",
    batch_size: int = None,
    pairs_per_epoch: int = None,
    pos_ratio: float = None,
    margin: float = None,
    train_last_n_layers: int = None,
    checkpoint_path: Path = None,
):
    device = resolve_device(device)

    epochs = epochs or TRAINING_CONFIG["num_epochs"]
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]
    pairs_per_epoch = pairs_per_epoch or TRAINING_CONFIG["pairs_per_epoch"]
    pos_ratio = pos_ratio if pos_ratio is not None else TRAINING_CONFIG["pos_ratio"]
    margin = margin if margin is not None else TRAINING_CONFIG["margin"]
    train_last_n_layers = train_last_n_layers or TRAINING_CONFIG["train_last_n_layers"]
    checkpoint_path = Path(checkpoint_path or TRAINING_CONFIG["checkpoint_path"])

    logger.info("Loading CLIP model for fine-tuning...")
    model_name = MODEL_CONFIG["clip"]["model_name"]
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.train()

    set_trainable_vision_layers(model, train_last_n_layers)

    manager = DatasetManager()
    dataset = manager.get_dataset(partition="train")
    item_to_images: Dict[str, List[str]] = {}
    for image_path, item_id in zip(dataset.image_paths, dataset.item_ids):
        item_to_images.setdefault(str(item_id), []).append(image_path)

    if not item_to_images:
        logger.error("No training images found. Check dataset paths and partitions.")
        return None

    pair_dataset = PairDataset(
        item_to_images=item_to_images,
        pairs_per_epoch=pairs_per_epoch,
        pos_ratio=pos_ratio,
    )
    dataloader = DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pairs,
    )

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    grad_accum = max(1, TRAINING_CONFIG["gradient_accumulation_steps"])
    num_update_steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
    total_steps = num_update_steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_training_steps=total_steps,
    )

    log_every = TRAINING_CONFIG["log_every"]
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (images_a, images_b, labels) in enumerate(dataloader, start=1):
            inputs_a = processor(images=images_a, return_tensors="pt")
            inputs_b = processor(images=images_b, return_tensors="pt")
            inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
            inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
            labels = labels.to(device)

            features_a = model.get_image_features(**inputs_a)
            features_b = model.get_image_features(**inputs_b)
            features_a = F.normalize(features_a, p=2, dim=1)
            features_b = F.normalize(features_b, p=2, dim=1)

            similarity = F.cosine_similarity(features_a, features_b)
            loss = contrastive_loss(similarity, labels, margin) / grad_accum
            loss.backward()

            if step % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum
            global_step += 1

            if global_step % log_every == 0:
                logger.info("Epoch %d Step %d/%d - Loss: %.4f", epoch, step, len(dataloader), epoch_loss / step)

        logger.info("Epoch %d completed. Avg loss: %.4f", epoch, epoch_loss / max(len(dataloader), 1))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Saved fine-tuned CLIP weights to %s", checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP embeddings")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["num_epochs"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"])
    parser.add_argument("--pairs-per-epoch", type=int, default=TRAINING_CONFIG["pairs_per_epoch"])
    parser.add_argument("--pos-ratio", type=float, default=TRAINING_CONFIG["pos_ratio"])
    parser.add_argument("--margin", type=float, default=TRAINING_CONFIG["margin"])
    parser.add_argument("--train-last-n-layers", type=int, default=TRAINING_CONFIG["train_last_n_layers"])
    parser.add_argument("--checkpoint-path", type=str, default=str(TRAINING_CONFIG["checkpoint_path"]))
    args = parser.parse_args()

    train_clip(
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        pairs_per_epoch=args.pairs_per_epoch,
        pos_ratio=args.pos_ratio,
        margin=args.margin,
        train_last_n_layers=args.train_last_n_layers,
        checkpoint_path=args.checkpoint_path,
    )
