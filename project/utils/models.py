"""
Model wrappers and loaders for YOLO, CLIP, and BLIP-2
Phase 3: Offline Pipeline - Step 1 & 2 & 3
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Iterable
from PIL import Image
import torch.nn.functional as F
from utils import setup_logger, get_device, normalize_embeddings_torch
from config import MODEL_CONFIG, MODEL_DIR

logger = setup_logger(__name__)


def _resolve_device(requested_device: Optional[str]) -> str:
    """Resolve runtime device with CUDA availability fallback."""
    device = requested_device or "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device


class YOLODetector:
    """YOLO object detection wrapper for product detection"""
    
    def __init__(self, model_name: str = "yolov8m.pt", device: Optional[str] = None):
        """
        Args:
            model_name: YOLO model name
            device: Device to use ("cuda" or "cpu")
        """
        from ultralytics import YOLO
        
        self.device = _resolve_device(device or MODEL_CONFIG["yolo"]["device"])

        model_path = Path(model_name)
        if not model_path.exists():
            model_in_dir = MODEL_DIR / model_name
            model_in_root = MODEL_DIR.parent / model_name

            if model_in_dir.exists():
                model_path = model_in_dir
            elif model_in_root.exists():
                model_path = model_in_root
            else:
                # Let Ultralytics resolve/download by model name.
                model_path = Path(model_name)

        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.conf_threshold = MODEL_CONFIG["yolo"]["conf_threshold"]
        
        logger.info(f"Loaded YOLO model: {model_name} on {self.device}")
    
    def detect_and_crop(
        self,
        image_path: str,
        return_bbox: bool = False,
        return_pil: bool = False,
        return_detection: bool = False,
        return_confidence: bool = False,
    ) -> Tuple:
        """
        Detect product in image and return cropped region
        
        Args:
            image_path: Path to input image
            return_bbox: Whether to return bounding box coordinates
            return_confidence: Whether to return the detection confidence (requires return_detection)
        
        Returns:
            Cropped image tensor or (cropped image, bbox) if return_bbox=True
        """
        from PIL import Image

        if return_confidence and not return_detection:
            raise ValueError("return_confidence requires return_detection=True")
        
        # Run detection
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        
        detected = len(results[0].boxes) > 0
        best_confidence = 0.0
        if not detected:
            logger.debug(f"No products detected in {image_path}")
            # Return full image if no detection.
            image = Image.open(image_path).convert('RGB')

            if return_bbox:
                w, h = image.size
                bbox = (int(0), int(0), int(w), int(h))
                if return_pil:
                    if return_detection:
                        if return_confidence:
                            return image, bbox, detected, best_confidence
                        return image, bbox, detected
                    return image, bbox

                image_array = np.array(image)
                image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 255.0
                if return_detection:
                    if return_confidence:
                        return image_tensor, bbox, detected, best_confidence
                    return image_tensor, bbox, detected
                return image_tensor, bbox

            if return_pil:
                if return_detection:
                    if return_confidence:
                        return image, detected, best_confidence
                    return image, detected
                return image

            image_array = np.array(image)
            image_tensor = torch.tensor(image_array).permute(2, 0, 1).float() / 255.0
            if return_detection:
                if return_confidence:
                    return image_tensor, detected, best_confidence
                return image_tensor, detected
            return image_tensor
        
        # Get largest detection
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Select box with highest confidence
        best_idx = np.argmax(confidences)
        best_confidence = float(confidences[best_idx])
        x1, y1, x2, y2 = boxes[best_idx].astype(int)
        
        # Convert numpy types to Python native types for JSON serialization
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Load image and crop
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Ensure bbox is within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_array.shape[1], x2)
        y2 = min(image_array.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            logger.debug(f"Invalid crop bounds for {image_path}; using full image")
            x1, y1 = 0, 0
            y2, x2 = image_array.shape[0], image_array.shape[1]

        # Ensure all coords are Python int for JSON serialization
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_pil = image.crop((x1, y1, x2, y2))

        if return_pil:
            if return_bbox:
                if return_detection:
                    if return_confidence:
                        return cropped_pil, (x1, y1, x2, y2), detected, best_confidence
                    return cropped_pil, (x1, y1, x2, y2), detected
                return cropped_pil, (x1, y1, x2, y2)
            if return_detection:
                if return_confidence:
                    return cropped_pil, detected, best_confidence
                return cropped_pil, detected
            return cropped_pil

        cropped_array = np.array(cropped_pil)
        cropped_tensor = torch.tensor(cropped_array).permute(2, 0, 1).float() / 255.0

        if return_bbox:
            if return_detection:
                if return_confidence:
                    return cropped_tensor, (x1, y1, x2, y2), detected, best_confidence
                return cropped_tensor, (x1, y1, x2, y2), detected
            return cropped_tensor, (x1, y1, x2, y2)

        if return_detection:
            if return_confidence:
                return cropped_tensor, detected, best_confidence
            return cropped_tensor, detected
        return cropped_tensor
    
    def detect_batch(self, image_paths: List[str]) -> List:
        """Detect products in batch of images"""
        results = []
        for image_path in image_paths:
            try:
                cropped = self.detect_and_crop(image_path)
                results.append(cropped)
            except Exception as e:
                logger.error(f"Error detecting in {image_path}: {e}")
                results.append(None)
        
        return results


class CLIPEmbedder:
    """CLIP embedder for image and text embeddings"""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        finetuned_weights: Optional[Path] = None,
        use_finetuned: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name for CLIP
            device: Device to use
        """
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = _resolve_device(device or MODEL_CONFIG["clip"]["device"])
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        weights_path = finetuned_weights or MODEL_CONFIG["clip"].get("finetuned_weights")
        if not use_finetuned:
            weights_path = None
        if weights_path is not None:
            weights_path = Path(weights_path)
            if weights_path.exists():
                try:
                    state_dict = torch.load(weights_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded fine-tuned CLIP weights from %s", weights_path)
                except Exception as exc:
                    logger.warning("Failed to load fine-tuned CLIP weights: %s", exc)
        self.model.eval()
        
        logger.info(f"Loaded CLIP model: {model_name} on {self.device}")

    def _to_pil_image(self, image) -> Image.Image:
        """Convert a tensor or array into a PIL image for CLIP preprocessing."""
        from torchvision.transforms import functional as TF

        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image.detach().cpu()
            return TF.to_pil_image(image)

        if isinstance(image, np.ndarray):
            return TF.to_pil_image(image)

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _normalize_image_inputs(self, images) -> List[Image.Image]:
        """Normalize input images to a list of PIL images."""
        if isinstance(images, (list, tuple)):
            return [self._to_pil_image(img) for img in images]

        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                return [self._to_pil_image(img) for img in images]
            return [self._to_pil_image(images)]

        return [self._to_pil_image(images)]
    
    @torch.no_grad()
    def get_image_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Get image embedding from CLIP
        
        Args:
            image_tensor: Image tensor (C, H, W) in range [0, 1]
        
        Returns:
            Image embedding as numpy array
        """
        images = self._normalize_image_inputs(image_tensor)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        image_features = normalize_embeddings_torch(image_features, norm="l2")
        
        return image_features.cpu().numpy()
    
    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding from CLIP
        
        Args:
            text: Input text
        
        Returns:
            Text embedding as numpy array
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        text_features = self.model.get_text_features(**inputs)
        text_features = normalize_embeddings_torch(text_features, norm="l2")
        
        return text_features.cpu().numpy()
    
    @torch.no_grad()
    def get_batch_image_embeddings(self, image_tensors: torch.Tensor) -> np.ndarray:
        """Get embeddings for batch of images"""
        images = self._normalize_image_inputs(image_tensors)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        image_features = normalize_embeddings_torch(image_features, norm="l2")
        
        return image_features.cpu().numpy()


class BLIP2Captioner:
    """BLIP-2 model for image captioning"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model name for BLIP-2
            device: Device to use
        """
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        
        self.device = _resolve_device(device or MODEL_CONFIG["blip2"]["device"])
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded BLIP-2 model: {model_name} on {self.device}")
    
    @torch.no_grad()
    def generate_caption(self, image_tensor: torch.Tensor, max_length: int = 77) -> str:
        """
        Generate caption for image
        
        Args:
            image_tensor: Image tensor (C, H, W) in range [0, 1]
            max_length: Maximum caption length
        
        Returns:
            Generated caption text
        """
        # Handle batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        # Convert tensor to PIL image for processor
        from PIL import Image as PILImage
        import torchvision.transforms as transforms
        
        # Denormalize
        image_array = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        image_array = (image_array * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(image_array)
        
        # Process
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device, dtype)
        
        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=3,
        )
        
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption.strip()
    
    @torch.no_grad()
    def generate_batch_captions(self, image_tensors: torch.Tensor) -> List[str]:
        """Generate captions for batch of images"""
        captions = []
        for i in range(len(image_tensors)):
            caption = self.generate_caption(image_tensors[i])
            captions.append(caption)
        
        return captions
    
    @torch.no_grad()
    def get_itm_score(self, image_tensor: torch.Tensor, text: str) -> float:
        """
        Get Image-Text Matching score (for re-ranking)
        
        Args:
            image_tensor: Image tensor
            text: Text description
        
        Returns:
            ITM similarity score
        """
        # This requires the ITM head which may not be in BlipForConditionalGeneration
        # For now, use CLIP for re-ranking instead
        logger.warning("ITM scoring not available in current BLIP-2 implementation")
        return 0.0


class BLIP2ITMScorer:
    """BLIP-2 image-text matching scorer for reranking."""

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: Optional[str] = None):
        from transformers import AutoProcessor, Blip2ForImageTextRetrieval

        self.device = _resolve_device(device or MODEL_CONFIG["blip2"]["device"])
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

        logger.info("Loaded BLIP-2 ITM model: %s on %s", model_name, self.device)

    def _to_pil_image(self, image) -> Image.Image:
        from torchvision.transforms import functional as TF

        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image.detach().cpu()
            return TF.to_pil_image(image)
        if isinstance(image, np.ndarray):
            return TF.to_pil_image(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def score(self, image, text: str) -> float:
        image = self._to_pil_image(image)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        if hasattr(outputs, "itm_score"):
            score_tensor = outputs.itm_score
        elif hasattr(outputs, "logits"):
            score_tensor = outputs.logits
        else:
            raise ValueError("BLIP-2 ITM output does not include a score")

        if score_tensor.ndim > 1 and score_tensor.shape[-1] > 1:
            score_tensor = score_tensor[:, -1]
        score_tensor = score_tensor.squeeze()
        if score_tensor.ndim == 0:
            return float(score_tensor.item())
        return float(score_tensor[0].item())


class EmbeddingFusion:
    """Fuse image and text embeddings"""
    
    @staticmethod
    def fuse_embeddings(
        image_emb: np.ndarray,
        text_emb: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Fuse image and text embeddings
        
        Args:
            image_emb: Image embedding
            text_emb: Text embedding
            alpha: Weight for image embedding (1-alpha for text)
        
        Returns:
            Fused embedding
        """
        if image_emb.ndim == 1:
            image_emb = image_emb[None, :]
        if text_emb.ndim == 1:
            text_emb = text_emb[None, :]

        fused = alpha * image_emb + (1 - alpha) * text_emb
        
        # Normalize
        fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-6)
        
        return fused


if __name__ == "__main__":
    # Test model loading
    logger.info("Testing model loading...")
    
    # Test YOLO
    try:
        yolo = YOLODetector()
        logger.info("[OK] YOLO loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] YOLO loading failed: {e}")
    
    # Test CLIP
    try:
        clip = CLIPEmbedder()
        logger.info("[OK] CLIP loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] CLIP loading failed: {e}")
    
    # Test BLIP-2
    try:
        blip2 = BLIP2Captioner()
        logger.info("[OK] BLIP-2 loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] BLIP-2 loading failed: {e}")
