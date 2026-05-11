"""
Phase 7: Streamlit demo app
Upload an image, inspect the YOLO crop, and view top-k results.
"""

from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from scripts.search import ProductSearchEngine
from utils import setup_logger, YOLODetector
from config import MODEL_CONFIG

logger = setup_logger(__name__)

st.set_page_config(page_title="Visual Product Search", layout="wide")
st.title("Visual Product Search Engine")
st.caption("Query-by-image fashion retrieval with YOLO, CLIP, BLIP-2, and FAISS")

uploaded = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])
partition = st.selectbox("Search index", ["gallery", "query"], index=0)
k = st.slider("Top-K", min_value=5, max_value=20, value=10, step=1)
alpha = st.slider("Fusion alpha", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
device = st.selectbox("Device", ["cuda", "cpu"], index=0 if MODEL_CONFIG["clip"]["device"] == "cuda" else 1)
use_crop = st.checkbox("Use YOLO crop", value=True)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1, 2])
    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Search"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        try:
            if use_crop:
                detector = YOLODetector(device=device)
                crop_image, _ = detector.detect_and_crop(temp_path, return_bbox=True, return_pil=True)
                with left:
                    st.image(crop_image, caption="YOLO crop (used for search)", use_container_width=True)

            engine = ProductSearchEngine(partition=partition, device=device)
            results = engine.search(temp_path, k=k, alpha=alpha, use_crop=use_crop)
            results = engine.rerank(temp_path, results, top_n=k, use_crop=use_crop)

            with right:
                st.subheader("Top Results")
                for rank, item in enumerate(results, start=1):
                    st.write(f"{rank}. {item.get('image_path', 'unknown')} - score: {item.get('score', 0.0):.4f}")
                    if item.get("caption"):
                        st.caption(item["caption"])
        except Exception as exc:
            st.error(f"Search failed: {exc}")
