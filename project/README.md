# Visual Product Search Engine - Project Structure

A comprehensive query-by-image product search system using YOLOv8, CLIP, BLIP-2, and FAISS.

## Project Structure

```
project/
├── config.py                 # Configuration and parameters
├── requirements.txt          # Python dependencies
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── utils.py             # Common utilities (logging, metrics, etc.)
│   ├── data_loader.py       # Dataset loading and management
│   ├── models.py            # Model wrappers (YOLO, CLIP, BLIP-2)
│   └── index.py             # FAISS indexing
├── scripts/                 # Phase scripts
│   ├── detect.py            # Phase 3, Step 1: YOLO detection
│   ├── caption.py           # Phase 3, Step 2: BLIP-2 captioning
│   ├── embed.py             # Phase 3, Step 3: CLIP embeddings
│   ├── index.py             # Phase 3, Step 4: Build FAISS index
│   ├── search.py            # Phase 4: Online search
│   ├── train.py             # Phase 5: Fine-tune CLIP
│   ├── evaluate.py          # Phase 6: Evaluation
│   └── run_experiments.py   # Phase 8: Run experiments
├── app/                     # Streamlit application
│   └── streamlit_app.py     # Phase 7: Interactive UI
├── models/                  # Downloaded model files
├── embeddings/              # Generated embeddings
├── index/                   # FAISS indices
├── results/                 # Evaluation results
└── logs/                    # Log files
```

## Installation

### 1. Clone/Access the project
```bash
cd g:\VR_Final_Project\project
```

### 2. Create Python environment
```bash
# Using venv
python -m venv venv
venv\Scripts\activate

# Or using conda
conda create -n vr-search python=3.11
conda activate vr-search
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Phase 1: Setup
All dependencies are pre-configured in `requirements.txt` and `config.py`.

### Phase 2: Data Preparation
The dataset is automatically loaded from `../Dataset` directory.

To inspect the dataset:
```python
from utils import DatasetManager
manager = DatasetManager()
stats = manager.get_statistics()
print(stats)
```

### Phase 3: Offline Pipeline

#### Step 1: Object Detection
```bash
python scripts/detect.py --partition gallery --device cuda
python scripts/detect.py --partition query --device cuda
```

Options:
- `--partition`: train, query, or gallery
- `--model`: YOLO model (default: yolov8m.pt)
- `--device`: cuda or cpu
- `--no-save-crops`: Skip saving cropped images

#### Step 2: Generate Captions
```bash
python scripts/caption.py --partition gallery --device cuda
python scripts/caption.py --partition query --device cuda
```

Options:
- `--partition`: Dataset partition
- `--model`: BLIP-2 model name
- `--device`: cuda or cpu
- `--use-original`: Use original images instead of detected crops

#### Step 3: Generate Embeddings
```bash
python scripts/embed.py --partition gallery --batch-size 32 --alpha 0.5
python scripts/embed.py --partition query --batch-size 32 --alpha 0.5
```

Options:
- `--partition`: Dataset partition
- `--model`: CLIP model name
- `--batch-size`: Batch size
- `--alpha`: Image embedding weight (0-1)
- `--no-text`: Skip text embeddings

#### Step 4: Build Index
```bash
python scripts/index.py  # Build all indices
# or
python scripts/index.py --partition gallery --index-type IVF
```

Options:
- `--partition`: Specific partition or all if not specified
- `--index-type`: IVF or Flat
- `--metric`: L2 or IP (inner product)
- `--clusters`: Number of clusters for IVF

### Phase 4: Online Search
```bash
python scripts/search.py --query-image <image_path> --k 10
```

### Phase 6: Evaluation
```bash
python scripts/evaluate.py --partition query
```

### Phase 7: Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Configuration

Edit `config.py` to customize:

- **Model parameters**: YOLO, CLIP, BLIP-2 settings
- **Embedding config**: Alpha (fusion weight), embedding dimension
- **Index config**: Index type, metric, number of clusters
- **Evaluation**: K values (5, 10, 15), metrics
- **Training**: Learning rate, batch size, epochs

## Experiments

Run three experimental setups:

**A: CLIP Only**
- Image embedding only (alpha=1.0)
- No fine-tuning
```bash
python scripts/embed.py --partition gallery --alpha 1.0 --no-text
```

**B: CLIP + BLIP-2 (No Fine-tune)**
- Fused image + text embeddings (alpha=0.5)
- No fine-tuning
```bash
python scripts/embed.py --partition gallery --alpha 0.5
```

**C: Fine-tuned CLIP + BLIP-2**
- Fused embeddings with contrastive learning
```bash
python scripts/train.py --epochs 10
python scripts/embed.py --partition gallery --alpha 0.5
```

## Output Files

After running all phases:

```
embeddings/
├── gallery/
│   ├── embeddings.npy
│   └── metadata.json
└── query/
    ├── embeddings.npy
    └── metadata.json

index/
├── gallery_index.faiss
├── gallery_metadata.pkl
├── gallery_config.json
├── query_index.faiss
├── query_metadata.pkl
└── query_config.json

results/
├── metrics_A.json  # CLIP only
├── metrics_B.json  # CLIP + BLIP-2
└── metrics_C.json  # Fine-tuned
```

## Dataset Statistics

- **Total images**: 52,712
- **In-shop clothes**: Various categories (Men's Denim, Jackets, Women's Dresses, etc.)
- **Splits**: Train, Query, Gallery
- **Annotations**: Bounding boxes, landmarks, attributes, descriptions

## Troubleshooting

### GPU Memory Issues
- Reduce batch size: `--batch-size 16`
- Use smaller model: `--model openai/clip-vit-small-patch32`
- Enable CPU mode: `--device cpu`

### Missing Embeddings
- Ensure Phase 3 is fully completed
- Check paths in `config.py`
- Verify dataset directory

### Slow Performance
- Use GPU: `--device cuda`
- Increase batch size (if GPU memory allows)
- Use IVF index with fewer clusters

## References

- YOLOv8: https://github.com/ultralytics/ultralytics
- CLIP: https://github.com/OpenAI/CLIP
- BLIP-2: https://github.com/salesforce/LAVIS
- FAISS: https://github.com/facebookresearch/faiss
- DeepFashion: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
