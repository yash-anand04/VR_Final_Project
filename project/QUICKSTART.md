# Quick Start Guide

## Setup (5 minutes)

Recommended Python version: 3.10.x to 3.12.x

### 1. Install Python Dependencies
```bash
# Option A: from project directory
cd g:\VR_Final_Project\project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Option B: from workspace root
cd g:\VR_Final_Project
python -m venv venv
venv\Scripts\activate
pip install -r project\requirements.txt
```

### 2. Verify Dataset Path
The project expects the dataset at: `g:\VR_Final_Project\Dataset\`

Check configuration in `config.py` if needed.

## Running the Pipeline (Variable time)

### Option A: Run Complete Pipeline (Recommended)
```bash
python project\setup_pipeline.py --phase 3 --device cuda
```

This will automatically:
1. Detect products using YOLO
2. Generate captions using BLIP-2
3. Extract embeddings using CLIP
4. Build FAISS indices

### Option B: Run Individual Steps
```bash
# Step 1: Detection
python project\scripts/detect.py --partition gallery --device cuda

# Step 2: Captioning  
python project\scripts/caption.py --partition gallery --device cuda

# Step 3: Embeddings
python project\scripts/embed.py --partition gallery --alpha 0.5

# Step 4: Indexing
python project\scripts/index.py --partition gallery
```

### Option C: Run from workspace root
```bash
cd g:\VR_Final_Project
g:\VR_Final_Project\venv\Scripts\python.exe project\setup_pipeline.py --phase 3 --device cuda
```

## Configuration

Edit `config.py` to customize:

- `MODEL_CONFIG`: Model names and parameters
- `EMBEDDING_CONFIG`: Alpha weight, embedding dimension
- `INDEX_CONFIG`: Index type, metric, clustering
- `DATA_PROCESSING`: Batch sizes, image sizes

## Common Issues

### "CUDA out of memory"
```bash
python setup_pipeline.py --phase 3 --batch-size 16 --device cuda
```

### "Models not found"
The first run will automatically download models. This may take time.

### "Dataset not found"
Check that `../Dataset` exists relative to the project folder.

## Next Steps

After Phase 3 completes:

1. Phase 4 - Online search
```bash
python setup_pipeline.py --phase 4 --query-image <path_to_image> --search-partition gallery --device cuda
```

2. Phase 5 - Fine-tuning scaffold
```bash
python setup_pipeline.py --phase 5 --epochs 10 --device cuda
```

3. Phase 6 - Evaluation
```bash
python setup_pipeline.py --phase 6 --search-partition gallery --device cuda
```

4. Phase 7 - Streamlit app
```bash
streamlit run app/streamlit_app.py
```

5. Phase 8 - Experiment comparison
```bash
python setup_pipeline.py --phase 8 --search-partition gallery --device cuda
```

## Documentation

- `README.md`: Full project documentation
- `config.py`: All configuration parameters
- `utils/models.py`: Model wrapper documentation
- Individual scripts have detailed docstrings

## Support

For issues or questions, check the logs in `logs/project.log`
