# 🚗 Car Vision - Make, Model & Year Classifier

A deep learning system that identifies car make, model, and year from images using transfer learning and multiple datasets.

## Features

- **Multi-Dataset Training**: Combines Most_Stolen_Cars, SubsetVMMR, and CompCars datasets
- **Transfer Learning**: Uses state-of-the-art models (EfficientNet, ResNet, etc.)
- **Web Interface**: Easy-to-use Streamlit interface for image upload and prediction
- **High Accuracy**: Trained on 148K+ car images across hundreds of classes
- **Apple Silicon Support**: Optimized for M1/M2/M3 Macs

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `streamlit>=1.28.0` - Web interface
- `timm>=0.9.0` - Pre-trained models
- `Pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`

### 2. Prepare Datasets

**⚠️ Important:** Datasets are **NOT included** in this repository (listed in `.gitignore` due to large size).

You'll need to download and organize datasets in your project folder:

```
car-vision/
├── Dataset/
│   ├── Most_Stolen_Cars/      # Place dataset here
│   │   ├── chevrolet_impala_2008/
│   │   ├── honda_civic_1998/
│   │   └── ...
│   └── SubsetVMMR/             # Place dataset here
│       ├── chevrolet_impala_2007/
│       ├── ford_f150_2006/
│       └── ...
└── compcarsdb/                 # Place dataset here
    ├── image/
    ├── label/
    └── misc/
```

**Dataset Details:**

| Dataset | Classes | Images | Structure |
|---------|---------|--------|-----------|
| **Most_Stolen_Cars** | 10 | ~5,911 | `make_model_year/` folders |
| **SubsetVMMR** | 53 | ~5,925 | `make_model_year/` folders |
| **CompCars** | Hundreds | 136,726 | Hierarchical: `image/make_id/model_id/year/` |

**Where to get datasets:**
- Check research paper repositories or dataset hosting platforms
- Datasets should be organized as shown above for the code to work automatically
- The training script will auto-detect and load all available datasets

### 3. Train the Model

**Quick test** (1-2 hours on Apple Silicon):
```bash
python src/train.py --datasets stolen_cars vmmr --epochs 10 --batch_size 32
```

**Full training** with all datasets (25-50 hours):
```bash
python src/train.py --datasets stolen_cars vmmr compcars --epochs 50 --batch_size 16
```

**Training Options:**
- `--model`: Model architecture (`efficientnet_b0`, `efficientnet_b3`, `resnet50`, etc.)
- `--batch_size`: Batch size (default: 32, reduce if out of memory)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--datasets`: Which datasets to include (options: `stolen_cars`, `vmmr`, `compcars`)

### 4. Launch Web Interface

```bash
streamlit run src/app_streamlit.py
```

Then open your browser to **http://localhost:8501**

The app will automatically load the trained model from `checkpoints/best_model.pth`

## Project Structure

```
car-vision/
├── Dataset/                    # (in .gitignore - download separately)
│   ├── Most_Stolen_Cars/      # 10 classes, ~5,911 images
│   └── SubsetVMMR/             # 53 classes, ~5,925 images
├── compcarsdb/                 # (in .gitignore - download separately) 136,726 images
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Data loading and preprocessing
│   ├── model.py                # Model architecture
│   ├── train.py                # Training script
│   ├── app_streamlit.py        # Streamlit web interface ⭐
│   └── app.py                  # (Deprecated Gradio version)
├── checkpoints/                # (auto-generated during training)
│   ├── best_model.pth          # Best performing model
│   ├── latest_checkpoint.pth   # Most recent checkpoint
│   └── class_mapping.json      # Class labels mapping
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## How It Works

1. **Data Loading**: The `CarDataset` class in `dataset.py` automatically loads and combines all three datasets
2. **Transfer Learning**: Pre-trained EfficientNet (or other) models from ImageNet are fine-tuned on car images
3. **Training**: Model learns to classify cars with data augmentation (rotation, flips, color jitter)
4. **Inference**: Streamlit app loads the trained model and provides predictions with confidence scores

## Model Architectures

Supported models (via `timm`):
- `efficientnet_b0` - Fast, accurate (default)
- `efficientnet_b3` - More accurate, slower
- `resnet50` - Classic, reliable
- `convnext_tiny` - Modern architecture
- `vit_small_patch16_224` - Vision transformer

## Usage Examples

### Training Examples

```bash
# Quick training on smaller datasets
python src/train.py --datasets stolen_cars vmmr --epochs 10

# Full training with all data
python src/train.py --datasets stolen_cars vmmr compcars --epochs 50

# Use ResNet50 instead of EfficientNet
python src/train.py --model resnet50 --epochs 30

# Larger batch size (requires more GPU memory)
python src/train.py --batch_size 64 --lr 0.002
```

### Web App Examples

```bash
# Launch Streamlit app (default port 8501)
streamlit run src/app_streamlit.py

# Custom port
streamlit run src/app_streamlit.py --server.port 8080

# Custom checkpoint path (change in sidebar after launching)
streamlit run src/app_streamlit.py
```

## Performance Tips

1. **Start Small**: Train on `stolen_cars` and `vmmr` first (~12K images) to validate the pipeline
2. **Use GPU/Apple Silicon**:
   - **Apple Silicon (M1/M2/M3)**: Automatically uses MPS backend (~5-15 min/epoch)
   - **NVIDIA GPU**: Automatically uses CUDA if available
   - **CPU only**: Works but slower (~20-40 min/epoch)
3. **Batch Size**: Adjust based on available memory
   - Apple Silicon: 16-32 typical
   - NVIDIA GPU: 32-64 typical
   - CPU: 8-16 recommended
4. **Data Augmentation**: Already included (rotation, flips, color jitter)
5. **Transfer Learning**: Pre-trained ImageNet weights loaded automatically

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 16`
- Use smaller model: `--model efficientnet_b0`

### Slow Training
- Use GPU if available
- Reduce image size in `dataset.py` (default: 224x224)
- Train on fewer datasets initially

### Low Accuracy
- Train for more epochs
- Include more datasets
- Try different model architectures
- Adjust learning rate

## Future Improvements

- [ ] Add support for bounding box detection
- [ ] Implement attention visualization
- [ ] Add API endpoint (FastAPI/Flask)
- [ ] Mobile optimization
- [ ] Real-time video classification
- [ ] Fine-grained attribute prediction (color, type, etc.)

## License

This project is for educational and research purposes.

## Acknowledgments

- **CompCars Dataset** - Large-scale car dataset with hierarchical labels
- **VMMR & Most Stolen Cars Datasets** - Curated car image collections
- **PyTorch** - Deep learning framework
- **timm** - PyTorch Image Models library for pre-trained models
- **Streamlit** - Web interface framework
