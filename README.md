# IMUSA @ FIRE 2026

**Indic Meme Understanding & Sentiment Analysis** — A multimodal shared task for sentiment classification of Punjabi memes.

## Task Overview

| | |
|:---|:---|
| **Task** | 4-class sentiment classification of Punjabi memes |
| **Dataset** | 3,502 memes (3,002 train / 500 test) |
| **Modalities** | Image + Text (Gurmukhi script) |
| **Categories** | 😏 Sarcasm · 😐 Neutral · ⚠️ Offensive · 💪 Motivational |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score |
| **Website** | [IMUSA @ FIRE 2026](https://yashdman.github.io/IMUSA/) |

## Project Structure

```
IMUSA/
├── configs/           # YAML configurations per run
├── data/              # Dataset (raw, processed, OCR cache)
├── src/
│   ├── data/          # Dataset, preprocessing, OCR
│   ├── models/        # Text encoder, image encoder, fusion, classifier
│   ├── training/      # Trainer, losses, augmentation
│   ├── evaluation/    # Metrics, error analysis
│   └── utils/         # Config, helpers
├── scripts/           # Training, prediction, ensemble entry points
├── notebooks/         # EDA, experiments, error analysis
├── outputs/           # Checkpoints, logs
└── submissions/       # Final run CSVs
```

## 3-Run Strategy

| Run | Model | Description |
|:---|:---|:---|
| **Run 1** | MuRIL (text-only) | Strong Punjabi text baseline |
| **Run 2** | CLIP ViT + MuRIL (cross-attention fusion) | Multimodal model |
| **Run 3** | Weighted Ensemble | Combines Run 1 + Run 2 + Image-only |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the downloaded dataset in `data/raw/`:
```
data/raw/
├── train/
│   ├── images/
│   └── labels.csv    # columns: image_path, text, label
└── test/
    ├── images/
    └── labels.csv    # columns: image_path, text
```

### 3. Train Models

```bash
# Run 1: Text-only baseline
python scripts/train.py --config configs/text_only.yaml

# Run 2: Multimodal fusion
python scripts/train.py --config configs/multimodal.yaml

# Image-only (for ensemble)
python scripts/train.py --config configs/image_only.yaml
```

### 4. Generate Predictions

```bash
# Single model prediction
python scripts/predict.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --test-csv data/raw/test/labels.csv \
    --output submissions/run2.csv

# Ensemble (Run 3)
python scripts/ensemble.py \
    --checkpoints outputs/checkpoints/run1.pt outputs/checkpoints/run2.pt outputs/checkpoints/image.pt \
    --test-csv data/raw/test/labels.csv \
    --val-csv data/processed/val.csv \
    --output submissions/run3_ensemble.csv \
    --method weighted
```

## Key Models Used

- **Text**: [MuRIL](https://huggingface.co/google/muril-base-cased) — Multilingual Representations for Indian Languages
- **Vision**: [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) — Contrastive Language-Image Pre-Training
- **Fusion**: Cross-attention, gated fusion, bilinear pooling

## Organizers

- Dr. Pankaj Kundan Dadure — UPES Dehradun
- Dr. Hitesh Kumar Sharma — UPES Dehradun
- Yash Kumar Dhiman — UPES Dehradun

## License

For research purposes under the FIRE 2026 shared task.
