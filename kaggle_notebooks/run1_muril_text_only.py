# ═══════════════════════════════════════════════════════════════════════════
# IMUSA @ FIRE 2026 — RUN 1: MuRIL Text-Only Classifier
# ═══════════════════════════════════════════════════════════════════════════
#
# KAGGLE SETUP INSTRUCTIONS:
#   1. Create a new Kaggle Notebook
#   2. Set accelerator: Settings → Accelerator → GPU T4 x2  (or P100)
#   3. Upload your dataset:
#      - Go to "Add Data" → "Upload" → create a dataset called "imusa-data"
#      - Upload a zip with: train_punjabi_dataset.csv, Test.csv,
#        train/images/ folder, test/images/ folder
#      - OR use "Add Data" → "Dataset" if already uploaded
#   4. Paste this entire script into a code cell
#   5. Update INPUT_DIR to match your dataset path (see below)
#   6. Run All
#
# EXPECTED RUNTIME: ~30-45 minutes on GPU T4
# EXPECTED VAL F1:  ~0.55–0.65 (text-only baseline)
# ═══════════════════════════════════════════════════════════════════════════

# ── 0. INSTALL DEPENDENCIES ─────────────────────────────────────────────────
import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip_install("transformers>=4.40")
pip_install("accelerate")
pip_install("sentencepiece")

# ── 1. IMPORTS ───────────────────────────────────────────────────────────────
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY'}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ── 2. CONFIGURATION ─────────────────────────────────────────────────────────
# !! UPDATE THIS PATH to match where your data is on Kaggle !!
# Typical paths:
#   /kaggle/input/imusa-data/           <- if you named the dataset "imusa-data"
#   /kaggle/input/imusa-punjabi-memes/  <- depends on your dataset name

INPUT_DIR     = "/kaggle/input/IMUSA"          # <-- CHANGE THIS
TRAIN_CSV     = os.path.join(INPUT_DIR, "train_punjabi_dataset.csv")
TEST_CSV      = os.path.join(INPUT_DIR, "Test.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train", "images")
TEST_IMG_DIR  = os.path.join(INPUT_DIR, "test", "images")
OUTPUT_DIR    = "/kaggle/working"

# Model config
MODEL_NAME    = "google/muril-base-cased"   # Best for Indian languages
MAX_LEN       = 128
BATCH_SIZE    = 32
EPOCHS        = 12
LR            = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
SEED          = 42
NUM_CLASSES   = 4
PATIENCE      = 3

LABEL2ID = {"Sarcasm": 0, "Neutral": 1, "Offensive": 2, "Motivational": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(NUM_CLASSES)]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 3. LOAD & EXPLORE DATA ───────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
test_df  = pd.read_csv(TEST_CSV,  encoding="utf-8-sig")

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"\nTrain columns: {list(train_df.columns)}")
print(f"\nClass distribution:")
print(train_df["Category"].value_counts())
print(f"\nSample train row:")
print(train_df.iloc[0])

# Map labels
train_df["label"] = train_df["Category"].map(LABEL2ID)

# Handle missing text
train_df["Text"] = train_df["Text"].fillna("").astype(str)
test_df["Text"]  = test_df["Text"].fillna("").astype(str)

# ── EDA Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
colors = ["#f97316", "#3b82f6", "#ef4444", "#22c55e"]
counts = [train_df[train_df["Category"] == c].shape[0] for c in CLASS_NAMES]
axes[0].bar(CLASS_NAMES, counts, color=colors)
axes[0].set_title("Class Distribution (Train)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Count")
for i, (bar, count) in enumerate(zip(axes[0].patches, counts)):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 str(count), ha='center', fontsize=11)

# Text length distribution
train_df["text_len"] = train_df["Text"].apply(lambda x: len(x.split()))
axes[1].hist(train_df["text_len"], bins=30, color="#6366f1", edgecolor="white")
axes[1].set_title("Meme Text Length Distribution (words)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_plots.png"), dpi=150)
plt.show()
print(f"Text length stats:\n{train_df['text_len'].describe()}")

# ── 4. TRAIN/VAL SPLIT ───────────────────────────────────────────────────────
train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=SEED
)
train_data = train_data.reset_index(drop=True)
val_data   = val_data.reset_index(drop=True)

print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")
print(f"Val class dist: {val_data['Category'].value_counts().to_dict()}")

# ── 5. COMPUTE CLASS WEIGHTS (for Focal Loss) ────────────────────────────────
label_counts = train_data["label"].value_counts().sort_index()
class_weights = len(train_data) / (NUM_CLASSES * label_counts.values)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"\nClass weights: {dict(zip(CLASS_NAMES, class_weights.cpu().numpy().round(3)))}")

# ── 6. TOKENIZER ─────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── 7. DATASET CLASS ─────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, "Text"])

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        if not self.is_test:
            item["label"] = torch.tensor(self.df.loc[idx, "label"], dtype=torch.long)

        return item

# ── 8. MODEL ─────────────────────────────────────────────────────────────────
class MuRILClassifier(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768 for muril-base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**kwargs)
        # [CLS] token representation
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# ── 9. FOCAL LOSS ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Down-weights well-classified examples — crucial for the 52-sample Offensive class."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# ── 10. METRICS ───────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "report": report, "cm": cm}

def print_metrics(metrics, epoch=None):
    prefix = f"[Epoch {epoch}] " if epoch else ""
    print(f"\n{prefix}{'='*55}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['precision']:.4f}")
    print(f"  Macro Recall:    {metrics['recall']:.4f}")
    print(f"  Macro F1:        {metrics['f1']:.4f}")
    print(f"{'='*55}")
    print(metrics["report"])

# ── 11. TRAINING SETUP ────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME} ...")
model = MuRILClassifier(MODEL_NAME, num_classes=NUM_CLASSES, dropout=0.3)

# Multi-GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

# DataLoaders
train_ds = TextDataset(train_data, tokenizer, MAX_LEN, is_test=False)
val_ds   = TextDataset(val_data,   tokenizer, MAX_LEN, is_test=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

# Loss, Optimizer, Scheduler
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)
scheduler     = get_scheduler("cosine", optimizer=optimizer,
                               num_warmup_steps=warmup_steps,
                               num_training_steps=total_steps)

scaler = GradScaler(enabled=torch.cuda.is_available())

# ── 12. TRAINING LOOP ─────────────────────────────────────────────────────────
best_f1          = 0.0
patience_counter = 0
history          = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
best_ckpt_path   = os.path.join(OUTPUT_DIR, "run1_muril_best.pt")

print(f"\n{'='*60}")
print(f"TRAINING: {MODEL_NAME}")
print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):
    # ── Train ────────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(f"  Epoch {epoch} step {step+1}/{len(train_loader)} | loss: {epoch_loss/(step+1):.4f}", end="\r")

    avg_train_loss = epoch_loss / len(train_loader)

    # ── Validate ─────────────────────────────────────────────
    model.eval()
    val_loss  = 0.0
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss   = criterion(logits, labels)

            val_loss  += loss.item()
            preds      = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    metrics = compute_metrics(all_true, all_preds)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_f1"].append(metrics["f1"])
    history["val_acc"].append(metrics["accuracy"])

    print(f"\nEpoch {epoch}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val F1: {metrics['f1']:.4f} | "
          f"Val Acc: {metrics['accuracy']:.4f}")

    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        # Save only model state_dict (lighter)
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state,
            "f1": best_f1,
            "accuracy": metrics["accuracy"],
        }, best_ckpt_path)
        print(f"  ✓ New best! F1={best_f1:.4f} — saved to {best_ckpt_path}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ✗ No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\n✅ Training complete! Best Val F1: {best_f1:.4f}")

# ── 13. PLOT TRAINING CURVES ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_ran = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs_ran, history["train_loss"], "b-o", label="Train Loss")
axes[0].plot(epochs_ran, history["val_loss"],   "r-o", label="Val Loss")
axes[0].set_title("Loss Curves", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, history["val_f1"],  "g-o", label="Val Macro F1")
axes[1].plot(epochs_ran, history["val_acc"], "m-o", label="Val Accuracy")
axes[1].set_title("Validation Metrics", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "run1_training_curves.png"), dpi=150)
plt.show()

# ── 14. FINAL EVALUATION ON VAL SET ─────────────────────────────────────────
print("\nLoading best checkpoint for final evaluation...")
checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
# Reload clean model (not DataParallel)
eval_model = MuRILClassifier(MODEL_NAME, num_classes=NUM_CLASSES)
eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model = eval_model.to(DEVICE)
eval_model.eval()

all_preds = []
all_true  = []
all_probs = []

with torch.no_grad():
    for batch in val_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)
        labels = batch["label"]

        with autocast(enabled=torch.cuda.is_available()):
            logits = eval_model(input_ids, attention_mask, token_type_ids)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())
        all_probs.extend(probs)

val_metrics = compute_metrics(all_true, all_preds)
print_metrics(val_metrics, epoch="BEST")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
cm_norm = val_metrics["cm"].astype(float)
cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Run 1 (MuRIL Text-Only)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "run1_confusion_matrix.png"), dpi=150)
plt.show()

# ── 15. GENERATE TEST PREDICTIONS ────────────────────────────────────────────
print("\nGenerating test predictions...")
test_ds     = TextDataset(test_df, tokenizer, MAX_LEN, is_test=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

test_preds = []
test_probs = []

eval_model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        with autocast(enabled=torch.cuda.is_available()):
            logits = eval_model(input_ids, attention_mask, token_type_ids)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        test_preds.extend(preds)
        test_probs.extend(probs)

# Save submission
submission = pd.DataFrame({
    "Id":       test_df["Id"].values,
    "Category": [ID2LABEL[p] for p in test_preds]
})
sub_path = os.path.join(OUTPUT_DIR, "run1_submission.csv")
submission.to_csv(sub_path, index=False)

# Save probabilities for ensemble
probs_df = pd.DataFrame(np.array(test_probs), columns=[f"prob_{c}" for c in CLASS_NAMES])
probs_df.insert(0, "Id", test_df["Id"].values)
probs_df.to_csv(os.path.join(OUTPUT_DIR, "run1_test_probs.csv"), index=False)

# Save val probabilities for ensemble
val_probs_df = pd.DataFrame(np.array(all_probs), columns=[f"prob_{c}" for c in CLASS_NAMES])
val_probs_df.insert(0, "Id", val_data["Id"].values)
val_probs_df.insert(1, "true_label", [ID2LABEL[t] for t in all_true])
val_probs_df.to_csv(os.path.join(OUTPUT_DIR, "run1_val_probs.csv"), index=False)

print(f"✅ Submission saved: {sub_path}")
print(f"   Shape: {submission.shape}")
print(f"   Distribution:\n{submission['Category'].value_counts()}")
print(f"\n📊 Final Val F1: {val_metrics['f1']:.4f}")
print(f"📊 Val Accuracy: {val_metrics['accuracy']:.4f}")
print("\n🎯 OUTPUT FILES:")
print(f"   {sub_path}")
print(f"   {OUTPUT_DIR}/run1_test_probs.csv    ← needed for Run 3 ensemble")
print(f"   {OUTPUT_DIR}/run1_val_probs.csv     ← needed for Run 3 ensemble")
print(f"   {OUTPUT_DIR}/run1_muril_best.pt     ← model checkpoint")
