# ═══════════════════════════════════════════════════════════════════════════
# IMUSA @ FIRE 2026 — RUN 2: Multimodal CLIP-ViT + MuRIL with Cross-Attention
# ═══════════════════════════════════════════════════════════════════════════
#
# KAGGLE SETUP INSTRUCTIONS:
#   1. Create a new Kaggle Notebook (separate from Run 1)
#   2. Accelerator: GPU T4 x2 or P100
#   3. Add the SAME "imusa-data" dataset
#   4. Update INPUT_DIR below
#   5. Run All
#
# EXPECTED RUNTIME: ~60-90 minutes on GPU T4
# EXPECTED VAL F1:  ~0.60–0.70 (multimodal beats text-only)
# ═══════════════════════════════════════════════════════════════════════════

# ── 0. INSTALL ────────────────────────────────────────────────────────────────
import subprocess, sys
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip_install("transformers>=4.40")
pip_install("accelerate")
pip_install("sentencepiece")
pip_install("albumentations")

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoTokenizer, AutoModel,
    CLIPModel, CLIPProcessor,
    get_scheduler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── 2. CONFIG ─────────────────────────────────────────────────────────────────
INPUT_DIR     = "/kaggle/input/imusa-data"          # <-- UPDATE THIS
TRAIN_CSV     = os.path.join(INPUT_DIR, "train_punjabi_dataset.csv")
TEST_CSV      = os.path.join(INPUT_DIR, "Test.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train", "images")
TEST_IMG_DIR  = os.path.join(INPUT_DIR, "test", "images")
OUTPUT_DIR    = "/kaggle/working"

TEXT_MODEL    = "google/muril-base-cased"
VISION_MODEL  = "openai/clip-vit-base-patch32"
MAX_LEN       = 128
IMG_SIZE      = 224
BATCH_SIZE    = 16          # smaller due to two models
EPOCHS        = 15
TEXT_LR       = 1e-5        # lower for pretrained text encoder
IMAGE_LR      = 1e-5        # lower for pretrained vision encoder
FUSION_LR     = 5e-5        # higher for randomly init fusion layers
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
SEED          = 42
NUM_CLASSES   = 4
PATIENCE      = 4

LABEL2ID  = {"Sarcasm": 0, "Neutral": 1, "Offensive": 2, "Motivational": 3}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(NUM_CLASSES)]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 3. LOAD DATA ──────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
test_df  = pd.read_csv(TEST_CSV,  encoding="utf-8-sig")

train_df["label"] = train_df["Category"].map(LABEL2ID)
train_df["Text"]  = train_df["Text"].fillna("").astype(str)
test_df["Text"]   = test_df["Text"].fillna("").astype(str)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Class dist:\n{train_df['Category'].value_counts()}")

# ── 4. IMAGE PATH RESOLVER ────────────────────────────────────────────────────
def resolve_image_path(image_id: str, img_dir: str) -> str:
    """
    Resolve image path, handling .jpg vs .jpeg extension mismatch in test set.
    CSV has .jpg but some test files are .jpeg on disk.
    """
    # Direct match
    path = os.path.join(img_dir, image_id)
    if os.path.exists(path):
        return path

    # Try swapping extension
    stem, ext = os.path.splitext(image_id)
    alt_ext = ".jpeg" if ext.lower() == ".jpg" else ".jpg"
    alt_path = os.path.join(img_dir, stem + alt_ext)
    if os.path.exists(alt_path):
        return alt_path

    # Return original even if not found (PIL will raise a clear error)
    return path

# ── 5. IMAGE TRANSFORMS ────────────────────────────────────────────────────────
# CLIP uses specific mean/std
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.85, 1.0), p=0.5),
    A.HorizontalFlip(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ToTensorV2()
])

# ── 6. DATASET ────────────────────────────────────────────────────────────────
class MultimodalMemeDataset(Dataset):
    def __init__(self, df, tokenizer, img_dir, transform, max_len=128, is_test=False):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.img_dir   = img_dir
        self.transform = transform
        self.max_len   = max_len
        self.is_test   = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # ── Text ────────────────────────────────────────────
        text = str(row["Text"])
        enc  = self.tokenizer(
            text, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = enc.get("token_type_ids", {})
        if hasattr(token_type_ids, "squeeze"):
            token_type_ids = token_type_ids.squeeze(0)

        # ── Image ────────────────────────────────────────────
        img_path = resolve_image_path(row["Id"], self.img_dir)
        try:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = self.transform(image=img)["image"]   # albumentations returns dict
        except Exception as e:
            print(f"[WARN] Could not load {img_path}: {e}")
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        item = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "image":          img,
        }
        if isinstance(token_type_ids, torch.Tensor):
            item["token_type_ids"] = token_type_ids

        if not self.is_test:
            item["label"] = torch.tensor(int(row["label"]), dtype=torch.long)

        return item

# ── 7. MODEL: Cross-Attention Fusion ─────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Text ↔ Image bidirectional cross-attention.
    Text queries attend to image, image queries attend to text.
    """
    def __init__(self, text_dim=768, image_dim=512, proj_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.proj_dim = proj_dim

        self.text_proj  = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)

        self.t2i_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.i2t_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(proj_dim)
        self.ff   = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_feat, image_feat):
        tp = self.text_proj(text_feat).unsqueeze(1)    # (B, 1, proj_dim)
        ip = self.image_proj(image_feat).unsqueeze(1)  # (B, 1, proj_dim)

        t2i, _ = self.t2i_attn(tp, ip, ip)  # text queries image
        i2t, _ = self.i2t_attn(ip, tp, tp)  # image queries text

        t2i = t2i.squeeze(1)  # (B, proj_dim)
        i2t = i2t.squeeze(1)

        combined = torch.cat([t2i, i2t], dim=-1)  # (B, proj_dim*2)
        fused    = self.ff(combined)               # (B, proj_dim)
        fused    = self.norm(fused + t2i)          # residual

        return fused


class MultimodalClassifier(nn.Module):
    def __init__(self, text_model_name, vision_model_name, num_classes=4,
                 proj_dim=256, num_heads=4, dropout=0.3):
        super().__init__()

        # Text encoder (MuRIL)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size   # 768

        # Image encoder — CLIP vision only
        clip       = CLIPModel.from_pretrained(vision_model_name)
        self.vision_model      = clip.vision_model
        self.visual_projection = clip.visual_projection
        image_dim = clip.config.projection_dim              # 512
        del clip

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            text_dim=text_dim, image_dim=image_dim,
            proj_dim=proj_dim, num_heads=num_heads, dropout=dropout
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values, token_type_ids=None):
        # ── Text features ────────────────────────────────────
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        text_out  = self.text_encoder(**kwargs)
        text_feat = text_out.last_hidden_state[:, 0, :]   # [CLS]

        # ── Image features ────────────────────────────────────
        vis_out    = self.vision_model(pixel_values=pixel_values)
        image_feat = self.visual_projection(vis_out.pooler_output)  # (B, 512)

        # ── Fuse ──────────────────────────────────────────────
        fused  = self.fusion(text_feat, image_feat)
        logits = self.classifier(fused)
        return logits

# ── 8. FOCAL LOSS ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "report": report, "cm": cm}

# ── 9. TRAINING SETUP ─────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {TEXT_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

# Train/Val split
train_data, val_data = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=SEED
)
train_data = train_data.reset_index(drop=True)
val_data   = val_data.reset_index(drop=True)

# Class weights
label_counts  = train_data["label"].value_counts().sort_index()
class_weights = len(train_data) / (NUM_CLASSES * label_counts.values)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.cpu().numpy().round(3)))}")

# Datasets & Loaders
train_ds = MultimodalMemeDataset(train_data, tokenizer, TRAIN_IMG_DIR, train_transform, MAX_LEN, is_test=False)
val_ds   = MultimodalMemeDataset(val_data,   tokenizer, TRAIN_IMG_DIR, val_transform,   MAX_LEN, is_test=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,   shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

# Model
print(f"Loading model: {TEXT_MODEL} + {VISION_MODEL} ...")
model = MultimodalClassifier(TEXT_MODEL, VISION_MODEL, NUM_CLASSES, proj_dim=256, num_heads=4)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

# Differential learning rates
def get_param_groups(model):
    """Different LR for text encoder, image encoder, and fusion/classifier."""
    # Unwrap DataParallel if needed
    m = model.module if hasattr(model, "module") else model

    text_params    = list(m.text_encoder.parameters())
    vision_params  = list(m.vision_model.parameters()) + list(m.visual_projection.parameters())
    fusion_params  = list(m.fusion.parameters()) + list(m.classifier.parameters())

    text_ids   = set(id(p) for p in text_params)
    vision_ids = set(id(p) for p in vision_params)

    return [
        {"params": [p for p in text_params   if p.requires_grad], "lr": TEXT_LR,   "name": "text"},
        {"params": [p for p in vision_params  if p.requires_grad], "lr": IMAGE_LR,  "name": "vision"},
        {"params": [p for p in fusion_params  if p.requires_grad], "lr": FUSION_LR, "name": "fusion"},
    ]

param_groups  = get_param_groups(model)
optimizer     = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
criterion     = FocalLoss(alpha=class_weights, gamma=2.0)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)
scheduler     = get_scheduler("cosine", optimizer=optimizer,
                               num_warmup_steps=warmup_steps,
                               num_training_steps=total_steps)
scaler        = GradScaler(enabled=torch.cuda.is_available())

# ── 10. TRAINING LOOP ─────────────────────────────────────────────────────────
best_f1          = 0.0
patience_counter = 0
history          = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
best_ckpt_path   = os.path.join(OUTPUT_DIR, "run2_multimodal_best.pt")

print(f"\n{'='*60}")
print(f"TRAINING: CLIP-ViT + MuRIL + Cross-Attention Fusion")
print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
print(f"  LR — Text: {TEXT_LR} | Vision: {IMAGE_LR} | Fusion: {FUSION_LR}")
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pixel_values   = batch["image"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, pixel_values, token_type_ids)
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

    # ── Validate ──────────────────────────────────────────
    model.eval()
    val_loss  = 0.0
    all_preds = []
    all_true  = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values   = batch["image"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids, attention_mask, pixel_values, token_type_ids)
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
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state,
            "f1": best_f1,
            "accuracy": metrics["accuracy"],
        }, best_ckpt_path)
        print(f"  ✓ New best! F1={best_f1:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ✗ No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\n✅ Training complete! Best Val F1: {best_f1:.4f}")

# ── 11. TRAINING CURVES ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_ran = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs_ran, history["train_loss"], "b-o", label="Train Loss")
axes[0].plot(epochs_ran, history["val_loss"],   "r-o", label="Val Loss")
axes[0].set_title("Loss Curves — Run 2 Multimodal", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_ran, history["val_f1"],  "g-o", label="Val F1")
axes[1].plot(epochs_ran, history["val_acc"], "m-o", label="Val Acc")
axes[1].set_title("Val Metrics — Run 2 Multimodal", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "run2_training_curves.png"), dpi=150)
plt.show()

# ── 12. FINAL EVALUATION ─────────────────────────────────────────────────────
print("\nLoading best checkpoint...")
checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
eval_model = MultimodalClassifier(TEXT_MODEL, VISION_MODEL, NUM_CLASSES, proj_dim=256, num_heads=4)
eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model = eval_model.to(DEVICE)
eval_model.eval()

all_preds, all_true, all_probs = [], [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pixel_values   = batch["image"].to(DEVICE)
        labels         = batch["label"]
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        with autocast(enabled=torch.cuda.is_available()):
            logits = eval_model(input_ids, attention_mask, pixel_values, token_type_ids)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())
        all_probs.extend(probs)

val_metrics = compute_metrics(all_true, all_preds)
print(f"\n{'='*55}")
print(f"  RUN 2 BEST RESULTS")
print(f"  Accuracy:     {val_metrics['accuracy']:.4f}")
print(f"  Macro F1:     {val_metrics['f1']:.4f}")
print(f"  Precision:    {val_metrics['precision']:.4f}")
print(f"  Recall:       {val_metrics['recall']:.4f}")
print(f"{'='*55}")
print(val_metrics["report"])

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_norm = val_metrics["cm"].astype(float) / val_metrics["cm"].sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Run 2 (CLIP+MuRIL Multimodal)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "run2_confusion_matrix.png"), dpi=150)
plt.show()

# ── 13. TEST PREDICTIONS ─────────────────────────────────────────────────────
print("\nGenerating test predictions...")
test_ds     = MultimodalMemeDataset(test_df, tokenizer, TEST_IMG_DIR, val_transform, MAX_LEN, is_test=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

test_preds, test_probs = [], []
eval_model.eval()

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        pixel_values   = batch["image"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        with autocast(enabled=torch.cuda.is_available()):
            logits = eval_model(input_ids, attention_mask, pixel_values, token_type_ids)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        test_preds.extend(preds)
        test_probs.extend(probs)

# Save submission
submission = pd.DataFrame({
    "Id":       test_df["Id"].values,
    "Category": [ID2LABEL[p] for p in test_preds]
})
sub_path = os.path.join(OUTPUT_DIR, "run2_submission.csv")
submission.to_csv(sub_path, index=False)

# Save probabilities for ensemble
probs_df = pd.DataFrame(np.array(test_probs), columns=[f"prob_{c}" for c in CLASS_NAMES])
probs_df.insert(0, "Id", test_df["Id"].values)
probs_df.to_csv(os.path.join(OUTPUT_DIR, "run2_test_probs.csv"), index=False)

val_probs_df = pd.DataFrame(np.array(all_probs), columns=[f"prob_{c}" for c in CLASS_NAMES])
val_probs_df.insert(0, "Id", val_data["Id"].values)
val_probs_df.insert(1, "true_label", [ID2LABEL[t] for t in all_true])
val_probs_df.to_csv(os.path.join(OUTPUT_DIR, "run2_val_probs.csv"), index=False)

print(f"✅ Submission saved: {sub_path}")
print(f"   Distribution:\n{submission['Category'].value_counts()}")
print(f"\n🎯 Final Val F1: {val_metrics['f1']:.4f}")
print("\n🎯 OUTPUT FILES:")
print(f"   {sub_path}")
print(f"   {OUTPUT_DIR}/run2_test_probs.csv  ← needed for Run 3 ensemble")
print(f"   {OUTPUT_DIR}/run2_val_probs.csv   ← needed for Run 3 ensemble")
