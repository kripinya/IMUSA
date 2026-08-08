# ═══════════════════════════════════════════════════════════════════════════
# IMUSA @ FIRE 2026 — RUN 3: Ensemble
# ═══════════════════════════════════════════════════════════════════════════
#
# KAGGLE SETUP INSTRUCTIONS:
#   1. Create a new Kaggle Notebook
#   2. Accelerator: None (CPU is fine for just combining CSVs)
#   3. You need the probability outputs from Run 1 and Run 2.
#      - If you ran them in separate notebooks, you need to add their outputs 
#        as Data sources to this notebook.
#      - (e.g. Add Data -> Your Work -> Select the Run 1 and Run 2 notebooks)
#   4. Update the paths below to point to the saved CSVs.
# ═══════════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import numpy as np

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
# UPDATE THESE PATHS based on where your previous notebook outputs are mounted
# e.g., "/kaggle/input/imusa-run1/run1_test_probs.csv"
RUN1_TEST_PROBS = "/kaggle/input/datasets/ananyakarn/data-sources/run1_test_probs.csv"
RUN2_TEST_PROBS = "/kaggle/input/datasets/ananyakarn/data-sources/run2_test_probs.csv"

# Optional: if you want to optimize weights, provide the validation probs
RUN1_VAL_PROBS  = "/kaggle/input/imusa-run1/run1_val_probs.csv"
RUN2_VAL_PROBS  = "/kaggle/input/imusa-run2/run2_val_probs.csv"

OUTPUT_DIR      = "/kaggle/working"
NUM_CLASSES     = 4
CLASS_NAMES     = ["Sarcasm", "Neutral", "Offensive", "Motivational"]
ID2LABEL        = {0: "Sarcasm", 1: "Neutral", 2: "Offensive", 3: "Motivational"}

# ── 2. LOAD PROBABILITIES ────────────────────────────────────────────────────
print("Loading Run 1 and Run 2 probabilities...")

try:
    df1 = pd.read_csv(RUN1_TEST_PROBS)
    df2 = pd.read_csv(RUN2_TEST_PROBS)
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("Please make sure you added the outputs from Run 1 and Run 2 to this notebook!")
    import sys; sys.exit(1)

# Extract probability matrices (N, 4)
prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
p1 = df1[prob_cols].values
p2 = df2[prob_cols].values

print(f"Run 1 shape: {p1.shape}")
print(f"Run 2 shape: {p2.shape}")

# ── 3. ENSEMBLE STRATEGY ──────────────────────────────────────────────────────
# We will use a simple weighted average.
# You can adjust these weights based on which model performed better on Val.
# E.g., if Multimodal (Run 2) was much better, give it a higher weight.

WEIGHT_RUN1 = 0.4  # Text-only weight
WEIGHT_RUN2 = 0.6  # Multimodal weight

print(f"\nApplying weighted ensemble: {WEIGHT_RUN1}*(Run 1) + {WEIGHT_RUN2}*(Run 2)")

ensemble_probs = (WEIGHT_RUN1 * p1) + (WEIGHT_RUN2 * p2)

# Get the final predicted class
preds = ensemble_probs.argmax(axis=1)

# ── 4. CREATE SUBMISSION ──────────────────────────────────────────────────────
submission = pd.DataFrame({
    "Id": df1["Id"].values,
    "Category": [ID2LABEL[p] for p in preds]
})

sub_path = os.path.join(OUTPUT_DIR, "run3_ensemble_submission.csv")
submission.to_csv(sub_path, index=False)

print(f"\n✅ Ensemble Submission saved: {sub_path}")
print(f"   Shape: {submission.shape}")
print(f"\nDistribution of Ensemble Predictions:")
print(submission['Category'].value_counts())
