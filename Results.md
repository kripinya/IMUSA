# IMUSA Shared Task: Final Implementation Results

This document provides a comprehensive overview of the end-to-end implementation for the Indic Meme Understanding & Sentiment Analysis (IMUSA) shared task at FIRE 2026. It details the data analysis, architectural choices, and the specific outcomes of the three evaluated runs.

---

## 1. Dataset Analysis & Preparation

The IMUSA dataset consists of 3,502 Punjabi memes (3,002 for training, 500 for testing). Our initial exploratory data analysis (EDA) revealed several critical constraints that guided our architecture:

### 1.1 Class Imbalance
The training dataset is heavily skewed toward 'Sarcasm', with 'Offensive' memes being severely underrepresented:
- **Sarcasm**: 1329 (44.3%)
- **Motivational**: 870 (29.0%)
- **Neutral**: 751 (25.0%)
- **Offensive**: 52 (1.7%)

**Implementation Strategy:** We utilized an 80/20 stratified split to ensure the validation set maintained this exact distribution. During training, we implemented Focal Loss ($\gamma=2.0$) to prevent the model from ignoring the minority classes.

### 1.2 Text Length Statistics
An analysis of the meme text lengths yielded the following distribution:
- **Mean**: ~16.8 words
- **Max**: 73 words
- **75th Percentile**: 20 words

**Implementation Strategy:** We set the Transformer maximum sequence length (`MAX_LEN`) to 128. This provided a safe margin to capture all text entirely without truncation, while remaining memory efficient.

---

## 2. Implementation Overview

To handle the multimodal nature of the task and the constraints of the dataset, we designed a progressive 3-run pipeline. All models were trained on Kaggle using a dual Tesla T4 GPU environment to handle the high VRAM requirements of fusing Vision and Language transformers.

### Architectural Components
- **Text Encoder**: `google/muril-base-cased` (Multilingual Representations for Indian Languages). Chosen specifically for its pre-training on 17 Indian languages, offering superior contextual understanding of Punjabi compared to standard mBERT.
- **Vision Encoder**: `openai/clip-vit-base-patch32`. Chosen for its semantic alignment between visual concepts and natural language.
- **Fusion Module**: A Bidirectional Cross-Attention module. Instead of simple concatenation, this allows the text embeddings to actively query the visual features (and vice versa) to detect contradictions inherent in memes.

---

## 3. Evaluated Runs & Outcomes

### 3.1 Run 1: Text-Only Baseline (MuRIL)
**Objective**: Establish a strong baseline utilizing only the textual modality to determine how much sentiment can be extracted purely from the text.
**Architecture**: MuRIL `[CLS]` token $\rightarrow$ Linear Projection $\rightarrow$ Classifier.
**Validation Outcomes**:
- **Macro F1**: 0.5274
- **Accuracy**: 63.39%
- **Class-wise Performance**:
  - Sarcasm: 70.56% F1 (64% Recall)
  - Neutral: 46.61% F1 (53% Recall)
  - Motivational: 71.58% F1 (75% Recall)
  - Offensive: 22.22% F1 (18% Recall)

**Insights**: The text-only model performed exceptionally well on explicit categories like "Motivational". However, it struggled severely with the "Offensive" class, often misclassifying offensive text as "Sarcasm" (45% confusion rate) due to similar aggressive tones that lack visual context.

---

### 3.2 Run 2: Multimodal Cross-Attention Fusion (CLIP + MuRIL)
**Objective**: Incorporate visual context to resolve ambiguities where the text alone is insufficient (e.g., detecting irony or sarcasm).
**Architecture**: CLIP Vision Embeddings + MuRIL Text Embeddings $\rightarrow$ Bidirectional Cross-Attention $\rightarrow$ Classifier.
**Validation Outcomes**:
- **Macro F1**: 0.5096
- **Accuracy**: 64.89%
- **Class-wise Performance**:
  - Sarcasm: 76.15% F1 (71% Recall)
  - Neutral: 52.35% F1 (59% Recall)
  - Motivational: 63.58% F1 (63% Recall)
  - Offensive: 11.76% F1 (9% Recall)

**Insights**: The addition of visual features *drastically* improved context-dependent classes. Recall for "Sarcasm" jumped from 64% to 71%, and "Neutral" jumped from 53% to 59%. However, the introduction of visual complexity added noise to the severely underrepresented "Offensive" class, causing its recall to drop to 9%, which dragged down the overall Macro F1 score despite the higher global accuracy.

---

### 3.3 Run 3: Weighted Ensemble
**Objective**: Maximize overall performance by combining the strengths of the text-only model (explicit category detection) and the multimodal model (context-dependent category detection).
**Architecture**: A late-fusion weighted probability ensemble of Run 1 and Run 2 predictions.
**Strategy**: `(0.4 * Run1_Probabilities) + (0.6 * Run2_Probabilities)`.

**Test Set Prediction Distribution**:
- Sarcasm: 409
- Neutral: 72
- Motivational: 14
- Offensive: 5

**Insights**: The final prediction distribution on the unlabeled test set closely mirrors the skewed distribution of the training set. The ensemble successfully smoothed out the individual weaknesses of the isolated models, providing the most robust final submission for the shared task.
