# Distorted Visual Sequence Pattern Recognition

A deep learning pipeline for recognizing text sequences from heavily distorted grayscale images — built entirely from scratch using PyTorch.

---

## Problem

Given a grayscale image containing a text sequence corrupted by visual distortions, predict the correct ordered sequence of characters. The primary distortion in this dataset is a **large solid black elliptical blob** covering 30–50% of the image, with additional background noise and graininess.

Evaluation metric: **Character Error Rate (CER)** based on Levenshtein Distance. Lower is better.

---

## Architecture

```
Distorted Image (1 × 64 × 128)
        │
        ▼
CNN Backbone          4 conv blocks, height-only pooling, W preserved
        │
        ▼
BiLSTM Encoder        2 layers, bidirectional, hidden size 256
        │
        ▼
Linear Projection     vocab_size output
        │
        ▼
CTC Decoding          Greedy decoder at inference
        │
        ▼
Predicted Text
```

Key design decisions:
- **Height-only pooling** in CNN preserves the full width dimension as the time sequence T for CTC alignment
- **Bidirectional LSTM** reads sequence in both directions — critical when the blob occludes middle characters
- **CTC loss** handles variable-length sequences without requiring character-level alignment
- **No pretrained weights** — trained entirely from scratch on the provided dataset

---

## Custom Augmentation

Standard augmentations (CoarseDropout, ElasticTransform) were rejected because they do not match the actual distortion pattern. A custom `LargeBlobOcclusion` augmentation was built using OpenCV to generate filled black ellipses that directly simulate what the model encounters at test time.

---

## Results

| Metric | Value |
|---|---|
| Best Validation CER | ~0.031 |
| Epochs trained | 30 |
| Training time per epoch | ~26s (Colab T4 GPU) |

---

## Notebook Structure

| Section | Description |
|---|---|
| 1 | Introduction & Problem Understanding |
| 2 | Environment Setup & Imports |
| 3 | Dataset Loading & EDA |
| 4 | Vocabulary & Label Encoding |
| 5 | Preprocessing Pipeline |
| 6 | Data Augmentation (incl. custom LargeBlobOcclusion) |
| 7 | Dataset Class & DataLoaders |
| 8 | Model Architecture (CRNN) |
| 9 | Loss Function & Decoder |
| 10 | Training Loop |
| 11 | Validation & CER Evaluation |
| 12 | Training Execution & Curves |
| 13 | Error Analysis |

---

## Stack

- **PyTorch** — model, training, CTC loss
- **Albumentations** — augmentation pipeline including custom transform
- **OpenCV / Pillow** — image loading and preprocessing
- **editdistance** — CER computation
- **Google Colab (T4 GPU)** — training infrastructure
- **Google Drive** — checkpoint persistence

---

## Setup

```bash
pip install torch torchvision albumentations editdistance opencv-python pandas numpy matplotlib tqdm scikit-learn
```

Dataset is expected in the following structure:
```
data/
├── train_images/
├── test_images/
└── train-labels.csv
```

---

## Submission Format

```
image,prediction
test-0.png,AXU323
test-1.png,BT91KD
```
