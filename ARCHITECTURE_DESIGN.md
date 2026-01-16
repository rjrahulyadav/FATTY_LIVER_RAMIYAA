# Architecture & System Design Documentation

## 🏗️ SIAMESE NEURAL NETWORK ARCHITECTURE

### Network Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIAMESE NETWORK ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────────┘

INPUT: Paired Images (Same Class or Different Class)
        ↓                                    ↓
    [Image 1]                          [Image 2]
   224×224 RGB                        224×224 RGB
        ↓                                    ↓
     ┌──────────────────────────────────────────┐
     │                                          │
     │    SHARED ENCODER: ResNet-50             │
     │  (Pre-trained on ImageNet)               │
     │                                          │
     │  Input: 224×224×3                        │
     │  Output: 2048-dim features               │
     │                                          │
     └──────────────────────────────────────────┘
          ↓                                  ↓
       [f₁ ∈ ℝ²⁰⁴⁸]                    [f₂ ∈ ℝ²⁰⁴⁸]
          ↓                                  ↓
     ┌────────────────────────┬─────────────────────────┐
     │                        │                         │
     ▼                        ▼                         ▼
[Projection Head]    [Projection Head]      [Classification Head]
[2048 → 512 → 128]   [2048 → 512 → 128]     [2048 → 5]
     ↓                        ↓                         ↓
[Embedding z₁]        [Embedding z₂]           [Logits]
 128-dim               128-dim                  5 classes
     ↓                        ↓                         ↓
  ┌─────────────────────────────────┐       [Classification]
  │  CONTRASTIVE LOSS (NT-Xent)     │
  │  ─ Maximize similarity(z₁, z₂)  │
  │    if same class                │       CROSS-ENTROPY LOSS
  │  ─ Minimize similarity if       │       ─ Minimize prediction
  │    different class              │         error on labels
  └─────────────────────────────────┘
          ↓
    [Loss Backprop]
          ↓
   [Gradient Update]


TWO FORWARD MODES:

Mode 1: CONTRASTIVE LEARNING (Pre-training)
  forward(image1, image2)
  → returns (proj1, proj2)
  → uses NTXentLoss
  → learns feature representations

Mode 2: CLASSIFICATION (Fine-tuning)
  forward(image)
  → returns logits
  → uses CrossEntropyLoss
  → predicts disease class
```

---

## 📊 DATA FLOW PIPELINE

### Training Data Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    DATASET LOADING                         │
│                                                            │
│  .mat File (MATLAB)              or  Image Folders        │
│  (10,500 augmented images)           (Normal/, Grade-I/)  │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│              FAT PERCENTAGE TO CLASS MAPPING               │
│                                                            │
│  fat_val < 5%          → Class 0 (Normal)                 │
│  5% ≤ fat_val ≤ 35%    → Class 1 (Grade-I)               │
│  35% < fat_val ≤ 65%   → Class 2 (Grade-II)              │
│  fat_val > 65%         → Class 3 (Grade-III)             │
│  class_val == CLD      → Class 4 (CLD)                    │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│           TRAIN/VAL/TEST SPLIT (80-10-10)                 │
│                                                            │
│  Training Set (80%)  → For model learning                 │
│  Validation Set (10%) → For hyperparameter tuning         │
│  Test Set (10%)      → For final evaluation              │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│               IMAGE AUGMENTATION PIPELINE                  │
│                                                            │
│  Input: Raw image                                         │
│    ↓                                                      │
│  Resize → 224×224 pixels                                 │
│    ↓                                                      │
│  Rotation → Random ±20° rotation                         │
│    ↓                                                      │
│  Affine → Random shear (10°) + zoom (0.8-1.2x)          │
│    ↓                                                      │
│  Flip → Horizontal (50%) + Vertical (50%)                │
│    ↓                                                      │
│  ColorJitter → Brightness, Contrast, Saturation         │
│    ↓                                                      │
│  Normalize → ImageNet stats                              │
│    ↓                                                      │
│  ToTensor → Convert to PyTorch tensor                    │
│    ↓                                                      │
│  Output: Augmented 224×224×3 tensor                      │
│                                                            │
│  Mean: [0.485, 0.456, 0.406]                             │
│  Std:  [0.229, 0.224, 0.225]                             │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│              BATCH CREATION & LOADING                      │
│                                                            │
│  Batch Size: 32 (configurable)                            │
│  Shuffle: True (training), False (val/test)              │
│  Num Workers: 4 (parallel loading)                        │
└────────────────────────────────────────────────────────────┘
                            ↓
                    [Ready for Training]
```

---

## 🔄 TWO-STAGE TRAINING PIPELINE

### Stage 1: Contrastive Pre-Training (Self-Supervised)

```
┌──────────────────────────────────────────────────────────┐
│         CONTRASTIVE PRE-TRAINING (Self-Supervised)       │
│              Epochs: 20 (default)                        │
└──────────────────────────────────────────────────────────┘

Input Batch:
    [Image 1, Image 2, ..., Image 32]
            ↓
    Augment each image differently
    (Same image, different augmentations)
            ↓
    Forward through Siamese Network
            ↓
    ┌─────────────────────────────────────────┐
    │  Output: Embedding Pairs                │
    │  (z₁₁, z₁₂), (z₂₁, z₂₂), ..., (z₃₂₁, z₃₂₂) │
    └─────────────────────────────────────────┘
            ↓
    ┌─────────────────────────────────────────┐
    │  NT-Xent Loss Computation               │
    │                                         │
    │  Loss = -(1/B) Σ log[                   │
    │      exp(sim(z_i, z_i+) / τ) /          │
    │      (Σ exp(sim(z_i, z_k) / τ))        │
    │    ]                                    │
    │                                         │
    │  Where:                                 │
    │  - sim() = cosine similarity             │
    │  - τ = temperature = 0.5                │
    │  - B = batch size = 32                  │
    └─────────────────────────────────────────┘
            ↓
    Backpropagation
            ↓
    Gradient Clipping (max_norm=1.0)
            ↓
    Adam Optimizer Step
            ↓
    Update encoder & projection head weights
            ↓
    Repeat for 20 epochs


Output: Pre-trained encoder with good feature representations
```

### Stage 2: Classification Training (Supervised)

```
┌──────────────────────────────────────────────────────────┐
│         CLASSIFICATION TRAINING (Supervised)             │
│              Epochs: 50 (default)                        │
└──────────────────────────────────────────────────────────┘

Training Loop:
    ┌─────────────────────────────────────────┐
    │  Input Batch: [Images, Labels]          │
    │  Labels: [0,1,2,3,4] (5 classes)        │
    └─────────────────────────────────────────┘
            ↓
    Forward Pass:
    Image → ResNet-50 Encoder → Classifier
            ↓
    ┌─────────────────────────────────────────┐
    │  Output: Class Logits                   │
    │  [batch_size, 5]                        │
    └─────────────────────────────────────────┘
            ↓
    Cross-Entropy Loss:
    Loss = -(1/B) Σ log[exp(z_ci) / Σ exp(z_j)]
            ↓
    Backpropagation
            ↓
    Gradient Clipping (max_norm=1.0)
            ↓
    Adam Optimizer Step
            ↓
    Update encoder & classifier weights


Validation Loop (Each Epoch):
    ┌─────────────────────────────────────────┐
    │  Evaluate on validation set              │
    │  Compute validation accuracy             │
    └─────────────────────────────────────────┘
            ↓
    If validation_acc > best_acc:
        Save model checkpoint
        best_acc = validation_acc
            ↓
    Continue training or early stop


Output: Best trained model (lowest validation loss)
        Saved as: best_model.pth
```

---

## 🎯 CLASSIFICATION CATEGORIES & MAPPING

### Fat Percentage Classification System

```
┌──────────────────────────────────────────────────────┐
│     FAT ACCUMULATION LEVELS & CLINICAL GRADES       │
└──────────────────────────────────────────────────────┘

Class 0: NORMAL
├─ Fat Content: < 5%
├─ Clinical Status: Healthy liver
├─ Ultrasound Appearance: Normal echogenicity
└─ Output Index: 0

Class 1: GRADE-I (MILD STEATOSIS)
├─ Fat Content: 5-35%
├─ Clinical Status: Mild fatty infiltration
├─ Ultrasound Appearance: Slightly increased echogenicity
└─ Output Index: 1

Class 2: GRADE-II (MODERATE STEATOSIS)
├─ Fat Content: 35-65%
├─ Clinical Status: Moderate fat accumulation
├─ Ultrasound Appearance: Increased echogenicity with vessel blurring
└─ Output Index: 2

Class 3: GRADE-III (SEVERE STEATOSIS)
├─ Fat Content: > 65%
├─ Clinical Status: Severe fat accumulation
├─ Ultrasound Appearance: Strong echogenicity, poor vessel visualization
└─ Output Index: 3

Class 4: CLD (CHRONIC LIVER DISEASE)
├─ Characteristics: Cirrhosis, fibrosis, scarring
├─ Clinical Status: Advanced liver damage
├─ Ultrasound Appearance: Heterogeneous echogenicity, nodular surface
└─ Output Index: 4


MODEL OUTPUT STRUCTURE:

Softmax Output:     [p₀, p₁, p₂, p₃, p₄]
Where:
  p₀ = P(Normal)
  p₁ = P(Grade-I)
  p₂ = P(Grade-II)
  p₃ = P(Grade-III)
  p₄ = P(CLD)

Prediction = argmax(p₀, p₁, p₂, p₃, p₄)

Confidence = max(p₀, p₁, p₂, p₃, p₄)
```

---

## 📈 EVALUATION METRICS FRAMEWORK

### Performance Metrics Calculation

```
┌────────────────────────────────────────────────────────┐
│              EVALUATION METRICS PIPELINE               │
└────────────────────────────────────────────────────────┘

Model Inference on Test Set:
    Model(test_images) → predictions
            ↓
    Softmax → probabilities
            ↓
    argmax → class predictions


METRIC 1: BINARY CLASSIFICATION ACCURACY
├─ Definition: Normal (0) vs Abnormal (1-4)
├─ Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
├─ Target: ≥ 99.90%
└─ Computation:
    binary_pred = (pred > 0).astype(int)
    binary_true = (true > 0).astype(int)
    accuracy = mean(binary_pred == binary_true)


METRIC 2: MULTI-CLASS ACCURACY
├─ Definition: Across all 5 classes
├─ Formula: Accuracy = (Correct Predictions) / (Total Predictions)
├─ Target: ≥ 99.77%
└─ Computation: accuracy = mean(pred == true)


METRIC 3: PER-CLASS METRICS
├─ Precision: TP / (TP + FP)  [How many predicted positives were correct]
├─ Recall: TP / (TP + FN)     [How many actual positives were found]
├─ F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
└─ For each class in [Normal, Grade-I, Grade-II, Grade-III, CLD]


METRIC 4: CONFUSION MATRIX
├─ Dimensions: 5×5 matrix
├─ Element [i,j]: Count of class i predicted as class j
├─ Diagonal elements: Correct predictions
└─ Off-diagonal elements: Misclassifications


METRIC 5: ROC-AUC SCORE
├─ Binary ROC-AUC: Target ≥ 0.990
│  └─ Measures: Normal vs Abnormal discrimination
│
├─ Multi-class ROC-AUC: Target ≥ 0.999
│  └─ Method: One-vs-Rest (OvR) approach
│  └─ Computes: AUC for each class vs all others
│  └─ Returns: Macro-averaged AUC
└─ Formula: Area under ROC curve
    (Sensitivity vs 1-Specificity at various thresholds)


METRIC 6: SENSITIVITY & SPECIFICITY (Per Class)
├─ Sensitivity = TP / (TP + FN)        [True Positive Rate]
├─ Specificity = TN / (TN + FP)        [True Negative Rate]
└─ Derived from confusion matrix


OUTPUT REPORT:
┌─────────────────────────────────────┐
│ Classification Report:              │
│ ─ Precision (per class)             │
│ ─ Recall (per class)                │
│ ─ F1-Score (per class)              │
│ ─ Support (samples per class)        │
├─────────────────────────────────────┤
│ Overall Metrics:                    │
│ ─ Binary Accuracy                   │
│ ─ Multi-class Accuracy              │
│ ─ Binary ROC-AUC                    │
│ ─ Multi-class ROC-AUC               │
├─────────────────────────────────────┤
│ Confusion Matrix:                   │
│ ─ 5×5 prediction breakdown          │
└─────────────────────────────────────┘
```

---

## 🔒 ROBUSTNESS & ERROR HANDLING

### Numerical Stability Measures

```
┌────────────────────────────────────────────────────────┐
│        NUMERICAL STABILITY & ERROR HANDLING           │
└────────────────────────────────────────────────────────┘

1. GRADIENT CLIPPING
   ├─ Max Norm: 1.0
   ├─ Applied after: Backpropagation
   ├─ Purpose: Prevent gradient explosion
   └─ Code: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


2. NaN/Inf DETECTION
   ├─ Check in projections: torch.isnan(proj1).any()
   ├─ Check in outputs: torch.isinf(outputs).any()
   ├─ Check in loss: torch.isnan(loss) or loss.item() > 1e6
   ├─ Action: Skip batch if detected
   └─ Purpose: Catch numerical issues early


3. LOSS VALUE VALIDATION
   ├─ Check: loss.item() > 1e6 (indicates instability)
   ├─ Action: Skip batch to prevent divergence
   └─ Logged: Warning messages for monitoring


4. CHECKPOINT VALIDATION
   ├─ Before saving: Check for NaNs in state dict
   ├─ Verify: All parameters are finite
   ├─ Action: Only save valid checkpoints
   └─ Purpose: Prevent corrupted models


5. BATCH HANDLING
   ├─ Check: images.shape[0] == 0
   ├─ Action: Skip empty batches
   └─ Purpose: Handle edge cases gracefully


6. TEMPERATURE SCALING (NTXentLoss)
   ├─ Minimum: 0.01 (prevent division by zero)
   ├─ Purpose: Stabilize contrastive loss
   └─ Range: 0.01 to 0.5 (adjustable)


STABILITY GUARANTEES:
  ✓ No gradient explosion (clipping)
  ✓ No numerical underflow/overflow (validation)
  ✓ Graceful error handling (skip problematic batches)
  ✓ Valid checkpoint persistence (pre-save validation)
```

---

## 📁 PROJECT STRUCTURE MAPPING

```
fatty-liver-classification/
│
├── models/
│   └── siamese_net.py              # ✅ Siamese network architecture
│       ├─ SiameseNetwork class
│       ├─ ResNet-50 encoder (pre-trained)
│       ├─ Projection head (2048 → 512 → 128)
│       ├─ Classification head (2048 → 5)
│       └─ Dual forward modes (contrastive & classification)
│
├── src/
│   └── data_loader.py               # ✅ Data loading & augmentation
│       ├─ FattyLiverDataset class
│       ├─ ContrastiveDataset class
│       ├─ .mat file loading
│       ├─ Augmentation pipeline (7 techniques)
│       ├─ Train/val/test splitting (80-10-10)
│       └─ Class mapping (fat % → class)
│
├── utils/
│   └── losses.py                    # ✅ Loss functions
│       ├─ ContrastiveLoss class
│       ├─ NTXentLoss class
│       ├─ Temperature scaling
│       └─ Normalization
│
├── scripts/
│   ├── train.py                     # ✅ Two-stage training
│   │   ├─ train_contrastive() [Stage 1]
│   │   ├─ train_classification() [Stage 2]
│   │   ├─ Gradient clipping
│   │   ├─ NaN detection
│   │   └─ Best model checkpointing
│   │
│   └── evaluate.py                  # ✅ Comprehensive evaluation
│       ├─ evaluate_model()
│       ├─ Binary accuracy
│       ├─ Multi-class accuracy
│       ├─ ROC-AUC scores
│       ├─ Classification report
│       └─ Confusion matrix
│
├── infer.py                         # ✅ Single image inference
│   ├─ infer_image()
│   ├─ .mat file image loading
│   └─ Probability output
│
├── main.py                          # ✅ CLI entry point
│   ├─ download_data() [instructions]
│   ├─ train_model() [training]
│   └─ evaluate_model() [evaluation]
│
├── best_model.pth                   # Model checkpoint (saved)
├── best_model.pth.backup            # Backup checkpoint
│
├── data/                            # Dataset directory (to download)
│   └── dataset_liver_bmodes_steatosis_assessment_IJCARS.mat
│
├── README.md                        # Project documentation
├── TODO.md                          # Task tracking
├── PROJECT_COMPLIANCE_REPORT.md     # Compliance verification
└── COMPLIANCE_CHECKLIST.md          # Quick reference checklist


KEY FILES BY REQUIREMENT:

Architecture:
  → models/siamese_net.py

Data Processing:
  → src/data_loader.py

Training:
  → scripts/train.py
  → utils/losses.py

Evaluation:
  → scripts/evaluate.py

Inference:
  → infer.py

Entry Point:
  → main.py
```

---

## 🔗 CONTROL FLOW DIAGRAMS

### User Interaction Flow

```
USER COMMANDS:
    ↓
┌─────────────────────────────────────────┐
│       python main.py [COMMAND]          │
└─────────────────────────────────────────┘
            ↓
        ┌───┴────────────┬──────────┬──────────┐
        ↓                ↓          ↓          ↓
   download_data    train        evaluate   (inference)
        ↓                ↓          ↓
    Manual DL        scripts/   scripts/
    from Kaggle      train.py   evaluate.py
                        ↓          ↓
                    [Training]  [Evaluation]
                        ↓          ↓
                    best_model  Metrics
                     .pth       (Accuracy,
                                 AUC, etc)
```

### Internal Execution Flow (Training)

```
TRAINING EXECUTION:

python main.py train
    ↓
main.py::train_model()
    ↓
subprocess: scripts/train.py
    ↓
main()
    ├─ Parse arguments
    ├─ Set device (CUDA or CPU)
    ├─ Load data
    │  └─ get_data_loaders() / get_contrastive_data_loader()
    ├─ Create model
    │  └─ SiameseNetwork().to(device)
    │
    ├─ STAGE 1: Contrastive Pre-training
    │  ├─ train_contrastive()
    │  ├─ Use: NTXentLoss
    │  ├─ Epochs: 20
    │  └─ Output: Pre-trained encoder
    │
    ├─ STAGE 2: Classification Training
    │  ├─ train_classification()
    │  ├─ Use: CrossEntropyLoss
    │  ├─ Epochs: 50
    │  ├─ Validation: Each epoch
    │  ├─ Best model: Saved
    │  └─ Output: best_model.pth
    │
    └─ Checkpoint validation
       └─ Verify no NaNs
```

---

## ⚡ PERFORMANCE CHARACTERISTICS

### Computational Requirements

```
Training:
├─ GPU Memory: ~4-6 GB (batch_size=32)
├─ Training Time: ~2-4 hours (depending on hardware)
├─ Batch Processing: 32 images/batch
└─ Total Epochs: 70 (20 contrastive + 50 classification)

Inference:
├─ Single Image: ~50-100 ms
├─ Throughput: 10-20 images/second (batch processing)
└─ Model Size: ~100 MB (ResNet-50 checkpoint)

Evaluation:
├─ Test Set Evaluation: ~5-10 minutes
└─ Metrics Computation: Real-time
```

---

## 📊 SUMMARY

This document provides comprehensive technical documentation of the Fatty Liver Classification project architecture and implementation, validating all requirements across:

- ✅ **Architecture:** Siamese network with ResNet-50
- ✅ **Data Processing:** Full augmentation pipeline
- ✅ **Training:** Two-stage self-supervised + supervised
- ✅ **Classification:** 5-class disease grading
- ✅ **Evaluation:** Complete metrics framework
- ✅ **Robustness:** Numerical stability & error handling

**All technical specifications are met and validated.**
