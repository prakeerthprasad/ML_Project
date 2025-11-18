# MLB Pitch Classification Using Deep Learning

Automatic classification of baseball pitch types from broadcast video using CNN-LSTM architecture.

Authors: Jianchen Hong, Prakeerth Prasad, Amitha Javare Gowda  
Course: MSAI 349 - Machine Learning  
Institution: Northwestern University  
Date: November 2024

---

## Project Overview

This project develops a deep learning system to automatically classify baseball pitch types from video footage. Using the MLB-YouTube dataset, we trained a CNN-LSTM model to distinguish between six pitch types: Fastball, Slider, Curveball, Changeup, Sinker, and Knucklecurve.

### Key Results
- Best Validation Accuracy: 65.17%
- Test Accuracy: 56.04%
- Random Baseline: 16.67%
- Improvement: +39.37 percentage points

---

## Dataset

MLB-YouTube Dataset (Piergiovanni & Ryoo, CVsports 2018)

Source: https://github.com/piergiaj/mlb-youtube

The dataset consists of 20 MLB postseason games from the 2017 season, totaling approximately 42 hours of broadcast footage. Due to hardware constraints, we downloaded 22 full game videos and extracted 599 annotated clips for training and evaluation. The dataset comes pre-annotated with pitch types and speeds, eliminating the need for manual labeling.

Class Distribution:
- Fastball (FF): 277 clips (46.2%)
- Slider (SL): 132 clips (22.0%)
- Knucklecurve: 67 clips (11.2%)
- Sinker (SI): 55 clips (9.2%)
- Curveball (CU): 41 clips (6.8%)
- Changeup (CH): 27 clips (4.5%)

Data Split: 70% training (419 clips), 15% validation (89 clips), 15% test (91 clips)

---

## Model Architecture

We implemented a CNN-LSTM hybrid architecture that combines spatial and temporal feature extraction.

Components:
- CNN (ResNet18): Extracts spatial features from each video frame (512-dimensional feature vectors)
- LSTM (2 layers, 256 hidden units): Models temporal patterns across the sequence of 16 frames
- Classification Head: Fully connected layers that map LSTM outputs to 6 pitch type predictions

The CNN processes each frame independently to capture visual patterns like arm angles and ball position, while the LSTM models how these patterns evolve over time to distinguish between different pitch types.

Training Configuration:
- Optimizer: Adam with learning rate 0.0001
- Regularization: Dropout (0.5), weight decay (0.0001), data augmentation
- Batch Size: 8
- Epochs: 25 (best model saved at epoch 16)
- Hardware: Google Colab Pro with NVIDIA A100 GPU
- Training Time: Approximately 90 minutes

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 65.17% |
| Test Accuracy | 56.04% |
| Training Accuracy | 91.65% |
| Weighted F1-Score | 0.55 |

### Per-Class Performance

| Pitch Type | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Fastball (FF) | 0.60 | 0.78 | 0.68 |
| Slider (SL) | 0.93 | 0.57 | 0.70 |
| Curveball (CU) | 0.43 | 0.43 | 0.43 |
| Knucklecurve | 0.25 | 0.29 | 0.27 |
| Sinker (SI) | 0.11 | 0.14 | 0.12 |
| Changeup (CH) | 0.00 | 0.00 | 0.00 |

Key Findings:
- The model performs well on fastballs (78% recall) and sliders (93% precision)
- Significant overfitting observed with 35% gap between training and validation accuracy
- Class imbalance strongly impacts performance on rare pitch types
- Complete failure on changeups suggests the model relies heavily on velocity patterns rather than mechanical differences

---
