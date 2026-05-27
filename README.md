# White Blood Cell Classification

> Classifying 8 types of white blood cells from microscopic images using deep learning.

## Overview

This project was developed for the **AN2DL – First Homework**. The goal is to automatically classify white blood cells (e.g., Neutrophil, Eosinophil, Lymphocyte, etc.) from 96×96 RGB images. We leveraged pretrained CNNs for feature extraction and addressed class imbalance through class weighting and data augmentation.

## Key Achievements

- **Online test accuracy:** 85%
- **Best validation F1 score:** 98.75%
- **Model:** EfficientNetV2M (pretrained on ImageNet)
- **Techniques:** Transfer learning, fine-tuning, RandAugment, class weights, Grad-CAM interpretability

## Dataset

- 13,759 images (after outlier removal)
- 8 balanced classes in test set
- Outliers (e.g., Shrek, Rick Roll) removed via truncation

## Methodology

1. **Preprocessing**  
   - Removed outliers using t-SNE + visual inspection  
   - Train/validation split with class imbalance preserved; test set balanced

2. **Handling class imbalance**  
   - Class weights during training (outperformed synthetic data)  
   - F1 score as primary metric

3. **Feature extraction**  
   - Compared 9 CNNs: VGG16, ResNet50, DenseNet121, InceptionV3, MobileNetV3Large, EfficientNetB0/V2S/V2M/V2L  
   - EfficientNetV2M chosen for best trade-off (F1 96.80%)

4. **Data augmentation**  
   - RandAugment significantly improved generalization  
   - Tried MixUp, CutMix, traditional transforms

5. **Classifier head**  
   - Single dense layer (128 units) + dropout  
   - Hyperparameter tuning: LR = 1e-4, unfrozen layers = 60%

6. **Explainability**  
   - Grad-CAM visualizations to highlight regions used for classification

## Results

| Configuration | Validation F1 (%) | Online Score (%) |
|---------------|------------------|------------------|
| Baseline       | 92.31            | -                |
| Class weights  | 96.78            | -                |
| + RandAugment  | 97.69            | 85.00            |
| Final model    | 98.75            | 85.00            |

## Repository Structure
