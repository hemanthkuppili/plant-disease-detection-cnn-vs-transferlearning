# 🌿 Comparative Analysis of Custom CNN and Pretrained Transfer Learning Models for Plant Disease Detection

This repository contains the implementation and research study on **plant disease detection** using both a **custom lightweight Convolutional Neural Network (CNN)** and **pretrained transfer learning models** — **VGG16**, **ResNet50**, and **MobileNetV2**.  
The project aims to identify plant leaf diseases efficiently while comparing the trade-off between **accuracy** and **computational efficiency** for sustainable AI applications in agriculture.

---

## 📘 Abstract

Accurate and early identification of plant diseases is essential for sustainable agriculture.  
This research performs a **comparative evaluation** between a **custom CNN** and three **pretrained transfer learning models (VGG16, ResNet50, MobileNetV2)** trained on the **PlantVillage dataset**.

All models use similar preprocessing, augmentation, and regularization techniques.  
Results show that while **transfer learning models** achieve slightly higher accuracy, the **custom CNN** performs competitively with **significantly fewer parameters**, making it suitable for **edge and low-resource devices**.

---

## 🧠 Key Features

- Custom lightweight **CNN** architecture optimized for resource-limited environments.  
- Comparison with **VGG16**, **ResNet50**, and **MobileNetV2** using **transfer learning**.  
- Advanced **data augmentation** and **regularization** (Dropout, L2, Batch Normalization).  
- Visual insights using **Grad-CAM** for model interpretability.  
- Reproducible training pipeline using **Google Colab + TensorFlow 2.x**.

---

## 🗂️ Dataset

**Dataset Used:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

- 54,000+ leaf images  
- 38 classes (including healthy and diseased categories)  
- Images resized to **128x128 pixels**  
- Augmentations:
  - Rotation ±20°
  - Width/height shift up to 20%
  - Zoom, shear, and horizontal flip  

---

## ⚙️ Methodology

### 🏗️ 1. Data Preprocessing
- Resize and normalize images  
- Apply real-time augmentation during training  

### 🧩 2. Model Architectures
| Model | Type | Parameters (M) | Description |
|--------|------|----------------|-------------|
| Custom CNN | From scratch | 3.2 | 4 Conv blocks, BatchNorm, Dropout, L2 Regularization |
| VGG16 | Transfer Learning | 14.7 | Classic architecture with fine-tuned top layers |
| ResNet50 | Transfer Learning | 23.5 | Residual connections for deep learning stability |
| MobileNetV2 | Transfer Learning | 3.4 | Lightweight model for mobile/edge devices |

### ⚙️ 3. Training Setup
- Optimizer: **Adam (lr=0.0005)**  
- Batch size: **32**  
- Epochs: **20 (with EarlyStopping & ReduceLROnPlateau)**  
- Framework: **TensorFlow 2.x / Keras**

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | Params (M) | Epochs |
|--------|-----------|------------|---------|-----------|-------------|---------|
| Custom CNN | 0.87 | 0.85 | 0.84 | 0.84 | 3.2 | 15 |
| VGG16 | 0.91 | 0.90 | 0.89 | 0.89 | 14.7 | 12 |
| ResNet50 | 0.93 | 0.91 | 0.92 | 0.91 | 23.5 | 14 |
| MobileNetV2 | 0.90 | 0.88 | 0.87 | 0.87 | 3.4 | 13 |

**Key Insights:**
- **ResNet50** achieved the highest accuracy.  
- **Custom CNN** offers competitive accuracy with minimal computational cost.  
- **MobileNetV2** provides a balance between performance and efficiency.

---

## 🌱 Discussion

- **Custom CNN** and **MobileNetV2** are ideal for **on-device** plant disease detection due to low computational requirements.  
- **Transfer Learning models** (ResNet50, VGG16) are more suitable for **cloud-based** analysis.  
- **Grad-CAM visualizations** enhance interpretability and build user trust in model predictions.

---
