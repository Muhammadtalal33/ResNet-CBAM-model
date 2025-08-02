
# Brain Tumor Classification using CNN with CBAM Attention (PyTorch)

## Abstract

Brain tumor classification is a crucial task in medical imaging, assisting radiologists in diagnosing tumor types from MRI scans. This project presents a Convolutional Neural Network (CNN) architecture augmented with the Convolutional Block Attention Module (CBAM) to improve classification accuracy. The CBAM module refines feature representations by applying sequential channel and spatial attention mechanisms, allowing the model to focus on the most informative regions of MRI images.

The dataset used includes four classes: **No Tumor, Pituitary Tumor, Meningioma Tumor, and Glioma Tumor**, sourced from a publicly available Kaggle dataset. The model is trained using PyTorch with data augmentation techniques, early stopping, and a learning rate scheduler. Evaluation metrics include accuracy and a confusion matrix visualization to assess model performance.

---

## Working Explanation

### 1. **Dataset Preparation**
- The dataset directory is structured into **Training** and **Testing** folders.
- Each folder contains subdirectories for tumor types.
- Images are loaded into a Pandas DataFrame with columns: `image_path`, `label`, and `split`.
- A custom PyTorch Dataset class (`BrainTumorDataset`) handles image loading and transformation.

### 2. **Data Augmentation & Preprocessing**
- Training images undergo resizing, random horizontal flipping, rotation, normalization, and tensor conversion.
- Validation and Test images are resized and normalized without augmentation.

### 3. **Model Architecture (CNN + CBAM)**
- The CNN consists of **4 Convolutional Blocks** each followed by:
  - ReLU activation
  - CBAM Attention Block (Channel & Spatial attention)
  - MaxPooling layer for spatial reduction.
- A Fully Connected Classifier Head maps extracted features to 4 output classes.
- CBAM enhances the model's ability to focus on relevant regions in the images.

### 4. **Training Pipeline**
- The model uses **CrossEntropyLoss** and **Adam Optimizer**.
- A **StepLR Scheduler** reduces the learning rate after every 7 epochs.
- **EarlyStopping** is implemented to halt training if validation accuracy doesn't improve for a specified patience (default: 5 epochs).

### 5. **Model Evaluation**
- The best model (saved using early stopping) is evaluated on the test dataset.
- Accuracy score and a **Confusion Matrix Heatmap** are generated to visualize classification performance across all classes.

### 6. **Visualizations**
- Training and Validation **Loss Curves**.
- Training and Validation **Accuracy Curves**.
- Confusion Matrix plotted using Seaborn.

---

## How to Run
1. Place the dataset in the specified directory (`/kaggle/input/brain-tumor-mri-dataset`).
2. Run the Python script.
3. Monitor training/validation losses and accuracies.
4. The best-performing model will be saved as `best_model.pth`.
5. Evaluate the model to see test accuracy and confusion matrix.

---

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- PIL (Pillow)

---

## Dataset Source
[Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

---
