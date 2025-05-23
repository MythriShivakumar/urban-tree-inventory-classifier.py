# Urban Tree Inventory Classifier

This notebook presents a complete preprocessing and classification pipeline for a tree inventory dataset using a neural network built with PyTorch. It addresses real-world data challenges and applies robust ML practices.

## 🌳 Dataset

- `Tree_Inventory_20250211.csv`
- Contains data on tree species, location, condition, and other attributes

## 🧹 Data Processing

- Dropping nulls and ensuring type consistency
- Lowercasing categorical text data
- Handling outliers via IQR-based filtering
- Standardizing numeric features

## 🧠 Model

- Neural network built with PyTorch
- Train/test split using `sklearn`
- Training tracked with loss and accuracy over epochs
- Optimizer: Adam
- Loss Function: Cross-Entropy

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve and AUC

## 🧰 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
