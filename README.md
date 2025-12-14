# pca_dimensionality_reduction
Breast Cancer classification using PCA and Logistic Regression with detailed evaluation and visualization.
# Breast Cancer Classification using PCA and Logistic Regression

This project presents an end-to-end machine learning experiment on the
Breast Cancer Wisconsin dataset, focusing on understanding the impact of
**Principal Component Analysis (PCA)** on classification performance using
**Logistic Regression**.

The project was developed and executed in a **local Python environment using PyCharm**.

---

## Overview

- **Dataset:** Breast Cancer Wisconsin Dataset (scikit-learn)
- **Task:** Binary classification (Malignant vs Benign)
- **Models:** Logistic Regression
- **Key Techniques:**
  - Feature scaling
  - Dimensionality reduction (PCA)
  - Model comparison (with vs without PCA)
  - Visualization and detailed evaluation

---

## Dataset Description

- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numerical features
- **Target Classes:**
  - '0' → Malignant
  - '1' → Benign

The dataset is clean and contains **no missing values**.

---

## Methodology

### Data Preprocessing
- Features are standardized using `StandardScaler`
- Standardization is applied before PCA to ensure equal feature contribution

### PCA Implementation
- PCA is applied with **2 principal components**
- Explained variance ratio:
  - PC1: **44.27%**
  - PC2: **18.97%**
- **Total variance retained:** **63.24%**

The first two components capture a significant portion of the dataset’s structure,
allowing meaningful dimensionality reduction.

---

## Visualization

- A 2D scatter plot of the first two principal components
- Data points are color-coded by class label
- Visualization reveals clear class separation in reduced space

---

## Model Training

Two Logistic Regression models are trained for comparison:

1. **With PCA**
   - Input features: 2 principal components
2. **Without PCA**
   - Input features: Original scaled features (30 dimensions)

Both models use the same train–test split (80/20) to ensure a fair comparison.

---

## Experimental Results

### Accuracy Comparison

| Model | Accuracy |
|------|----------|
| Logistic Regression (With PCA) | **99.12%** |
| Logistic Regression (Without PCA) | 97.37% |

Despite using only two components, the PCA-based model achieves higher accuracy.

---

### Confusion Matrix Analysis

**With PCA**
[[42 1]]
[ 0 71]]

- False Positives: 1  
- False Negatives: 0  

**Without PCA**
[[41 2]
[ 1 70]]

- False Positives: 2  
- False Negatives: 1  

The PCA-based model eliminates false negatives, which is particularly
important in medical diagnosis tasks.

---

### Classification Report Summary

- **With PCA:** Precision, recall, and F1-score ≈ **0.99** for both classes
- **Without PCA:** Slightly lower recall and F1-score

Overall, PCA improves generalization by reducing noise and feature redundancy.

---

## Key Observations

- Dimensionality reduction does **not necessarily reduce performance**
- PCA can improve model generalization by removing correlated features
- Logistic Regression benefits from orthogonal, low-dimensional inputs
- Careful evaluation is essential beyond accuracy alone

---

## Project Structure
Dimensionality_Reduction/
1. PCA_Breast_Cancer_Data.py
2. README.md

---

## How to Run (PyCharm / Local Environment)

1. Clone the repository
2. Open the project in **PyCharm**
3. Install dependencies:
pip install pandas scikit-learn matplotlib seaborn
4. 4. Run the script:
python breast_cancer_pca.py
---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyCharm (IDE)

---

## Learning Focus

This project focuses on:
- Understanding PCA and dimensionality reduction
- Designing fair model comparisons
- Applying robust evaluation metrics
- Interpreting results in a real-world context

This project is intended for **learning and experimentation**,
not for production deployment.

---

This project is created as part of a structured learning process
to build strong foundations in machine learning, data preprocessing and to implement dimensionality reduction and analyze its effect.

