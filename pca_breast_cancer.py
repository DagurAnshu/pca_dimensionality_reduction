# Load dataset
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer(as_frame=True)
df = data.frame

print(df.head())
print(df.columns)
print(df.isnull().sum())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Preprocessing: Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA Implementation

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", pca.explained_variance_ratio_.sum())

# Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.values, edgecolors='k')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar(label='Target')
plt.show()


# Train-Test Split

from sklearn.model_selection import train_test_split

# With PCA
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# Without PCA (original scaled features)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model Selection

from sklearn.linear_model import LogisticRegression

model_pca = LogisticRegression(max_iter=1000)
model_raw = LogisticRegression(max_iter=1000)

# Training

model_pca.fit(X_train, y_train)
model_raw.fit(X_train2, y_train2)

# Prediction

y_pred = model_pca.predict(X_test)
y_pred2 = model_raw.predict(X_test2)

# Evaluation

from sklearn.metrics import accuracy_score

acc1 = accuracy_score(y_test, y_pred)
acc2 = accuracy_score(y_test2, y_pred2)

print("Accuracy with PCA:", acc1)
print("Accuracy without PCA:", acc2)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
acc1 = accuracy_score(y_test, y_pred)
acc2 = accuracy_score(y_test2, y_pred2)

print("Accuracy with PCA:", acc1)
print("Accuracy without PCA:", acc2)

# Confusion Matrix

print("\nConfusion Matrix (With PCA):")
print(confusion_matrix(y_test, y_pred))

print("\nConfusion Matrix (Without PCA):")
print(confusion_matrix(y_test2, y_pred2))

# Classification Report

print("\nClassification Report (With PCA):")
print(classification_report(y_test, y_pred))

print("\nClassification Report (Without PCA):")
print(classification_report(y_test2, y_pred2))

# -------------------------------
# Visual Confusion Matrix
# -------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# With PCA
cm_pca = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (With PCA)")
plt.show()

# Without PCA
cm_raw = confusion_matrix(y_test2, y_pred2)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Without PCA)")
plt.show()
