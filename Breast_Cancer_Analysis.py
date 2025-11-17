# ==========================================
# Breast Cancer Data Analysis and ML Project
# ==========================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. LOAD DATA
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = Malignant, 1 = Benign

# Preview
print(df.head())
print(df.info())
print(df.describe())

# 3. VISUALIZATION

# 3.1 Class distribution
plt.figure(figsize=(6,4))
plt.bar(["Malignant", "Benign"], df["target"].value_counts().sort_index())
plt.title("Distribution of Tumor Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 3.2 Correlation heatmap
plt.figure(figsize=(14,10))
corr = df.corr()
plt.imshow(corr, cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()

# 3.3 Scatter plot: mean radius vs mean texture
plt.figure(figsize=(6,4))
plt.scatter(df["mean radius"], df["mean texture"], c=df["target"])
plt.title("Mean Radius vs Mean Texture")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.colorbar(label="Target (0=Malignant, 1=Benign)")
plt.show()

# 4. PREPARE DATA FOR MACHINE LEARNING
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 5. MACHINE LEARNING MODEL
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 6. EVALUATION
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. CONCLUSION
print("\nAll visualizations displayed above.")
print("ML model executed successfully with printed accuracy and classification metrics.")

# 8. FEATURE IMPORTANCE VISUALIZATION

# Extract feature importances
importances = model.feature_importances_
feature_names = X.columns

# Sort by importance
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(12,8))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.title("Random Forest Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()