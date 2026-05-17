# -*- coding: utf-8 -*-
"""
Decision Tree Model Training and Evaluation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            confusion_matrix, classification_report)
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('iris.csv')

# FIX: Remove duplicates before training (iris.csv has 1 duplicate)
df = df.drop_duplicates()
print(f"Dataset shape after removing duplicates: {df.shape}")

# Preprocessing - FIX: Use species_name (text labels) instead of numeric species
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['species_name']  # Text labels: setosa, versicolor, virginica

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree - FIX: max_depth capped at 5
print("=== TRAINING DECISION TREE ===")
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = dt_classifier.predict(X_test_scaled)

# Metrics
print("\n=== MODEL PERFORMANCE ===")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n=== CLASSIFICATION REPORT ===")
# FIX: Use text class names
class_names = [str(c) for c in label_encoder.classes_]
print(classification_report(y_test, y_pred, target_names=class_names))

# Save model and artifacts
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n=== MODEL SAVED ===")
print("Saved: decision_tree_model.pkl")
print("Saved: scaler.pkl")
print("Saved: label_encoder.pkl")

# Plot and save decision tree
print("\nGenerating decision tree visualization...")
plt.figure(figsize=(20, 12))
plot_tree(dt_classifier, 
          feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
          class_names=class_names,  # FIX: Use text names
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree for Iris Classification')
plt.tight_layout()
plt.savefig('decision_tree_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: decision_tree_plot.png")
