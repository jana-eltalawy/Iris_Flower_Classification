# -*- coding: utf-8 -*-
"""
Data Preprocessing for Iris Decision Tree Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv('iris.csv')

print("=== ORIGINAL DATASET ===")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Check for missing values
print(f"\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Check for duplicates
print(f"\n=== DUPLICATES ===")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove duplicates if any
df_clean = df.drop_duplicates()
print(f"Shape after removing duplicates: {df_clean.shape}")

# Separate features and target
X = df_clean[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# FIX: Use species_name (text) for encoding so we get flower names, not numbers
y = df_clean['species_name']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n=== FEATURES (X) ===")
print(X.head())
print(f"\nFeature names: {list(X.columns)}")

print(f"\n=== TARGET (y) ===")
print(f"Classes: {label_encoder.classes_}")
print(f"Encoded: {np.unique(y_encoded)}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n=== TRAIN-TEST SPLIT ===")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Feature scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n=== SCALING ===")
print(f"Training features scaled: {X_train_scaled.shape}")
print(f"Testing features scaled: {X_test_scaled.shape}")

# Save preprocessed data
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save scaler and encoder for later use
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n=== PREPROCESSING COMPLETE ===")
print("Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("Saved: scaler.pkl, label_encoder.pkl")
