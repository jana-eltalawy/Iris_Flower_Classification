# -*- coding: utf-8 -*-
"""
Data Visualization for Iris Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('iris.csv')

# FIX: Remove duplicates for cleaner visualizations
df = df.drop_duplicates()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Pairplot - Feature relationships
print("Generating pairplot...")
fig = sns.pairplot(df, hue='species_name', 
                   vars=['sepal length (cm)', 'sepal width (cm)', 
                         'petal length (cm)', 'petal width (cm)'],
                   palette='viridis', diag_kind='kde')
fig.fig.suptitle('Iris Dataset - Pairplot of Features', y=1.02, fontsize=14)
plt.savefig('visualization_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualization_pairplot.png")

# 2. Correlation Heatmap
print("Generating correlation heatmap...")
plt.figure(figsize=(8, 6))
corr_matrix = df[['sepal length (cm)', 'sepal width (cm)', 
                  'petal length (cm)', 'petal width (cm)']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('visualization_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualization_heatmap.png")

# 3. Distribution plots for each feature
print("Generating distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ['sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)']

for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    for species in df['species_name'].unique():
        subset = df[df['species_name'] == species]
        ax.hist(subset[feature], alpha=0.6, label=species, bins=15)
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title(f'Distribution of {feature}')

plt.suptitle('Feature Distributions by Species', fontsize=14)
plt.tight_layout()
plt.savefig('visualization_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualization_distributions.png")

# 4. Box plots
print("Generating box plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    sns.boxplot(data=df, x='species_name', y=feature, ax=ax, palette='viridis')
    ax.set_title(f'{feature} by Species')

plt.suptitle('Box Plots of Features by Species', fontsize=14)
plt.tight_layout()
plt.savefig('visualization_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualization_boxplots.png")

# 5. Class distribution
print("Generating class distribution...")
plt.figure(figsize=(8, 5))
species_counts = df['species_name'].value_counts()
colors = ['#440154', '#21918c', '#fde725']
plt.bar(species_counts.index, species_counts.values, color=colors)
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Class Distribution in Iris Dataset')
for i, v in enumerate(species_counts.values):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: visualization_class_distribution.png")

print("\n=== ALL VISUALIZATIONS COMPLETE ===")
