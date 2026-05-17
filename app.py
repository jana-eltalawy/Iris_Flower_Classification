# -*- coding: utf-8 -*-
"""
Iris Flower Classification - Streamlit App
Decision Tree Classifier with Data Preprocessing, Visualization, and Metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            confusion_matrix, classification_report)

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# Title
st.title("🌸 Iris Flower Classification using Decision Tree")
st.markdown("---")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('iris.csv')
    # FIX: Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df

df = load_data()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", 
    ["Dataset", "Data Visualization", 
     "Model Training", "Evaluation Metrics", "Prediction"])

# ============================================
# PAGE 1: DATASET
# ============================================
if page == "Dataset":
    st.header("📊 Iris Dataset")

    st.markdown("""
    The **Iris dataset** is a classic dataset in machine learning, containing 150 samples 
    of iris flowers from three species:
    - **Setosa**
    - **Versicolor** 
    - **Virginica**

    Each sample has 4 features:
    - Sepal Length (cm)
    - Sepal Width (cm)
    - Petal Length (cm)
    - Petal Width (cm)
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Class Distribution")
    species_counts = df['species_name'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(species_counts)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#440154', '#21918c', '#fde725']
        ax.bar(species_counts.index, species_counts.values, color=colors)
        ax.set_xlabel('Species')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        for i, v in enumerate(species_counts.values):
            ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)

# ============================================
# PAGE 2: DATA VISUALIZATION
# ============================================
if page == "Data Visualization":
    st.header("📈 Data Visualization")

    viz_option = st.selectbox("Select Visualization:", [
        "Correlation Heatmap", "Box Plots"
    ])

    if viz_option == "Correlation Heatmap":
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = df[['sepal length (cm)', 'sepal width (cm)', 
                          'petal length (cm)', 'petal width (cm)']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)

    elif viz_option == "Box Plots":
        st.subheader("Box Plots by Species")
        features = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, feature in enumerate(features):
            ax = axes[idx // 2, idx % 2]
            sns.boxplot(data=df, x='species_name', y=feature, ax=ax, palette='viridis')
            ax.set_title(f'{feature} by Species')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================
# PAGE 3: MODEL TRAINING
# ============================================
if page == "Model Training":
    st.header("🤖 Decision Tree Model Training")

    # Prepare data - FIX: Use species_name (text labels) instead of numeric species
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    y = df['species_name']  # Text labels: setosa, versicolor, virginica

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    st.subheader("Model Configuration")
    # FIX: Max depth capped at 5 (was 10)
    max_depth = st.slider("Max Depth:", 1, 5, 3)
    random_state = st.number_input("Random State:", 0, 100, 42)

    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt_classifier.fit(X_train_scaled, y_train)

    st.success("✅ Model trained successfully!")

    # Save model
    st.session_state['model'] = dt_classifier
    st.session_state['scaler'] = scaler
    st.session_state['label_encoder'] = label_encoder
    st.session_state['X_test'] = X_test_scaled
    st.session_state['y_test'] = y_test

    # Show decision tree
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 12))

    # Class names are now actual flower names
    class_names_list = [str(c) for c in label_encoder.classes_]

    plot_tree(dt_classifier, 
              feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
              class_names=class_names_list,
              filled=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title('Decision Tree for Iris Classification')
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['sepal length', 'sepal width', 'petal length', 'petal width'],
        'Importance': dt_classifier.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)
    st.dataframe(importance_df)

# ============================================
# PAGE 4: EVALUATION METRICS
# ============================================
if page == "Evaluation Metrics":
    st.header("📊 Model Evaluation Metrics")

    # Check if model exists in session
    if 'model' not in st.session_state:
        # Auto-train for convenience
        X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
        y = df['species_name']  # FIX: Use text labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # FIX: Default max_depth=3 (not 5) to show impure leaves and uncertain probabilities
        dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
        dt_classifier.fit(X_train_scaled, y_train)

        st.session_state['model'] = dt_classifier
        st.session_state['scaler'] = scaler
        st.session_state['label_encoder'] = label_encoder
        st.session_state['X_test'] = X_test_scaled
        st.session_state['y_test'] = y_test
        st.info("Model auto-trained for demonstration (max_depth=3 to show probability uncertainty).")

    model = st.session_state['model']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    label_encoder = st.session_state['label_encoder']

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    col1, col2, col3 = st.columns(3)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")

    # NOTE: Explain accuracy vs confidence
    st.info("""
    **Note about Accuracy vs. Prediction Confidence:**
    - **Accuracy** measures how many test samples were correctly classified (e.g., 93% means 28/30 correct).
    - **Prediction Probability** shows how confident the model is for a specific input. 
    - With Decision Trees, probabilities come from the fraction of classes in a leaf node. 
    - **max_depth=5** often creates pure leaves → probabilities are always 1.0 (100% confident, even when wrong).
    - **max_depth=3** creates impure leaves → you will see real probabilities like [0.0, 0.87, 0.13] showing uncertainty.
    - If you want to see uncertain predictions, train with **max_depth=2 or 3**.
    """)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    # Class names are now actual flower names
    class_names_list = [str(c) for c in label_encoder.classes_]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names_list,
                yticklabels=class_names_list, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=class_names_list, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Raw confusion matrix values
    st.subheader("Confusion Matrix Values")
    cm_df = pd.DataFrame(cm, 
                         index=[f"Actual {c}" for c in class_names_list],
                         columns=[f"Predicted {c}" for c in class_names_list])
    st.dataframe(cm_df)

# ============================================
# PAGE 5: PREDICTION
# ============================================
if page == "Prediction":
    st.header("🔮 Predict Iris Species")

    # Check if model exists
    if 'model' not in st.session_state:
        # Auto-train
        X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
        y = df['species_name']  # FIX: Use text labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
        dt_classifier.fit(X_train_scaled, y_train)

        st.session_state['model'] = dt_classifier
        st.session_state['scaler'] = scaler
        st.session_state['label_encoder'] = label_encoder
        st.info("Model auto-trained for prediction (max_depth=3).")

    model = st.session_state['model']
    scaler = st.session_state['scaler']
    label_encoder = st.session_state['label_encoder']

    st.markdown("Enter the flower measurements below to predict the species:")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input("Sepal Length (cm):", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm):", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

    with col2:
        petal_length = st.number_input("Petal Length (cm):", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal Width (cm):", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    if st.button("Predict", type="primary"):
        # Prepare input
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        # FIX: inverse_transform now returns actual flower names because we trained on species_name
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        predicted_class_str = str(predicted_class)

        # Prediction probabilities
        probabilities = model.predict_proba(input_scaled)[0]

        # Display result
        st.success(f"### Predicted Species: **{predicted_class_str.upper()}** 🌸")

        # Show probabilities
        st.subheader("Prediction Probabilities")

        # Class names are now actual flower names
        class_names_list = [str(c) for c in label_encoder.classes_]

        prob_df = pd.DataFrame({
            'Species': class_names_list,
            'Probability': probabilities
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#440154', '#21918c', '#fde725']
        bars = ax.bar(prob_df['Species'], prob_df['Probability'], color=colors)
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')

        st.pyplot(fig)
        st.dataframe(prob_df)

        # Show input summary
        st.subheader("Input Summary")
        input_summary = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_summary)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit | Decision Tree Classifier")
