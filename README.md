# 🌸 Iris Flower Classification - Decision Tree

A complete machine learning project for classifying Iris flowers using Decision Tree Algorithm, built with Streamlit.

## 📁 Project Structure

```
iris_project/
├── iris.csv                  # Iris dataset (150 samples + 1 duplicate)
├── app.py                    # Streamlit application (FIXED)
├── preprocessing.py          # Data preprocessing script (FIXED)
├── visualization.py          # Data visualization script (FIXED)
├── model.py                  # Model training & evaluation script (FIXED)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Features

### Streamlit App Pages:
1. **Dataset** - View the Iris dataset preview, statistical summary, and class distribution
2. **Data Visualization** - Interactive visualizations:
   - Correlation Heatmap
   - Box Plots by Species
3. **Model Training** - Train Decision Tree with adjustable parameters:
   - **Max Depth slider: 1 to 5** (capped at 5)
   - View tree visualization and feature importance
4. **Evaluation Metrics** - Display:
   - Accuracy, Precision, Recall
   - Confusion Matrix (with flower names)
   - Classification Report
   - Note explaining Accuracy vs Prediction Confidence
5. **Prediction** - Input flower measurements and get:
   - **Species prediction as flower name** (setosa/versicolor/virginica)
   - Prediction probabilities bar chart

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Running the App

```bash
streamlit run app.py
```

## 📊 Dataset

The Iris dataset contains 150 samples with 4 features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Three species:
- Setosa
- Versicolor
- Virginica

**Note:** The CSV contains 1 duplicate row which is handled in preprocessing.

## 🤖 Model

- Algorithm: Decision Tree Classifier
- Preprocessing: StandardScaler for feature scaling, duplicate removal
- Target: Text labels (species_name) encoded to preserve flower names
- Split: 80% training, 20% testing (stratified)
- Max Depth: Capped at 5 to prevent overfitting
- Metrics: Accuracy, Precision, Recall, Confusion Matrix

## 📝 Individual Scripts

You can also run the standalone scripts:

```bash
python preprocessing.py    # Run preprocessing (removes duplicates, scales, saves artifacts)
python visualization.py    # Generate visualizations (removes duplicates first)
python model.py            # Train and evaluate model (removes duplicates, uses text labels)
```

## 🔧 Recent Fixes

1. **Predictions now show flower names** instead of numeric indices (0/1/2)
2. **Max depth slider capped at 5** (was 10)
3. **Duplicate row removed** during preprocessing
4. **Accuracy vs Confidence explained** in Evaluation page
5. **All scripts synchronized** to use `species_name` (text labels)
