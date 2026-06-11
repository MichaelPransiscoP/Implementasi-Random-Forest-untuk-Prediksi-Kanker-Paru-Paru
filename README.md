# Lung Cancer Prediction using Random Forest

## Overview

This project implements a Machine Learning model for predicting lung cancer risk levels using the Random Forest algorithm. The model utilizes clinical patient data and applies preprocessing, feature selection, hyperparameter optimization, and model evaluation to achieve high prediction performance.

This project was developed as an undergraduate thesis in Informatics.

---

## Features

- Data preprocessing
- Label Encoding & One-Hot Encoding
- Feature Selection using RFECV
- Hyperparameter Tuning using RandomizedSearchCV
- Random Forest Classification
- Cross Validation
- Feature Importance Analysis
- Model Performance Evaluation
- Visualization of Results

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Project Structure

```
project/
│
├── dataset/
│   └── cancer_patient_data.csv
├── visualization/
├── main.py
└── README.md
```

---

## Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Encoding
4. Train-Test Split
5. Feature Selection (RFECV)
6. Hyperparameter Optimization (RandomizedSearchCV)
7. Train Random Forest Model
8. Cross Validation
9. Model Evaluation
10. Feature Importance Visualization

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Cross Validation Score
- Confusion Matrix

---

## Installation

Clone repository:

```bash
git clone https://github.com/username/lung-cancer-random-forest.git
```

Enter project:

```bash
cd lung-cancer-random-forest
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---

## Requirements

Example:

```
python>=3.10

numpy
pandas
scikit-learn
matplotlib
seaborn
```

or

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## Dataset

The project uses a clinical lung cancer dataset containing patient information and risk-related features for classification.

> If the dataset is subject to licensing restrictions, please ensure compliance before redistribution.

---

## Results

The implementation includes:

- Random Forest Classifier
- RFECV Feature Selection
- Hyperparameter Optimization
- Feature Importance Ranking
- Cross Validation Analysis
- Performance Comparison

---

## Future Improvements

- Web deployment using Flask/FastAPI
- Explainable AI (SHAP/LIME)
- Comparison with XGBoost and LightGBM
- Integration with medical decision support systems

---

## Author

Michael Pransisco Purba

Bachelor Thesis  
Informatics
