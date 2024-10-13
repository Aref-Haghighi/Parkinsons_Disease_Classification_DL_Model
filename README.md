# Parkinson's Disease Classification with Deep Learning

This repository contains a deep learning model developed to classify Parkinson's disease using patient data from the UCI Machine Learning Repository. The model leverages techniques such as oversampling with SMOTE to handle class imbalance and SHAP explainability to interpret the results.

## Overview

The project demonstrates a complete machine learning pipeline, including data preprocessing, model building, training, evaluation, and explainability. The goal is to classify whether a patient has Parkinson's disease based on several voice and speech-related features.

### Key Features:
- **Data Preprocessing**: Scaled the dataset and handled class imbalance using SMOTE.
- **Model Architecture**: Sequential neural network with four hidden layers, batch normalization, dropout regularization, and L2 kernel regularization.
- **Class Weights**: Computed class weights to address class imbalance during model training.
- **Metrics**: Accuracy, AUC, Precision-Recall Curve, and SHAP summary plots to interpret feature importance.
- **Callbacks**: ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau to optimize training.

## Dataset

The dataset used in this project is from the UCI Machine Learning Repository:
- **Name**: Parkinson's Disease Dataset
- **Link**: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data)

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `scikit-learn`
- `imblearn`
- `shap`

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
