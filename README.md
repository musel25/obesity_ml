# Obesity Classification — End-to-End ML Pipeline

A compact, end-to-end classification pipeline on the UCI Obesity dataset. It covers data preprocessing, feature selection via mutual information, dimensionality reduction (PCA, t‑SNE for visualization), and a broad model comparison across 31 classifiers. The top model is XGBoost with ~95.5% accuracy on the held-out data.

## Overview
- Problem: Multiclass classification of obesity levels from lifestyle and biometric features.
- Dataset: UCI “Obesity Levels Based on Eating Habits and Physical Condition”.
- Methods: Mutual information feature selection; PCA and t‑SNE for structure/visualization; extensive model sweep (31 models).
- Result: Best model — XGBoost (Accuracy: 0.955, F1: 0.956).

## Dataset
- Source: UCI Machine Learning Repository — Obesity Levels Based on Eating Habits and Physical Condition
  - https://archive.ics.uci.edu/dataset/544/obesity+levels+based+on+eating+habits+and+physical+condition
- Included here as `ObesityDataSet_raw.csv` for convenience.

## Methods
- Preprocessing: Basic cleaning/encoding; train/validation/test split.
- Feature selection: Mutual information to rank informative features.
- Dimensionality reduction: 
  - PCA for linear structure and variance explanation.
  - t‑SNE for nonlinear structure and cluster visualization.
- Modeling: 31 classifiers spanning tree-based methods, linear models, SVMs, kNN, ensembles, neural networks, and boosting.
- Evaluation: Accuracy, F1, Precision, Recall; comparison logged to `model_performance_comparison.csv`.

## Results (top slice)
From `model_performance_comparison.csv`:

- XGBoost (Base): Accuracy 0.9551, F1 0.9556
- Neural Network (Config 1): Accuracy 0.9528, F1 0.9525
- Gradient Boosting (Base): Accuracy 0.9527, F1 0.9533
- Random Forest (Base): Accuracy 0.9504, F1 0.9509
- Neural Network (Config 2): Accuracy 0.9481, F1 0.9483

Full comparisons are in the CSV.

## Repository Contents
- `DAPRFinalProj.ipynb` — main notebook with the full pipeline.
- `ObesityDataSet_raw.csv` — raw dataset file.
- `model_performance_comparison.csv` — summary of model metrics.

## Getting Started
1. Create a Python environment (3.9+ recommended).
2. Install packages, for example:
   ```bash
   pip install numpy pandas scikit-learn xgboost matplotlib seaborn
   ```
3. Open and run `DAPRFinalProj.ipynb` in Jupyter/VS Code, executing cells in order.

## Notes
- PCA and t‑SNE are used primarily for exploration/visualization, not as inputs to the best-performing classifier.
- For reproducibility, ensure consistent random seeds across preprocessing and model training.

## Link
- GitHub: https://github.com/musel25/obesity_ml

