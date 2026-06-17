# Multi-Model Machine Learning Pipeline: Exploration, Regression, and Classification

This repository contains an end-to-end data processing, exploratory analysis, and predictive modeling pipeline executed across three distinct foundational datasets. The project showcases robust preprocessing architectures, feature engineering, and hyperparameter optimization utilizing Scikit-Learn pipelines to prevent data leakage and maximize model generalization.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure & Datasets](#repository-structure--datasets)
- [Core Pipelines](#core-pipelines)
  - [1. Iris Species Data Analysis (Advanced Regression)](#1-iris-species-data-analysis-advanced-regression)
  - [2. Boston Housing Value Prediction (Regularized Linear Regression)](#2-boston-housing-value-prediction-regularized-linear-regression)
  - [3. Consumer Purchase Behavior (Binary Logistic Regression)](#3-consumer-purchase-behavior-binary-logistic-regression)
- [Installation & Requirements](#installation--requirements)
- [Key Features & Methodologies](#key-features--methodologies)

---

## 🔍 Project Overview
The objective of this project is to implement, optimize, and compare multiple machine learning algorithms across a spectrum of predictive tasks (continuous estimation and categorical classification). By employing systematic hyperparameter tuning via grid search cross-validation, the workflows isolate optimal configurations for each distinct problem space.

---

## 📊 Repository Structure & Datasets

The pipeline operates on three primary data configurations:

1. **Iris Dataset (`IRIS.csv`):** 150 instances featuring botanical dimensions (sepal/petal lengths and widths) across three species (*Setosa*, *Versicolor*, *Virginica*).
2. **Boston Housing Dataset:** Environmental, structural, and socioeconomic metrics across Boston suburbs used to estimate median home values.
3. **Purchase History Dataset:** Consumer demographic and financial profiles used to determine transaction likelihood.

---

## 📈 Core Pipelines

### 1. Iris Species Data Analysis (Advanced Regression)
Transitions a classic classification challenge into a continuous relationship modeling task by predicting structural dimensions while incorporating categorical species encoders.
* **Target Management:** Automates target tracking via `LabelEncoder` to convert species into numerical identifiers (0, 1, and 2).
* **Models Evaluated:** K-Nearest Neighbors Regressor (KNNR) and Support Vector Regressor (SVR).
* **Grid Search Parameters:** 
  * *KNN:* `n_neighbors`, distance metrics, and weighting schemas.
  * *SVR:* Regularization scale (`C`), gamma parameters, and kernel functions (Linear, RBF, Poly).

### 2. Boston Housing Value Prediction (Regularized Linear Regression)
Addresses continuous estimation of the median value of owner-occupied homes (`MEDV`) while actively managing feature multicollinearity.
* **Feature Processing:** Standard scaling of continuous variables to guarantee uniform penalty distributions across regularized loss functions.
* **Models Implemented:** Ridge Regression ($L_2$ regularization) and Lasso Regression ($L_1$ feature selection).
* **Grid Search Parameters:** Systematic variation of the regularization penalty coefficient (`alpha`).

### 3. Consumer Purchase Behavior (Binary Logistic Regression)
Implements a binary classification framework to determine whether a user will execute a purchase (`Purchased`: 0 or 1) based on profile attributes.
* **Feature Processing:** One-hot encoding for demographic classifications alongside `StandardScaler` transformations for high-variance numeric fields like `Age` and `EstimatedSalary`.
* **Model Implemented:** Logistic Regression.
* **Grid Search Parameters:** Inverse regularization strength (`C`) combined with diverse optimization solver architectures (`lbfgs`, `liblinear`).

---

## 🛠️ Installation & Requirements

Ensure you have a Python environment configured with the required data science dependencies. Install them via:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
