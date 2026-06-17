# Iris Species Data Analysis and Regression Modeling

This repository provides an end-to-end data processing, exploratory analysis, and predictive modeling pipeline for the classic Iris dataset. The project demonstrates the application of robust data preprocessing, visualization, and advanced machine learning models (K-Nearest Neighbors and Support Vector Regression) optimized via hyperparameter tuning.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Installation & Requirements](#installation--requirements)
- [Project Pipeline](#project-pipeline)
- [Models Evaluated](#models-evaluated)
- [Results & Performance](#results--performance)

## 🔍 Project Overview
The primary goal of this project is to model relationship dynamics between botanical features of the Iris flower. By taking structural measurements (sepal and petal dimensions) and categorical species labels, the pipeline transitions the standard classification problem into a continuous predictive regression task, optimized using multi-criteria tuning.

## 📊 Dataset
The project utilizes the **Iris Dataset** (`IRIS.csv`), containing 150 instances uniformly distributed across three distinct species:
- `Iris-setosa` (Encoded as `0`)
- `Iris-versicolor` (Encoded as `1`)
- `Iris-virginica` (Encoded as `2`)

Each instance includes 4 numeric features: Sepal Length, Sepal Width, Petal Length, and Petal Width.

## 🚀 Key Features
- **Exploratory Data Analysis (EDA):** Advanced statistical profiling and feature interaction visualization using pairwise plots (`seaborn.pairplot`).
- **Feature Engineering:** Automated categorical target encoding via `LabelEncoder` and data scaling via `StandardScaler`.
- **Pipeline Architecture:** Scikit-Learn `Pipeline` integration to eliminate data leakage during cross-validation.
- **Hyperparameter Optimization:** Grid Search Cross-Validation (`GridSearchCV`) for algorithmic fine-tuning.

## 🛠️ Installation & Requirements
To run this notebook, ensure you have the following packages installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
