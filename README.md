# Intro to Machine Learning: Exploration, Regression, and Classification Pipelines

A comprehensive suite of foundational and advanced machine learning workflows demonstrating data preprocessing architectures, feature engineering, and hyperparameter optimization. The repository encompasses classic regression and classification algorithms implemented via modular Scikit-Learn pipelines as well as custom scratch optimization frameworks to prevent data leakage and maximize validation fidelity.

## 📋 Table of Contents
- [Project Objectives](#project-objectives)
- [Repository Structure & Datasets](#repository-structure--datasets)
- [Core Pipeline Implementations](#core-pipeline-implementations)
  - [1. Iris Species Analysis (KNN & SVR Continuous Regression)](#1-iris-species-analysis-knn--svr-continuous-regression)
  - [2. Boston Housing Price Prediction (Scratch GD & Regularized Linear Regression)](#2-boston-housing-price-prediction-scratch-gd--regularized-linear-regression)
  - [3. Consumer Purchase Prediction (Binary Logistic Regression Framework)](#3-consumer-purchase-prediction-binary-logistic-regression-framework)
- [Installation & Dependencies](#installation--dependencies)
- [Key Methodology Highlights](#key-methodology-highlights)

---

## 🎯 Project Objectives
* **Modular Pipeline Architecture:** Construct end-to-end data processing workflows integrating automated transformers, avoiding analytical biases or structural data tracking leaks.
* **Algorithmic Evaluation:** Compare the behavior and accuracy of neighborhood-based, kernelized, linear, and probabilistic models across continuous and discrete target tasks.
* **Scratch Optimization:** Design and implement Batch and Stochastic Gradient Descent optimization algorithms from scratch to understand loss convergence dynamics.
* **Hyperparameter Control:** Leverage cross-validated Grid Searches (`GridSearchCV`) to fine-tune structural penalties, distance benchmarks, and mathematical regularization weights.

---

## 📊 Repository Structure & Datasets

The repository processes three fundamental data domains to evaluate multi-model robustness:
1. **Iris Dataset (`IRIS.csv`):** 150 entries documenting botanical metrics (sepal/petal dimensions) spanning three distinct categories mapped via numerical encoders.
2. **Boston Housing Dataset:** Diverse neighborhood socioeconomic, structural, and environmental parameters mapping out residential asset pricing patterns.
3. **Consumer Purchase Profile Data:** User financial brackets, transaction interaction footprints, and age tracking criteria used to gauge market response.

---

## 📈 Core Pipeline Implementations

### 1. Iris Species Analysis (KNN & SVR Continuous Regression)
This pipeline maps categorical species markers alongside baseline measurements to model continuous morphological relationships across botanical sub-types.
* **Feature Processing:** Automated string mapping using `LabelEncoder` to cast qualitative fields into numerical coordinates, matched with a structured `StandardScaler` workflow.
* **Algorithms Implemented:** K-Nearest Neighbors Regressor (KNNR) and Support Vector Regressor (SVR).
* **Grid Search Optimization Tuning:**
  * *KNN Regressor:* Systematic testing of neighbor counts (`n_neighbors`), spatial metric frameworks, and inverse-distance weighting configurations.
  * *Support Vector Regressor:* Optimization of penalty thresholds (`C`), gamma distributions, and diverse geometric boundary functions (Linear, RBF, Polynomial).

### 2. Boston Housing Price Prediction (Scratch GD & Regularized Linear Regression)
Addresses continuous valuation modeling of suburban median real estate properties (`MEDV`) while handling high-variance collinear inputs.
* **Methodology & Custom Implementations:**
  * Custom scratch implementations of **Batch Gradient Descent (BGD)** and **Stochastic Gradient Descent (SGD)**.
  * Structural feature engineering analyzing and stripping out highly correlated features to combat multicollinearity.
  * Inbuilt library benchmark evaluation utilizing Scikit-Learn's `LinearRegression` and L2-regularized `Ridge` architectures.
* **Key Observations:** The custom Batch Gradient Descent framework consistently outpaced SGD updates due to highly stable gradient trajectory vectors. Dropping redundant collinear factors directly enhanced overall model transparency and interpretability.

### 3. Consumer Purchase Prediction (Binary Logistic Regression Framework)
Implements a categorical tracking classification matrix to map buyer profiles to transaction completion events (`Purchased`: 0 or 1).
* **Feature Processing:** Integrated feature transformation processing categorical string values into one-hot binary maps alongside scaled standard variations for high-variance dimensions (`Age` and `EstimatedSalary`).
* **Algorithm Implemented:** Logistic Regression optimized via Stratified K-Fold cross-validation splits.
* **Grid Search Optimization Tuning:** Comprehensive balancing of inverse regularization boundaries (`C`) mapped against diverse optimization solver engines (`lbfgs`, `liblinear`).

---

## 🛠️ Installation & Dependencies

To execute these notebooks locally, set up your Python runtime environment with the following data science stack packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
