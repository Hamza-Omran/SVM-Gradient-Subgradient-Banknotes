# Banknote Authentication Using SVM (Archived)

A convex optimization project implementing Support Vector Machine (SVM) classifiers using Gradient Descent and Subgradient Descent for banknote authentication.  
This repository is preserved only for historical academic reference and does not represent current professional coding standards.

## Team Members

- Hamza Omran – ID: 22011501  
- Mohammed Rafat  
- Wael Ahmed Rabie Al_naqiti – ID: 22010290  
- Moaz Mostafa Abd-Elhamied – ID: 22010263  
- Aly Eldin Yasser  
- Yasser Ashraf Mohammed – ID: 22010409  

## Contributions

This project was completed collaboratively as a team.  
My contributions include:

- Leading the design and implementation of the SVM optimization approach  
- Fully implementing **Subgradient Descent (SubGD)** and its mini-batch extensions  
- Implementing and debugging the **Gradient Descent SVM**  
- Leading preprocessing, standardization, and duplicate cleaning  
- Creating all performance comparisons, metrics tables, and convergence analysis  
- Preparing the notebook structure and visualizations  
- Supporting dataset cleaning and evaluation pipeline  

## Project Overview

This project builds binary classifiers to distinguish genuine banknotes from forged ones using:

1. **Gradient Descent SVM (GD-SVM)** with smoothed hinge loss  
2. **Subgradient Descent SVM (SubGD-SVM)** with mini-batch optimization  

A logistic regression model is included as a baseline comparison.

## Dataset

**Source:** UCI Machine Learning Repository – Banknote Authentication Dataset  
Link: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

### Characteristics (after processing)

- Instances: 1372  
- Features: 4 image-derived statistical features  
- Target: Binary (0 = forged, 1 = genuine)  
- Missing values: None  
- Duplicates: Removed  

### Features

- **variance** – texture contrast  
- **skewness** – asymmetry of pixel intensity distribution  
- **curtosis** – tailedness of distribution  
- **entropy** – randomness or complexity  

## Methodology

### Data Preprocessing

- Removed duplicates  
- Outlier inspection using Z-scores (kept all data)  
- Standardized features using StandardScaler and RobustScaler  
- 80/20 train-test split with stratification  

### Exploratory Data Analysis

- Histograms, boxplots, pairplots  
- Correlation heatmap  
- Class distribution analysis  
- Identified meaningful outliers and class separability patterns  

## Models Implemented

### Logistic Regression (Baseline)
- High accuracy (~98%)  
- Used for performance comparison  

### SVM with Gradient Descent (GD)

- Smoothed hinge loss  
- Learning rate = 0.001  
- Regularization λ = 0.01  
- 1000 iterations  
- Train/Test Accuracy ≈ 95%  
- Smooth convergence but slower  

### SVM with Subgradient Descent (SubGD)

- Standard hinge loss  
- Mini-batch updates  
- Tuned learning rates and batch sizes  
- Regularization λ = 0.0005  
- Train/Validation Accuracy ≈ 98–100%  
- Fast convergence  
- Initially biased due to slight class imbalance  

## Hyperparameter Tuning

Grid search over:

- Learning rates: 0.0005–0.05  
- Batch sizes: 8–64  
- Metrics: validation accuracy, training loss  

Selected best-performing configuration using convergence and accuracy comparison plots.

## Results Summary

| Model | Train Accuracy | Test/Val Accuracy | Convergence |
|-------|----------------|-------------------|-------------|
| Logistic Regression | ~98% | ~98% | Fast |
| GD-SVM | ~95% | ~95% | Slow, stable |
| SubGD-SVM | ~98–100% | ~98–100% | Fast |

### Key Observations

- SubGD converges significantly faster than GD  
- GD produces smoother training behavior  
- SubGD initially misclassified genuine notes due to class imbalance  
- Both SVM variants avoid overfitting with proper regularization  

## Visualizations

The notebook includes:

- Histograms and boxplots  
- Correlation heatmap  
- Pairplots with class separation  
- Training/validation accuracy curves  
- Loss function curves  
- Confusion matrices  

## Dependencies

numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
ucimlrepo

perl
Copy code

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy ucimlrepo
Usage
Open the notebook

Run cells sequentially to:

Load and preprocess the data

Perform EDA

Train baseline models

Train GD-SVM and SubGD-SVM

Compare performance and convergence

Key Findings
Both SVM implementations achieve >95% accuracy

SubGD converges faster than GD

GD produces more stable updates

Mini-batches significantly enhance SubGD performance

Standardization is essential for stable convergence

Outliers contribute meaningful class-separating information

Conclusions
When to Use Each Model
SubGD-SVM: faster training, scalable to larger datasets

GD-SVM: smoother convergence, more balanced predictions

Optimization Insights
Regularization strongly affects boundary placement

Learning rate and batch size critically influence convergence

Validation monitoring prevents overfitting

License

Academic use only.
This project is archived and no longer maintained.

References

UCI Banknote Authentication Dataset

Vapnik: Support Vector Machines

Boyd & Vandenberghe: Convex Optimization
