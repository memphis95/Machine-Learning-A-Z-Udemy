# Machine Learning Algorithms

This repository contains the working example of the machine learning algorithms.

Most of the algorithms are from the Machine Learning A-Z course by Kirill Eremenko and Hadelin de Ponteves on Udemy.

## Contents

#### 1. Data Preprocessing
   It is the method of analyzing, filtering, transforming and encoding data so that a machine learning algorithm 
   can understand and work with the processed output.
   - Data Overview

   - Identify missing data

   - Identify outliers

   - Removing inconsistencies

#### 2. Regression

- Linear Regression

- Multiple Linear Regression

- Polynomial Regression

- Support Vector Regression

- Decision Tree Regression

- Random Forest Regression
    

#### 3. Classification

- Logistic Regression

- K-Nearest Neighbors(KNN)

- Support Vector Machine(SVM)

- Kernal SVM

- Naive Bayes

- Decision Tree Classifier

- Random Forest Classification

#### 4. Clustering

- K-Means Clustering

- Hierarchical Clustering

- DBSCAN Clustering [DBSCAN](https://github.com/memphis95/Machine-Learning-A-Z-Udemy/blob/main/4.%20Clustering/3.%20DBSCAN%20Clustering/clustering.ipynb)
    


#### 5. Association Rule Learning
- Apriori
- Eclat

#### 6. Ensemble Learning

#### 7. Dimensionality Reduction
- PCA
- LDA
- Kernal PCA

#### 8. Reinforcement Learning

#### 9. Natural Language Processing

#### 10. Deep Learning

#### 11. Model Selection & Boosting

Two main classes of techniques to approximate the ideal case of model selection

- Probability Measures : Choose a model via in-sample error and complexity.
Analytically scoring a candidate model using both its performance on the training dataset and the complexity of the model.

Note : Training error is optimistically biased, therefore not a good basis of choosing a model. The performance can be penalized based on how optimistic the training error is believed to be. It is achieved using algorithm-specific methods, often linear, that penalize the score based on the complexity of the model.

Probabilistic model selection measures
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)
- Minimum Description Length (MDL)
- Structural Risk Minimization (SRM)

- Resampling Methods : Choose a model via estimated out-of-sample error.
estimate the performance of a model( the model development process) on out-of-sample data.
It is achieved by splitting the training dataset into sub train and test sets, fitting a model on the sub train set, and evaluating it on test set. This process may then be repeated multiple times and the mean perofrmance across each trail is reported.

Resampling model selection methods 
- Random trail/test splits
- Cross-Validation(k-fold, LOOCV)
- Bootstrap