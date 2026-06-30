# 🛒🛍️E-Commerce Product Classification — NLP Text Quality Study

This repository contains the practical component of a study on how text data quality and preprocessing techniques affect the performance of machine learning models for automatic e-commerce product classification.

## Dataset

**Ecommerce Text Classification** ([Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification))
50,425 product descriptions across 4 categories, loaded with manually assigned column names (`category`, `description`).

## What This Notebook Covers

**1. Initial Data Quality Assessment**
Loading and inspecting the raw dataset — checking for missing values, duplicate rows, and duplicate descriptions. The analysis found 22,622 duplicated product descriptions, which is flagged as a significant data quality issue since identical entries appearing in both train and test sets can lead to inflated model scores.

**2. Class Distribution Analysis**
Visualizing the distribution of products across categories before and after duplicate removal, to assess class balance and its potential impact on model training.

**3. Text Quality Analysis**
Examining description length, vocabulary size, lexical diversity, noise, and other textual properties to understand the characteristics of the raw input data.

**4. Preprocessing Strategies**
Comparing multiple text representation approaches:
- Bag of Words (BoW)
- TF-IDF
- Word embeddings
- BERT-like transformer-based models

## Goal

The central question is: *how much does text preprocessing quality actually matter for classification performance?* Each strategy is evaluated so the impact of data quality decisions can be measured directly against model outcomes.

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
transformers
```
