Gene Expression Analysis for Alzheimer’s Disease Biomarkers
Overview

This project investigates gene expression biomarkers associated with Alzheimer’s Disease (AD) using transcriptomic microarray data from the GSE33000 dataset. By applying statistical analysis, dimensionality reduction, clustering, and machine learning models, we explore molecular differences between Alzheimer’s patients and healthy controls and evaluate the feasibility of gene-expression–based classification.

Objectives

Identify differentially expressed genes associated with Alzheimer’s Disease

Visualize high-dimensional gene expression patterns

Explore clustering behavior of samples

Build and evaluate classification models to predict AD status

Dataset

Source: Gene Expression Omnibus (GEO)

Accession: GSE33000

Samples: 310 postmortem prefrontal cortex samples

Groups: Alzheimer’s Disease patients vs. healthy controls

Platform: Affymetrix Human Genome U133 Plus 2.0 Array

Data Type: Microarray gene expression intensity values

Technologies Used

Python

pandas, numpy

scikit-learn

matplotlib, seaborn

GEOparse

Methods
1. Data Preprocessing

Retrieved expression and metadata using GEOparse

Matched samples with diagnosis labels

Removed non-numeric entries and duplicate genes

Applied:

Log2 transformation

Z-score normalization

2. Differential Expression Analysis

Performed two-sample t-tests comparing AD vs. control samples

Ranked genes by p-value

Identified top candidate biomarkers

3. Dimensionality Reduction & Visualization

PCA (Principal Component Analysis)

Revealed trends indicating separation between AD and control samples

t-SNE (t-distributed Stochastic Neighbor Embedding)

Improved visualization of local clustering structure

4. Clustering

Applied K-means clustering (k = 2) to normalized expression data

Clusters showed partial alignment with clinical diagnosis

5. Classification Models

Logistic Regression

Baseline linear classification model

Random Forest

Achieved better performance

Captured non-linear gene interactions

Provided feature importance rankings for genes

Key Findings

Several genes showed significant differential expression between AD and controls

PCA and t-SNE visualizations demonstrated meaningful separation trends

Clustering partially reflected disease status

Random Forest outperformed Logistic Regression, highlighting the value of non-linear models for transcriptomic data

How to Run

Clone this repository

Install required dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn GEOparse


Open the Jupyter Notebook:

jupyter notebook Gene-Expression-Analysis-for-Alzheimer-s-Biomarker.ipynb

Limitations & Future Work

Microarray data limits transcript resolution compared to RNA-seq

Future extensions may include:

Pathway enrichment analysis

Validation using independent datasets

RNA-seq–based differential expression

Deep learning models for classification
