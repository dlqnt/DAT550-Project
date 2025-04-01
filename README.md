# DAT550-Project

Install necessary Python libraries (sklearn, pandas, numpy, nltk, tensorflow/pytorch)


# Data Acquisition and Understanding

Download the BuzzFeed News dataset from GitHub: https://github.com/BuzzFeedNews/2017-08-partisan-sites-and-facebook-pages
Understand the dataset structure: partisan labels (left/right), Facebook page data, engagement metrics
Consider using resources from the SemEval-2019 hyperpartisan news detection task

# Phase 1: Baseline Models (Weeks 1-2)

## Data Preprocessing

Clean text (remove HTML tags, normalize, handle missing values)
Create train/validation/test splits
Explore basic feature extraction (TF-IDF, word counts)


## Initial Models

Implement baseline models (Naive Bayes, Decision Trees)
Evaluate with basic metrics (accuracy, precision, recall, F1)



# Phase 2: Advanced Models (Weeks 3-4)

## Advanced Feature Engineering

Word embeddings (Word2Vec, GloVe, or BERT)
Stylometric features (readability scores, sentence structure)
Sentiment and emotional tone analysis
Explore article metadata and link patterns


## Deep Learning Models

Implement CNN and/or LSTM architectures
Consider pre-trained models like BERT or ELMo
Hyperparameter tuning

## Ensemble Methods

Combine multiple models for improved performance

# Suggested Model Architecture
Based on the SemEval-2019 results:

ELMo + CNN approach (highest accuracy in SemEval)

Use ELMo for contextual word embeddings
Feed embeddings to a CNN architecture


Ensemble of features approach

Combine linguistic features with embeddings
Use a linear classifier (SVM or logistic regression)
