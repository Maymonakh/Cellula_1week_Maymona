# Toxic Text Classification

This repository contains my solutions for the RNN and LSTM text classification tasks completed during my internship at **Cellula Technologies**.

## Project Overview

The goal of this project is to classify text into different safety and toxicity categories using deep learning models.

The dataset contains:

* `query`
* `image descriptions`
* `Toxic Category`

The `query` and `image descriptions` were combined and used as the input text.

The dataset contains **9 categories**:

* Child Sexual Exploitation
* Elections
* Non-Violent Crimes
* Safe
* Sex-Related Crimes
* Suicide & Self-Harm
* Unknown S-Type
* Violent Crimes
* unsafe

## Models

### Task 1 — RNN

An RNN model was trained for toxic text classification.

**Target F1 Score:** 0.75

**Results:**

* Weighted F1 Score: **0.9225**
* Macro F1 Score: **0.9210**

### Task 2 — LSTM

An LSTM model was trained using the same preprocessing steps and dataset.

**Target F1 Score:** 0.85

**Results:**

* Weighted F1 Score: **0.9544**
* Macro F1 Score: **0.9515**

The LSTM achieved a higher F1 score than the RNN on this dataset.

## Preprocessing

The following steps were used for both models:

1. Combine the `query` and `image descriptions`.
2. Split the data into training and testing sets.
3. Tokenize the text.
4. Convert the text into sequences.
5. Pad the sequences to a fixed length of 50.
6. Encode the target categories.
7. Split the training data into training and validation sets.
8. Use class weights to handle class imbalance.

## Evaluation

The models were evaluated using:

* Weighted F1 Score
* Macro F1 Score
* Accuracy
* Classification Report
* Confusion Matrix

Training and validation curves were also included to show the model performance during training.

## Requirements

The project uses Python and the following main libraries:

* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* ReportLab

## Training

Both models use the same dataset and preprocessing pipeline. The main difference is the recurrent layer used in each model.

* RNN: `SimpleRNN`
* LSTM: `LSTM`
