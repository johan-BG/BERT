# Spam Mail Detection with DistilBERT

This repository contains a project that leverages the [DistilBERT](https://huggingface.co/distilbert-base-uncased) model to detect spam emails. By fine-tuning a pre-trained DistilBERT model on a labeled dataset of spam and ham emails, the system can classify incoming emails as either spam or not.

## Overview

Spam detection is a crucial task in email filtering. This project uses state-of-the-art Natural Language Processing (NLP) techniques to analyze email text and accurately predict whether it is spam. The project is built using the [Hugging Face Transformers](https://github.com/huggingface/transformers) library along with either PyTorch as the backend.

## Features

- **Pre-trained Model:** Utilizes the lightweight and fast `distilbert-base-uncased` model.
- **Fine-tuning:** Custom fine-tuning on a spam vs. ham email dataset.
- **Evaluation Metrics:** Accuracy, and confusion metric for performance evaluation.
- **Easy Integration:** Ready-to-use scripts for training and prediction.

## Project Structure

### How It Works
1. **Data Preparation:** The project starts with data preprocessing of a dataset containing spam and ham emails. Text data is tokenized and encoded using the DistilBERT tokenizer.
2. **Model Fine-tuning:** The pre-trained DistilBERT model is fine-tuned on the spam detection task using a sequence classification approach. Hyperparameters such as batch size, learning rate, and number of epochs are tuned for optimal performance.
3. **Evaluation:** The model’s performance is evaluated using metrics like accuracy, precision, recall, and F1-score to ensure high reliability in distinguishing spam emails.
4. **Deployment:** Once trained and evaluated, the model is deployed for making predictions on new email data. The project includes scripts for both training and inference, making it easy to integrate into larger systems.

### Additional Resources
- **Trained Model:** [Access the fine-tuned model on Google Drive](https://drive.google.com/drive/folders/1Yv9Lv8wTmnumtcYLHOgk93ht_lWAUj3Q?usp=sharing)
- **Training Results & Logs:** [Access training logs and evaluation results on Google Drive](https://drive.google.com/drive/folders/1Uo-CrfPyvTq17-tPnM5Rjym0Ln_r5P1c?usp=sharing)


This project demonstrates the application of state-of-the-art NLP techniques in solving a real-world problem—filtering unwanted spam emails—while maintaining efficiency and scalability.
