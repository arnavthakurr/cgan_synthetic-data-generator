# Synthetic Financial Transaction Data Generator using Conditional GAN

## Overview
This project aims to generate realistic synthetic financial transaction data using a Conditional Generative Adversarial Network (CGAN).  
The goal is to address data privacy and data scarcity issues in fraud detection systems.

Synthetic data allows machine learning models to be trained without exposing real user financial information.

---

## Problem Statement
Fraud detection systems require large volumes of transaction data.  
However, real financial datasets are:
- Highly sensitive
- Difficult to share due to privacy laws
- Imbalanced (very few fraud cases)

This project generates synthetic transaction data that mimics real-world fraud patterns.

---

## Project Objectives
- Preprocess real-world fraud dataset (IEEE-CIS Fraud Detection)
- Build a Conditional GAN using PyTorch
- Generate synthetic fraud and non-fraud transactions
- Prepare pipeline for training and evaluation

---

## Current Project Status
Phase 1: Data preprocessing completed  
Phase 2: GAN architecture and training pipeline setup completed  
Next Phase: Model training and evaluation

---

## Dataset
We use the **IEEE-CIS Fraud Detection Dataset** from Kaggle.

Due to GitHub file size limits and privacy considerations, datasets are not included in this repository.

Download dataset from:
https://www.kaggle.com/competitions/ieee-fraud-detection

Required files:
- train_transaction.csv
- train_identity.csv

---

## Project Structure
