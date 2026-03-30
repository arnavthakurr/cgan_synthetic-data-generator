# Synthetic Financial Transaction Data Generator Using Conditional GAN

## Overview
This project generates synthetic financial transaction data with a Conditional GAN (CGAN) built in PyTorch. The goal is to support fraud detection experiments without exposing real user transaction records.

## Problem Statement
Fraud detection datasets are difficult to share because they are sensitive, highly imbalanced, and often restricted by privacy rules. This project explores whether a CGAN can learn fraud-related patterns well enough to produce useful synthetic samples for downstream modeling and analysis.

## Features
- Preprocesses the IEEE-CIS fraud detection dataset
- Trains a conditional GAN on fraud and non-fraud labels
- Saves generator and discriminator checkpoints locally
- Produces synthetic transaction samples for evaluation
- Visualizes real vs synthetic feature distributions

## Dataset
This project uses the IEEE-CIS Fraud Detection dataset from Kaggle:

https://www.kaggle.com/competitions/ieee-fraud-detection

Place these files in the project root before preprocessing:
- `train_transaction.csv`
- `train_identity.csv`

Large datasets and generated CSV outputs are excluded from GitHub.

## Project Structure
```text
cgan_synthetic-data-generator/
|-- data_preprocessing.py
|-- gan_model.py
|-- train.py
|-- evaluate.py
|-- fraud_classifier.py
|-- README.md
`-- .gitignore
```

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Seaborn

## Setup
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run preprocessing:

```bash
python data_preprocessing.py
```

5. Train the CGAN:

```bash
python train.py
```

6. Compare real and synthetic distributions:

```bash
python evaluate.py
```

## Outputs
- `processed_data.csv` after preprocessing
- `generator.pth` and `discriminator.pth` after training
- `synthetic_transactions.csv` after synthetic data generation

These outputs are generated locally and are ignored in Git for cleaner version control.

## Notes
- The raw Kaggle dataset files are required locally and are not included in this repository.
- Model checkpoints and generated CSV files stay on your machine and are excluded from Git tracking.

## Future Work
- Improve GAN stability and evaluation metrics
- Add quantitative quality checks for synthetic data
- Train fraud classifiers on real vs synthetic data
- Build a simple demo or dashboard for results

## Author
Arnav Singh Tomar
