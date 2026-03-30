import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
real_df = pd.read_csv("processed_data.csv")
synthetic_df = pd.read_csv("synthetic_transactions.csv")

# Drop label column
real_df = real_df.drop("isFraud", axis=1)
synthetic_df = synthetic_df.drop("isFraud", axis=1)

# Pick a few features to compare
features = real_df.columns[:5]

for feature in features:
    plt.figure(figsize=(6,4))
    sns.kdeplot(real_df[feature], label="Real", fill=True)
    sns.kdeplot(synthetic_df[feature], label="Synthetic", fill=True)
    plt.title(f"Feature Distribution: {feature}")
    plt.legend()
    plt.show()