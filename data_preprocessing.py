import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("Loading datasets...")

# load datasets
transaction = pd.read_csv("train_transaction.csv")
identity = pd.read_csv("train_identity.csv")

print("Transaction shape:", transaction.shape)
print("Identity shape:", identity.shape)

# merge datasets
df = transaction.merge(identity, how="left", on="TransactionID")

# take smaller sample for development
df = df.sample(n=50000, random_state=42)
print("Sampled dataset shape:", df.shape)

print("Merged shape:", df.shape)

# keep only useful columns (for now)
df = df.drop(columns=["TransactionID"])

# target label
labels = df["isFraud"]
df = df.drop("isFraud", axis=1)

# fill missing values (VERY IMPORTANT)
df = df.fillna(0)

# keep only numeric columns (GAN needs numbers)
df = df.select_dtypes(include=["number"])

print("Numeric features:", df.shape)

# scale data 0–1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

processed = pd.DataFrame(scaled_data)
processed["isFraud"] = labels.values

processed.to_csv("processed_data.csv", index=False)

print("Preprocessing complete! File saved as processed_data.csv")