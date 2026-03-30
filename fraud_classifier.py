import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading real dataset...")
real_df = pd.read_csv("processed_data.csv")

print("Loading synthetic dataset...")
synthetic_df = pd.read_csv("synthetic_transactions.csv")

# Combine datasets
combined_df = pd.concat([real_df, synthetic_df])

# Split features and labels
X = combined_df.drop("isFraud", axis=1)
y = combined_df["isFraud"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training fraud classifier...")
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))