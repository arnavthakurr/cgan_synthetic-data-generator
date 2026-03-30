import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from gan_model import Generator, Discriminator

print("Loading processed dataset...")
df = pd.read_csv("processed_data.csv")

labels = df["isFraud"].values
data = df.drop("isFraud", axis=1).values

print("Dataset loaded:", data.shape)

# convert to tensors
data_tensor = torch.tensor(data, dtype=torch.float32)
label_tensor = torch.tensor(labels.reshape(-1,1), dtype=torch.float32)

dataset = TensorDataset(data_tensor, label_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print("DataLoader ready!")

# initialize models
G = Generator()
D = Discriminator()

print("Generator and Discriminator initialized!")

# test one batch
for real_data, real_labels in loader:
    print("Batch shape:", real_data.shape)
    break

print("Pipeline setup successful!")