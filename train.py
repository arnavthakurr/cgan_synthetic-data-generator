import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from gan_model import Generator, Discriminator

# Hyperparameters
batch_size = 64
epochs = 50        # was 5
lr = 0.0001        # slower learning = more stable
latent_dim = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")
df = pd.read_csv("processed_data.csv")

labels = df["isFraud"].values
data = df.drop("isFraud", axis=1).values

data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
label_tensor = torch.tensor(labels.reshape(-1,1), dtype=torch.float32).to(device)

dataset = TensorDataset(data_tensor, label_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

data_dim = data.shape[1]

# Models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

print("Starting GAN training...")

for epoch in range(epochs):
    for real_data, real_labels in loader:

        batch_size_curr = real_data.size(0)

        real_targets = torch.ones(batch_size_curr,1).to(device)
        fake_targets = torch.zeros(batch_size_curr,1).to(device)

        # ----- Train Discriminator -----
        noise = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_data = G(noise, real_labels)

        D_real = D(real_data, real_labels)
        loss_real = criterion(D_real, real_targets)

        D_fake = D(fake_data.detach(), real_labels)
        loss_fake = criterion(D_fake, fake_targets)

        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ----- Train Generator -----
        noise = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_data = G(noise, real_labels)

        D_fake = D(fake_data, real_labels)
        loss_G = criterion(D_fake, real_targets)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {loss_D.item():.4f}  Loss G: {loss_G.item():.4f}")
    
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")
print("Models saved!")

print("Training finished!")

# ==============================
# Generate Synthetic Data
# ==============================

print("Generating synthetic transactions...")

num_samples = 5000

noise = torch.randn(num_samples, latent_dim).to(device)

# generate equal fraud / non-fraud labels
fake_labels = torch.randint(0,2,(num_samples,1)).float().to(device)

with torch.no_grad():
    synthetic_data = G(noise, fake_labels).cpu().numpy()

synthetic_df = pd.DataFrame(synthetic_data)
synthetic_df["isFraud"] = fake_labels.cpu().numpy()

synthetic_df.to_csv("synthetic_transactions.csv", index=False)

print("Synthetic dataset saved as synthetic_transactions.csv")