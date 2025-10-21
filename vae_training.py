import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
from dataloader import BreakHisDataset
from models.conv_vae import ConvVAE, vae_loss
from sklearn.model_selection import train_test_split
import wandb
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#Data loadng
data = pd.read_csv('dataset/Folds.csv')

# Convert magnification (e.g., "100X" → 100)
data['mag'] = data['mag'].astype(str).str.replace('X', '', regex=False).str.strip().astype(int)

#Config Definition
class Config:
    root_dir = "/home/teja-bulusu/Desktop/breakhis_start/dataset/BreaKHis_v1/"
    batch_size = 64
    lr = 1e-4
    epochs = 50 
    mag = 100  # <-- set magnification (e.g., 40, 100, 200, 400)
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# W&B initialization
wandb.init(project="breakhis-vae", config=vars(config))
print("WandB logging started...")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# train/val split
train_data_full = data[data['grp'] == "train"]

train_df, val_df = train_test_split(
    train_data_full,
    test_size=0.2,
    stratify=train_data_full['filename'].apply(lambda x: x.split('/')[3]),
    random_state=42
)

# Data loaders for specific magnification
train_dataset_mag = BreakHisDataset(
    data=train_df,
    root_dir=config.root_dir,
    mag=config.mag,
    transform=transform
)

val_dataset_mag = BreakHisDataset(
    data=val_df,
    root_dir=config.root_dir,
    mag=config.mag,
    transform=transform
)

train_loader = DataLoader(
    train_dataset_mag,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)

val_loader = DataLoader(
    val_dataset_mag,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)

#training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = ConvVAE(latent_dim=256).to(device)
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

num_epochs = 50
train_losses = []

for epoch in range(1, num_epochs + 1):
    vae.train()
    total_loss = 0
    for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recons, mu, logvar = vae(imgs)
        loss = vae_loss(recons, imgs, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})
    print(f"Epoch [{epoch}/{num_epochs}] - Loss: {avg_loss:.4f}")

    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            recons, mu, logvar = vae(imgs)
            loss = vae_loss(recons, imgs, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    wandb.log({"val_loss": val_loss, "epoch": epoch})



# Save model
torch.save(vae.state_dict(), "cvae_breakhis.pth")
wandb.save("cvae_breakhis.pth")


#validation evaluation
vae.eval()
imgs, _ = next(iter(val_loader))
imgs = imgs.to(device)
with torch.no_grad():
    recons, _, _ = vae(imgs)

def denorm(img_tensor):
    img_tensor = img_tensor * 0.5 + 0.5
    return img_tensor.clamp(0, 1)

import matplotlib.pyplot as plt
n = 8
fig, axes = plt.subplots(2, n, figsize=(15, 4))
for i in range(n):
    axes[0, i].imshow(denorm(imgs[i]).permute(1, 2, 0).cpu().numpy())
    axes[0, i].axis("off")
    axes[1, i].imshow(denorm(recons[i]).permute(1, 2, 0).cpu().numpy())
    axes[1, i].axis("off")
plt.suptitle("Original (top) vs Reconstructed (bottom)")
plt.show()
