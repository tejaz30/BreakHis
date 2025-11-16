import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np
import random
import os

from models.conv_vae import ConvVAE, vae_loss
from dataloader import BreakHisDataset  # same as before

 
# Reproducibility
 
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

 
# Configuration
 
class Config:
    root_dir = "/home/teja-bulusu/Desktop/breakhis_start/dataset/BreaKHis_v1/"
    batch_size = 64
    lr = 1e-4
    epochs = 50
    mag = 100
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

wandb.init(project="breakhis-vae-optimized", config=vars(config))
print("WandB logging started...")

 
# Data loading
 
data = pd.read_csv('dataset/Folds.csv')
data['mag'] = data['mag'].astype(str).str.replace('X', '', regex=False).astype(int)

train_df = data[data['grp'] == "train"]
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['filename'].apply(lambda x: x.split('/')[3]),
    random_state=42
)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = BreakHisDataset(train_df, root_dir=config.root_dir, mag=config.mag, transform=transform)
val_dataset = BreakHisDataset(val_df, root_dir=config.root_dir, mag=config.mag, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

 
# Model setup
 
device = torch.device(config.device)
vae = ConvVAE(latent_dim=512).to(device)
optimizer = torch.optim.AdamW(vae.parameters(), lr=config.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

 
# Training loop
 
train_losses, val_losses = [], []

for epoch in range(1, config.epochs + 1):
    vae.train()
    total_loss = 0
    beta = min(0.0005, (epoch / 50) * 0.0005)  # KL warmup

    for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recons, mu, logvar = vae(imgs)
        loss = vae_loss(recons, imgs, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            recons, mu, logvar = vae(imgs)
            loss = vae_loss(recons, imgs, mu, logvar, beta)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "beta": beta, "epoch": epoch})
    print(f"Epoch [{epoch}/{config.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | β={beta:.6f}")

 
# Save model
 
os.makedirs("checkpoints", exist_ok=True)
torch.save(vae.state_dict(), "checkpoints/vae_breakhis_best.pth")
wandb.save("checkpoints/vae_breakhis_best.pth")

 
# Reconstruction visualization
 
vae.eval()
imgs, _ = next(iter(val_loader))
imgs = imgs.to(device)

with torch.no_grad():
    recons, _, _ = vae(imgs)

n = 8
fig, axes = plt.subplots(2, n, figsize=(15, 4))
for i in range(n):
    axes[0, i].imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
    axes[0, i].axis("off")
    axes[1, i].imshow(recons[i].permute(1, 2, 0).cpu().numpy())
    axes[1, i].axis("off")

plt.suptitle("Original (top) vs Reconstructed (bottom)")
plt.show()
