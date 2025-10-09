import wandb
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from dataset import BreakHisDataset
from torchvision import transforms, models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Read CSV
data = pd.read_csv('breakhis_data/Folds.csv')

# Clean 'mag' column so it's always int (e.g., "100X" → 100)
data['mag'] = data['mag'].astype(str).str.replace('X', '', regex=False).str.strip().astype(int)

# Dataset directory
root_dir = 'breakhis_data/BreaKHis_v1/'

# Classes
class_names = ['benign', 'malignant']

# Number of workers for parallel data loading
NUM_WORKERS = os.cpu_count()

from sklearn.model_selection import train_test_split
# Data Transformations
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# Split data into training, validation, and test sets

# Only training rows
train_df = data[data['grp'] == "train"]


# Test set from CSV
test_df = data[data['grp'] == "test"]


def train_one_mag(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Dataset for a single magnification
        train_dataset_mag = BreakHisDataset(
            data=train_df,
            root_dir=root_dir, 
            mag=config.mag,
            fold = [1,2,3,4],
            transform= train_transform
        )
        
        val_dataset_mag = BreakHisDataset(
            data=train_df, 
            root_dir=root_dir, 
            mag=config.mag,
            fold = 5,
            transform= val_test_transform
        )
        

        train_loader = DataLoader(
            train_dataset_mag, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=NUM_WORKERS
        )

        val_loader = DataLoader(
            val_dataset_mag, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        # Building model dynamically
        if config.model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif config.model_name == "resnet34":
            model = models.resnet34(pretrained=True)
        elif config.model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        else:
            raise ValueError("Unknown model!")

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        model = model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Basic Training loop
        best_acc = 0
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                running_corrects += torch.sum(preds == labels.data)
            
            scheduler.step()
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            # Validation
            model.eval()
            val_corrects = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float()
                    val_corrects += torch.sum(preds == labels.data)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = val_corrects.double() / len(val_loader.dataset)
            val_precision = precision_score(all_labels, all_preds)
            val_recall = recall_score(all_labels, all_preds)

            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall
            })

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"best_{config.model_name}_{config.mag}.pth")

#  Sweep config
sweep_config = {
    "method": "grid",   # grid search across magnifications + models
    "parameters": {
        "mag": {"values": ["40X", "100X", "200X", "400X"]},
        "model_name": {"values": ["resnet18", "resnet34", "resnet50"]},
        "lr": {"values": [1e-2, 1e-3]},
        "batch_size": {"values": [32, 64]},
        "epochs": {"value": 12}
    }
}

#  Create sweep
sweep_id = wandb.sweep(sweep_config, project="breakhis-magnification")
wandb.agent(sweep_id, function=train_one_mag)
