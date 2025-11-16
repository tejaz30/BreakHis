import wandb
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from dataloader import BreakHisDataset
from torchvision import transforms, models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Read CSV
data = pd.read_csv('dataset/Folds.csv')

# Converting the mag column in the csv file (e.g., "100X" → 100)
data['mag'] = data['mag'].astype(str).str.replace('X', '', regex=False).str.strip().astype(int)

# Dataset directory
root_dir = 'dataset/BreaKHis_v1/'

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

# Only training rows
train_df = data[data['grp'] == "train"]


# Test set from CSV
test_df = data[data['grp'] == "test"]

# Check patient ID distribution across folds
data['patient_id'] = data['filename'].apply(lambda x: x.split('-')[2])
fold_counts = data.groupby('patient_id')['fold'].nunique()

# Any patient appearing in more than 1 fold?
print(fold_counts[fold_counts > 1])
