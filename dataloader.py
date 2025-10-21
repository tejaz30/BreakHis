import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class BreakHisDataset(Dataset):
    def __init__(self, data, root_dir, group=None, mag=None, fold=None, transform=None):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()

        # Filter by train/test group
        if group is not None:
            self.data = self.data[self.data['grp'] == group]

        # Filter by magnification
        if mag is not None:
            if isinstance(mag, str) and mag.endswith("X"):
                try:
                    mag = int(mag[:-1])
                except ValueError:
                    raise ValueError(f"Invalid magnification format: {mag}")
            self.data = self.data[self.data['mag'] == mag]

        # Filter by fold (optional)
        if fold is not None:
            if isinstance(fold, (list, tuple, set)):
                self.data = self.data[self.data['fold'].isin(fold)]
            else:
                self.data = self.data[self.data['fold'] == fold]

        self.data = self.data.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        # Label: benign vs malignant
        self.data['label'] = self.data['filename'].apply(lambda x: x.split('/')[3])
        self.class_names = sorted(self.data['label'].unique())
        self.data['label_int'] = self.data['label'].apply(lambda x: self.class_names.index(x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row['label_int']
        return image, label


