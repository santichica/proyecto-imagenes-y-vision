"""
PyTorch Dataset for HAM10000 binary classification (Nevus vs Melanoma).
Loads images from split CSVs produced by 02_split.py.
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch


ROOT = Path(__file__).resolve().parents[2]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class HAM10000Dataset(Dataset):
    def __init__(self, split_csv: Path, transform=None):
        self.df = pd.read_csv(split_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(ROOT / row["image_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights per sample for WeightedRandomSampler."""
        counts = self.df["label"].value_counts().sort_index()
        weights_per_class = 1.0 / counts.values.astype(float)
        sample_weights = self.df["label"].map(
            {cls: w for cls, w in enumerate(weights_per_class)}
        ).values
        return torch.tensor(sample_weights, dtype=torch.float)


def make_loaders(
    splits_dir: Path,
    batch_size: int,
    num_workers: int = 4,
    device: torch.device | None = None,
):
    train_ds = HAM10000Dataset(splits_dir / "train.csv", transform=TRAIN_TRANSFORMS)
    val_ds   = HAM10000Dataset(splits_dir / "val.csv",   transform=EVAL_TRANSFORMS)
    test_ds  = HAM10000Dataset(splits_dir / "test.csv",  transform=EVAL_TRANSFORMS)

    sampler = WeightedRandomSampler(
        weights=train_ds.class_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )

    pin_memory = device is not None and device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
