import os
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class CarDataset(Dataset):
    """
    Unified dataset class that handles multiple car datasets:
    - Most_Stolen_Cars
    - SubsetVMMR
    - CompCars
    """

    def __init__(self, root_dir: str, transform=None, include_datasets: List[str] = None):
        """
        Args:
            root_dir: Root directory containing all datasets
            transform: Optional transform to be applied on images
            include_datasets: List of datasets to include ['stolen_cars', 'vmmr', 'compcars']
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if include_datasets is None:
            include_datasets = ['stolen_cars', 'vmmr', 'compcars']

        # Load samples from each dataset
        if 'stolen_cars' in include_datasets:
            self._load_most_stolen_cars()

        if 'vmmr' in include_datasets:
            self._load_vmmr()

        if 'compcars' in include_datasets:
            self._load_compcars()

        print(f"Total samples loaded: {len(self.samples)}")
        print(f"Total classes: {len(self.class_to_idx)}")

    def _load_most_stolen_cars(self):
        """Load Most_Stolen_Cars dataset"""
        dataset_path = self.root_dir / 'Dataset' / 'Most_Stolen_Cars'
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} not found")
            return

        for class_dir in sorted(dataset_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    class_idx = len(self.class_to_idx)
                    self.class_to_idx[class_name] = class_idx
                    self.idx_to_class[class_idx] = class_name

                class_idx = self.class_to_idx[class_name]

                for img_file in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_file), class_idx, class_name))
                for img_file in class_dir.glob('*.jpeg'):
                    self.samples.append((str(img_file), class_idx, class_name))
                for img_file in class_dir.glob('*.png'):
                    self.samples.append((str(img_file), class_idx, class_name))

        print(f"Loaded Most_Stolen_Cars: {len([s for s in self.samples if 'Most_Stolen_Cars' in s[0]])} samples")

    def _load_vmmr(self):
        """Load SubsetVMMR dataset"""
        dataset_path = self.root_dir / 'Dataset' / 'SubsetVMMR'
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} not found")
            return

        for class_dir in sorted(dataset_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    class_idx = len(self.class_to_idx)
                    self.class_to_idx[class_name] = class_idx
                    self.idx_to_class[class_idx] = class_name

                class_idx = self.class_to_idx[class_name]

                for img_file in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_file), class_idx, class_name))
                for img_file in class_dir.glob('*.jpeg'):
                    self.samples.append((str(img_file), class_idx, class_name))
                for img_file in class_dir.glob('*.png'):
                    self.samples.append((str(img_file), class_idx, class_name))

        print(f"Loaded SubsetVMMR: {len([s for s in self.samples if 'SubsetVMMR' in s[0]])} samples")

    def _load_compcars(self):
        """Load CompCars dataset"""
        dataset_path = self.root_dir / 'compcarsdb' / 'image'
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} not found")
            return

        # CompCars has structure: image/make_id/model_id/year/image.jpg
        for make_dir in sorted(dataset_path.iterdir()):
            if make_dir.is_dir():
                for model_dir in make_dir.iterdir():
                    if model_dir.is_dir():
                        for year_dir in model_dir.iterdir():
                            if year_dir.is_dir():
                                # Create class name from make/model/year
                                class_name = f"compcars_{make_dir.name}_{model_dir.name}_{year_dir.name}"

                                if class_name not in self.class_to_idx:
                                    class_idx = len(self.class_to_idx)
                                    self.class_to_idx[class_name] = class_idx
                                    self.idx_to_class[class_idx] = class_name

                                class_idx = self.class_to_idx[class_name]

                                for img_file in year_dir.glob('*.jpg'):
                                    self.samples.append((str(img_file), class_idx, class_name))
                                for img_file in year_dir.glob('*.jpeg'):
                                    self.samples.append((str(img_file), class_idx, class_name))
                                for img_file in year_dir.glob('*.png'):
                                    self.samples.append((str(img_file), class_idx, class_name))

        print(f"Loaded CompCars: {len([s for s in self.samples if 'compcarsdb' in s[0]])} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, class_idx


def get_transforms(train: bool = True, img_size: int = 224):
    """
    Get data transforms for training or validation

    Args:
        train: If True, includes data augmentation
        img_size: Input image size
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
    img_size: int = 224,
    include_datasets: List[str] = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders

    Returns:
        train_loader, val_loader, class_info
    """
    # Create full dataset
    train_transform = get_transforms(train=True, img_size=img_size)
    val_transform = get_transforms(train=False, img_size=img_size)

    full_dataset = CarDataset(root_dir, transform=train_transform, include_datasets=include_datasets)

    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform

    # Pin memory only for CUDA (not supported on MPS)
    use_pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    class_info = {
        'class_to_idx': full_dataset.class_to_idx,
        'idx_to_class': full_dataset.idx_to_class,
        'num_classes': len(full_dataset.class_to_idx)
    }

    return train_loader, val_loader, class_info
