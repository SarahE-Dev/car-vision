import warnings
# Suppress urllib3 OpenSSL warning on macOS (harmless)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import json
import os
from pathlib import Path
from datetime import datetime

from dataset import create_dataloaders
from model import create_model, count_parameters


class Trainer:
    """Training manager for car classification"""

    def __init__(
        self,
        root_dir: str,
        model_name: str = 'efficientnet_b0',
        batch_size: int = 32,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = None,
        checkpoint_dir: str = 'checkpoints',
        include_datasets: list = None
    ):
        """
        Args:
            root_dir: Root directory containing datasets
            model_name: Backbone model architecture
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            device: Device to train on (cuda/cpu/mps)
            checkpoint_dir: Directory to save checkpoints
            include_datasets: Which datasets to include
        """
        self.root_dir = root_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Auto-detect best device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Create dataloaders
        print("Loading datasets...")
        self.train_loader, self.val_loader, self.class_info = create_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            train_split=0.8,
            num_workers=4,
            include_datasets=include_datasets
        )

        self.num_classes = self.class_info['num_classes']
        print(f"Number of classes: {self.num_classes}")

        # Save class mapping
        self._save_class_mapping()

        # Create model
        print(f"Creating model: {model_name}")
        self.model = create_model(self.num_classes, model_name, self.device)
        count_parameters(self.model)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_acc = 0.0

    def _save_class_mapping(self):
        """Save class index mapping to JSON"""
        mapping_file = self.checkpoint_dir / 'class_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump({
                'class_to_idx': self.class_info['class_to_idx'],
                'idx_to_class': {str(k): v for k, v in self.class_info['idx_to_class'].items()},
                'num_classes': self.class_info['num_classes']
            }, f, indent=2)
        print(f"Class mapping saved to {mapping_file}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'history': self.history
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print("=" * 80)

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 80)

            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print epoch summary
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            self.save_checkpoint(epoch, val_acc, is_best)

        print("\n" + "=" * 80)
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")

        return self.history


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train car classification model')
    parser.add_argument('--root_dir', type=str, default='/Users/saraheatherly/Desktop/car-vision',
                        help='Root directory containing datasets')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        help='Model architecture (efficientnet_b0, resnet50, etc.)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--datasets', nargs='+', default=['stolen_cars', 'vmmr'],
                        help='Datasets to include (stolen_cars, vmmr, compcars)')

    args = parser.parse_args()

    trainer = Trainer(
        root_dir=args.root_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        include_datasets=args.datasets
    )

    trainer.train()


if __name__ == '__main__':
    main()
