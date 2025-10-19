import torch
import torch.nn as nn
import timm


class CarClassifier(nn.Module):
    """
    Car classification model using transfer learning
    """

    def __init__(self, num_classes: int, model_name: str = 'efficientnet_b0', pretrained: bool = True):
        """
        Args:
            num_classes: Number of car classes to predict
            model_name: Name of the backbone model from timm
            pretrained: Whether to use ImageNet pretrained weights
        """
        super(CarClassifier, self).__init__()

        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

        print(f"Created {model_name} with {num_classes} output classes")

    def forward(self, x):
        return self.backbone(x)


def create_model(num_classes: int, model_name: str = 'efficientnet_b0', device: str = 'cuda'):
    """
    Factory function to create and initialize model

    Popular model options:
    - efficientnet_b0: Fast, accurate, good baseline
    - efficientnet_b3: More accurate, slower
    - resnet50: Classic, reliable
    - convnext_tiny: Modern architecture
    - vit_small_patch16_224: Vision transformer

    Args:
        num_classes: Number of classes
        model_name: Model architecture name
        device: Device to load model on

    Returns:
        model on specified device
    """
    model = CarClassifier(num_classes=num_classes, model_name=model_name, pretrained=True)

    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    elif device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
        print("Model loaded on Apple Silicon GPU (MPS)")
    else:
        model = model.cpu()
        print("Model loaded on CPU")

    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return total, trainable
