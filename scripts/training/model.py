"""
EfficientNet-B0 classifier head for binary skin lesion classification.
"""
import timm
import torch
import torch.nn as nn


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")
    else:
        dev = torch.device("cpu")
        print("GPU no disponible — usando CPU")
    return dev
