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
    if not torch.cuda.is_available():
        print("GPU no disponible — usando CPU")
        return torch.device("cpu")

    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    mem = props.total_memory / 1e9
    cc = (props.major, props.minor)

    supported_archs = torch.cuda.get_arch_list()
    gpu_sm = f"sm_{props.major}{props.minor}"
    if gpu_sm not in supported_archs:
        print(
            f"GPU detectada ({name}, {gpu_sm}) no es compatible con esta build de PyTorch.\n"
            f"  PyTorch soporta: {', '.join(supported_archs)}\n"
            f"  Cayendo a CPU para evitar errores de kernel."
        )
        return torch.device("cpu")

    print(f"GPU: {name} ({mem:.1f} GB, CC {cc[0]}.{cc[1]})")
    return torch.device("cuda")
