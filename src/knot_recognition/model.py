from dataclasses import dataclass
import inspect

import torch.nn as nn
import torchvision.models as models


@dataclass(frozen=True)
class ResNetConfig:
    num_classes: int = 1000
    pretrained: bool = True
    model_name: str = "resnet18"
    freeze_backbone: bool = False


def build_resnet(config: ResNetConfig):
    return get_resnet(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        model_name=config.model_name,
        freeze_backbone=config.freeze_backbone,
    )


def get_resnet(num_classes=1000, pretrained=True, model_name="resnet18", freeze_backbone=False):
    ctor = getattr(models, model_name, None)
    if ctor is None:
        raise ValueError(f"Unknown model_name '{model_name}'. Expected torchvision resnet variant.")

    weights_map = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
    }

    sig = inspect.signature(ctor)
    if "weights" in sig.parameters:
        weights_enum = getattr(models, weights_map.get(model_name, ""), None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum is not None) else None
        model = ctor(weights=weights)
    elif "pretrained" in sig.parameters:
        model = ctor(pretrained=pretrained)
    else:
        model = ctor()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model
