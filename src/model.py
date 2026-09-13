"""
Model: ResNet-18 pretrained on ImageNet, with a fresh classification head
for the 37 pet breeds. Transfer learning keeps training fast on a CPU.
"""
import torch
import torch.nn as nn
from torchvision import models

from . import config as C


def build_model(freeze_backbone: bool = None) -> nn.Module:
    if freeze_backbone is None:
        freeze_backbone = C.FREEZE_BACKBONE

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False

    # Replace the final fully connected layer with one for our classes.
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, C.NUM_CLASSES)  # new head is always trainable
    return net.to(C.DEVICE)


def target_layer(net: nn.Module):
    """
    The layer Grad-CAM hooks into. For ResNet the last conv block
    (layer4) is the standard, recommended choice.
    """
    return net.layer4[-1]


def load_trained(path=None) -> nn.Module:
    path = path or C.MODEL_PATH
    net = build_model()
    state = torch.load(path, map_location=C.DEVICE)
    net.load_state_dict(state)
    net.eval()
    return net
