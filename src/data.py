"""
Data handling for the Oxford-IIIT Pet dataset.

torchvision downloads the dataset automatically the first time this runs
(~800 MB). It also ships segmentation trimaps, which we expose for the
optional localization test.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from . import config as C


def build_transforms(train: bool):
    """Standard ImageNet-style transforms. Training adds light augmentation."""
    if train:
        return transforms.Compose([
            transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(C.MEAN, C.STD),
        ])
    return transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.MEAN, C.STD),
    ])


def get_datasets():
    """Return (train_ds, test_ds) using the official split, category labels."""
    train_ds = OxfordIIITPet(
        root=str(C.DATA_DIR),
        split="trainval",
        target_types="category",
        transform=build_transforms(train=True),
        download=True,
    )
    test_ds = OxfordIIITPet(
        root=str(C.DATA_DIR),
        split="test",
        target_types="category",
        transform=build_transforms(train=False),
        download=True,
    )
    return train_ds, test_ds


def get_loaders():
    train_ds, test_ds = get_datasets()
    train_loader = DataLoader(
        train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
        num_workers=C.NUM_WORKERS, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=C.BATCH_SIZE, shuffle=False,
        num_workers=C.NUM_WORKERS, pin_memory=False,
    )
    return train_loader, test_loader


def get_eval_dataset_with_masks():
    """
    Test dataset returning BOTH the category label and the segmentation mask.
    Used only by the optional localization test. Images are returned as
    normalized tensors; masks as single-channel tensors at IMG_SIZE.
    """
    img_tf = build_transforms(train=False)
    mask_tf = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.PILToTensor(),   # keeps integer class ids in the trimap
    ])
    return OxfordIIITPet(
        root=str(C.DATA_DIR),
        split="test",
        target_types=["category", "segmentation"],
        transform=img_tf,
        target_transform=None,   # we transform the tuple manually below
        download=True,
    ), img_tf, mask_tf


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display / for masking in raw pixel space."""
    mean = torch.tensor(C.MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(C.STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
