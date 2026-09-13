"""
Stage 1 - Train the classifier.

Run:  python -m src.train

Uses transfer learning (frozen ResNet-18 backbone + new head) so it finishes
in roughly 15-40 minutes on a typical CPU laptop. The trained weights are
saved to models/resnet18_pets.pt.
"""
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from . import config as C
from .data import get_loaders
from .model import build_model


def set_seed(seed=C.SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)
        preds = net(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    set_seed()
    print(f"Device: {C.DEVICE}")
    train_loader, test_loader = get_loaders()
    net = build_model()

    # Only parameters that require grad (the new head, unless backbone unfrozen)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=C.LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, C.EPOCHS + 1):
        net.train()
        running = 0.0
        t0 = time.time()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{C.EPOCHS}"):
            x, y = x.to(C.DEVICE), y.to(C.DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        acc = evaluate(net, test_loader)
        print(f"  loss={train_loss:.4f}  test_acc={acc:.3f}  "
              f"({time.time()-t0:.0f}s)")

    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), C.MODEL_PATH)
    print(f"Saved model -> {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
