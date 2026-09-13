"""
Make figures for your report / slides - all methods side by side.

Run:  python -m src.visualize --index 0

Produces:
  outputs/fig_overlays_<i>.png   original + one overlay per method
  outputs/fig_curves_<i>.png     deletion & insertion curves for all methods
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from . import config as C
from .data import get_datasets, denormalize
from .model import load_trained
from .explain import get_explanation, predict_class
from .faithfulness import deletion_insertion


def _raw01(x):
    return np.clip(denormalize(x[0]).permute(1, 2, 0).cpu().numpy(), 0, 1)


def overlay(img01, heat, alpha=0.5):
    import cv2
    hm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1 - alpha) * img01 + alpha * hm, 0, 1)


def main(index):
    net = load_trained()
    _, test_ds = get_datasets()
    x, y = test_ds[index]
    x = x.unsqueeze(0).to(C.DEVICE)
    pred = predict_class(net, x)
    img01 = _raw01(x)
    methods = C.METHODS

    maps = {m: get_explanation(m, net, x, img01, pred) for m in methods}

    # ---- overlays grid ----
    ncol = len(methods) + 1
    fig, ax = plt.subplots(1, ncol, figsize=(3 * ncol, 3.4))
    ax[0].imshow(img01); ax[0].set_title(f"Original\n(pred={pred}, true={int(y)})")
    ax[0].axis("off")
    for j, m in enumerate(methods, 1):
        ax[j].imshow(overlay(img01, maps[m]))
        ax[j].set_title(C.METHOD_INFO.get(m, (m, ""))[0])
        ax[j].axis("off")
    fig.tight_layout()
    p1 = C.OUTPUT_DIR / f"fig_overlays_{index}.png"
    fig.savefig(p1, dpi=130); plt.close(fig)

    # ---- curves ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in methods:
        name = C.METHOD_INFO.get(m, (m, ""))[0]
        xd, yd, _ = deletion_insertion(net, x, maps[m], pred, mode="deletion")
        xi, yi, _ = deletion_insertion(net, x, maps[m], pred, mode="insertion")
        ax[0].plot(xd, yd, label=name)
        ax[1].plot(xi, yi, label=name)
    ax[0].set_title("Deletion (steeper drop = more faithful)")
    ax[0].set_xlabel("fraction removed"); ax[0].set_ylabel("class prob"); ax[0].legend(fontsize=8)
    ax[1].set_title("Insertion (steeper rise = more faithful)")
    ax[1].set_xlabel("fraction inserted"); ax[1].set_ylabel("class prob"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    p2 = C.OUTPUT_DIR / f"fig_curves_{index}.png"
    fig.savefig(p2, dpi=130); plt.close(fig)

    print(f"Saved -> {p1}\n         {p2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    main(ap.parse_args().index)
