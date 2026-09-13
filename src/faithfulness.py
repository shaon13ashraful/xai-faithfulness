"""
Stage 3 - The core of the project: measure whether explanations are FAITHFUL.

Implements three tests from the slides:

  1. Deletion / Insertion  -> area under the confidence curve (AUC).
       Deletion: remove most-important pixels first; a faithful map makes
                 confidence fall FAST (low deletion-AUC = good).
       Insertion: add most-important pixels first into a blurred image;
                 a faithful map makes confidence rise FAST (high insertion-AUC = good).

  2. Sanity check (Adebayo et al.) -> similarity between the explanation on the
       trained model vs. a randomized model. LOW similarity = good (the map
       actually depends on the learned weights).

  3. Localization (optional, uses Pet segmentation masks) -> fraction of the
       explanation's mass that lands on the true object. HIGH = good.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

# np.trapz was renamed to np.trapezoid in NumPy 2.x; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ----------------------------------------------------------------------
# 1. Deletion / Insertion
# ----------------------------------------------------------------------
@torch.no_grad()
def _prob_of_class(net, x, class_idx):
    logits = net(x)
    return torch.softmax(logits, dim=1)[0, class_idx].item()


def deletion_insertion(net, input_tensor, attribution, class_idx,
                       steps=None, mode="deletion"):
    """
    input_tensor: (1,3,H,W) normalized.
    attribution:  (H,W) in [0,1].
    Returns (xs, ys, auc): xs = fraction of pixels changed, ys = class prob,
    auc = area under that curve (trapezoidal).
    """
    steps = steps or C.DELETION_STEPS
    device = input_tensor.device
    H, W = attribution.shape
    n_pixels = H * W

    order = np.argsort(attribution.flatten())[::-1]  # most important first

    if mode == "deletion":
        current = input_tensor.clone()
        baseline = torch.zeros_like(input_tensor)     # deleted pixels -> 0
    else:  # insertion
        # start from a blurred version, reveal important pixels into it
        current = _blur(input_tensor).clone()
        baseline = input_tensor.clone()

    ys = [_prob_of_class(net, current, class_idx)]
    xs = [0.0]

    per_step = max(1, n_pixels // steps)
    flat_idx = order

    for s in range(1, steps + 1):
        idx = flat_idx[(s - 1) * per_step: s * per_step]
        rows, cols = np.unravel_index(idx, (H, W))
        for c in range(3):
            current[0, c, rows, cols] = baseline[0, c, rows, cols]
        ys.append(_prob_of_class(net, current, class_idx))
        xs.append(min(1.0, s * per_step / n_pixels))

    xs = np.array(xs)
    ys = np.array(ys)
    auc = _trapz(ys, xs)
    return xs, ys, auc


def _blur(x, k=11, sigma=5.0):
    """Gaussian blur used as the 'absent' state for insertion."""
    channels = x.shape[1]
    coords = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.device)
    kernel = torch.outer(g, g)[None, None].repeat(channels, 1, 1, 1)
    return F.conv2d(x, kernel, padding=k // 2, groups=channels)


# ----------------------------------------------------------------------
# 2. Sanity check (model parameter randomization)
# ----------------------------------------------------------------------
def randomize_model(net):
    """Return a deep copy of the model with reinitialized weights."""
    import copy
    rnd = copy.deepcopy(net)
    for m in rnd.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    rnd.eval()
    return rnd


def map_similarity(map_a, map_b):
    """
    Similarity between two attribution maps via Pearson correlation of the
    flattened maps. Range ~[-1, 1]. For the sanity check we WANT this low.
    """
    a = map_a.flatten().astype(np.float64)
    b = map_b.flatten().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------------------------------------------------------------
# 3. Localization (optional; needs a binary object mask)
# ----------------------------------------------------------------------
def localization_score(attribution, object_mask):
    """
    Fraction of total attribution mass that falls inside the object mask.
    attribution: (H,W) in [0,1]. object_mask: (H,W) boolean/0-1.
    Returns value in [0,1]; higher = explanation concentrates on the object.
    """
    attribution = attribution.astype(np.float64)
    total = attribution.sum()
    if total == 0:
        return 0.0
    inside = (attribution * (object_mask > 0)).sum()
    return float(inside / total)


def trimap_to_object_mask(trimap):
    """
    Oxford Pet trimaps use: 1 = foreground (pet), 2 = background, 3 = boundary.
    We treat foreground + boundary as the object.
    trimap: (H,W) integer array. Returns boolean (H,W).
    """
    t = np.asarray(trimap)
    return (t == 1) | (t == 3)
