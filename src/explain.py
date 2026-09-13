"""
Stage 2 - Generate explanations (six methods, three families).

Unified interface:
    get_explanation(name, net, input_tensor, raw01, class_idx) -> (H,W) map in [0,1]

Families:
  CAM         : gradcam, gradcampp, scorecam        (use last conv layer)
  Gradient    : integrated_grad                      (input-space gradients)
  Perturbation: shap, rise, occlusion                (probe the model by hiding pixels)

Every method returns a 2D attribution map of shape (IMG_SIZE, IMG_SIZE),
normalized to [0, 1], where higher = more important for the target class.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .model import target_layer


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _norm01(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    m = m - m.min()
    mx = m.max()
    if mx > 0:
        m = m / mx
    return m


@torch.no_grad()
def predict_class(net, input_tensor):
    net.eval()
    return int(net(input_tensor).argmax(1).item())


# ----------------------------------------------------------------------
# CAM family (Grad-CAM, Grad-CAM++, Score-CAM) via pytorch_grad_cam
# ----------------------------------------------------------------------
def _cam(method_cls, net, input_tensor, class_idx, **kw):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(int(class_idx))] if class_idx is not None else None
    cam = method_cls(model=net, target_layers=[target_layer(net)])
    # With FREEZE_BACKBONE the conv parameters have requires_grad=False, so the
    # activations at the target layer would carry no grad_fn and pytorch_grad_cam
    # would silently record no gradients (Grad-CAM then fails with grads=None).
    # Making the *input* require grad puts the whole forward pass back on the
    # autograd graph without unfreezing any weights.
    x = input_tensor.clone().detach().requires_grad_(True)
    grayscale = cam(input_tensor=x, targets=targets, **kw)  # (1,H,W)
    return _norm01(grayscale[0])


def gradcam_map(net, input_tensor, class_idx=None):
    from pytorch_grad_cam import GradCAM
    return _cam(GradCAM, net, input_tensor, class_idx)


def gradcampp_map(net, input_tensor, class_idx=None):
    from pytorch_grad_cam import GradCAMPlusPlus
    return _cam(GradCAMPlusPlus, net, input_tensor, class_idx)


def scorecam_map(net, input_tensor, class_idx=None):
    from pytorch_grad_cam import ScoreCAM
    # Score-CAM is gradient-free but does many forward passes internally.
    return _cam(ScoreCAM, net, input_tensor, class_idx)


# ----------------------------------------------------------------------
# Gradient family: Integrated Gradients (self-contained implementation)
# ----------------------------------------------------------------------
def integrated_gradients_map(net, input_tensor, class_idx, steps=None,
                             baseline=None):
    """
    Riemann-sum Integrated Gradients w.r.t. the target class logit.
    input_tensor: (1,3,H,W) normalized. Returns (H,W) in [0,1].
    """
    steps = steps or C.IG_STEPS
    net.eval()
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)   # black image baseline

    total_grad = torch.zeros_like(input_tensor)
    for k in range(1, steps + 1):
        alpha = k / steps
        x = (baseline + alpha * (input_tensor - baseline)).clone().detach()
        x.requires_grad_(True)
        logits = net(x)
        score = logits[0, int(class_idx)]
        net.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        score.backward()
        total_grad = total_grad + x.grad.detach()

    avg_grad = total_grad / steps
    ig = (input_tensor - baseline) * avg_grad          # (1,3,H,W)
    attr = ig[0].sum(0).abs().cpu().numpy()            # (H,W) magnitude
    return _norm01(attr)


# ----------------------------------------------------------------------
# Perturbation family: SHAP
# ----------------------------------------------------------------------
def _predict_fn(net):
    mean = np.array(C.MEAN, dtype=np.float32)
    std = np.array(C.STD, dtype=np.float32)

    def f(images):
        x = images.astype(np.float32) / (255.0 if images.max() > 1.5 else 1.0)
        x = (x - mean) / std
        x = torch.tensor(x).permute(0, 3, 1, 2).to(C.DEVICE)
        with torch.no_grad():
            probs = torch.softmax(net(x), dim=1)
        return probs.cpu().numpy()
    return f


def shap_map(net, raw01, class_idx, nsamples=None):
    import shap
    nsamples = nsamples or C.SHAP_NSAMPLES
    net.eval()
    f = _predict_fn(net)
    masker = shap.maskers.Image("blur(64,64)", raw01.shape)
    explainer = shap.Explainer(f, masker)
    sv = explainer(raw01[np.newaxis, ...], max_evals=nsamples,
                   batch_size=16, outputs=[int(class_idx)])
    vals = sv.values[0, ..., 0]              # (H,W,3)
    mag = np.abs(vals).sum(-1)               # (H,W)
    return _norm01(mag)


# ----------------------------------------------------------------------
# Perturbation family: RISE (Petsiuk et al., 2018)
# ----------------------------------------------------------------------
def rise_map(net, input_tensor, class_idx, n_masks=None, scale=None,
             prob=None, batch=64):
    """
    RISE: probe the model with many random low-res masks, weight each mask by
    the class probability it produces, and average. Returns (H,W) in [0,1].
    """
    n_masks = n_masks or C.RISE_N_MASKS
    scale = scale or C.RISE_MASK_SCALE
    prob = prob if prob is not None else C.RISE_PROB
    net.eval()
    _, _, H, W = input_tensor.shape
    device = input_tensor.device

    rng = np.random.default_rng(C.SEED)
    up_h = (H // scale + 1) * scale
    up_w = (W // scale + 1) * scale

    sal = np.zeros((H, W), dtype=np.float64)
    processed = 0
    while processed < n_masks:
        b = min(batch, n_masks - processed)
        grid = (rng.random((b, 1, scale, scale)) < prob).astype(np.float32)
        grid_t = torch.tensor(grid, device=device)
        up = F.interpolate(grid_t, size=(up_h, up_w),
                           mode="bilinear", align_corners=False)
        masks = torch.empty((b, 1, H, W), device=device)
        for j in range(b):
            oy = int(rng.integers(0, up_h - H + 1))
            ox = int(rng.integers(0, up_w - W + 1))
            masks[j, 0] = up[j, 0, oy:oy + H, ox:ox + W]
        masked = input_tensor * masks
        with torch.no_grad():
            probs = torch.softmax(net(masked), dim=1)[:, int(class_idx)]
        probs = probs.cpu().numpy()
        m_np = masks.cpu().numpy()[:, 0]
        sal += (probs[:, None, None] * m_np).sum(0)
        processed += b

    sal /= (n_masks * prob)
    return _norm01(sal)


# ----------------------------------------------------------------------
# Perturbation family: sliding-window Occlusion
# ----------------------------------------------------------------------
def occlusion_map(net, input_tensor, class_idx, patch=None, stride=None):
    """
    Slide a gray patch across the image; importance = drop in class prob when
    that region is covered. Returns (H,W) in [0,1].
    """
    patch = patch or C.OCCLUSION_PATCH
    stride = stride or C.OCCLUSION_STRIDE
    net.eval()
    _, _, H, W = input_tensor.shape

    with torch.no_grad():
        base = torch.softmax(net(input_tensor), dim=1)[0, int(class_idx)].item()

    heat = np.zeros((H, W), dtype=np.float64)
    counts = np.zeros((H, W), dtype=np.float64)
    fill = 0.0  # gray in normalized space ~ 0

    ys = sorted(set(list(range(0, H - patch + 1, stride)) + [H - patch]))
    xs = sorted(set(list(range(0, W - patch + 1, stride)) + [W - patch]))

    coords, batch_imgs = [], []
    B = 32

    def flush():
        nonlocal batch_imgs, coords
        if not batch_imgs:
            return
        x = torch.cat(batch_imgs, 0)
        with torch.no_grad():
            p = torch.softmax(net(x), dim=1)[:, int(class_idx)].cpu().numpy()
        for (yy, xx), pv in zip(coords, p):
            drop = base - pv
            heat[yy:yy + patch, xx:xx + patch] += drop
            counts[yy:yy + patch, xx:xx + patch] += 1
        batch_imgs, coords = [], []

    for y in ys:
        for x0 in xs:
            occ = input_tensor.clone()
            occ[0, :, y:y + patch, x0:x0 + patch] = fill
            batch_imgs.append(occ)
            coords.append((y, x0))
            if len(batch_imgs) == B:
                flush()
    flush()
    counts[counts == 0] = 1
    heat = heat / counts
    return _norm01(heat)


# ----------------------------------------------------------------------
# Unified dispatcher
# ----------------------------------------------------------------------
_DISPATCH = {
    "gradcam": lambda net, x, raw, c: gradcam_map(net, x, c),
    "gradcampp": lambda net, x, raw, c: gradcampp_map(net, x, c),
    "scorecam": lambda net, x, raw, c: scorecam_map(net, x, c),
    "integrated_grad": lambda net, x, raw, c: integrated_gradients_map(net, x, c),
    "shap": lambda net, x, raw, c: shap_map(net, raw, c),
    "rise": lambda net, x, raw, c: rise_map(net, x, c),
    "occlusion": lambda net, x, raw, c: occlusion_map(net, x, c),
}


def get_explanation(name, net, input_tensor, raw01, class_idx):
    """
    name: one of config.METHODS.
    input_tensor: (1,3,H,W) normalized tensor.
    raw01: (H,W,3) float image in [0,1] (needed by SHAP).
    Returns (H,W) attribution in [0,1].
    """
    if name not in _DISPATCH:
        raise ValueError(f"Unknown method: {name}")
    return _DISPATCH[name](net, input_tensor, raw01, class_idx)
