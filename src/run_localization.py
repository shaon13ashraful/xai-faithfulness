"""
Optional Stage 3b - Localization test for ALL methods using Pet masks.

Run:  python -m src.run_localization

For each method, measures the fraction of explanation mass that lands on the
actual pet (vs. background), averaged over the test set, and ranks methods.

Produces: outputs/localization.txt  and  outputs/localization.csv
"""
import csv
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from . import config as C
from . import scoring as S
from .data import build_transforms, denormalize
from .model import load_trained
from .explain import get_explanation, predict_class
from .faithfulness import localization_score, trimap_to_object_mask


def main():
    net = load_trained()
    img_tf = build_transforms(train=False)
    mask_resize = transforms.Resize(
        (C.IMG_SIZE, C.IMG_SIZE),
        interpolation=transforms.InterpolationMode.NEAREST,
    )

    ds = OxfordIIITPet(
        root=str(C.DATA_DIR), split="test",
        target_types=["category", "segmentation"], download=True,
    )
    n = min(C.N_EVAL_IMAGES, len(ds))
    methods = C.METHODS
    per_image = {m: {"localization": []} for m in methods}
    rows = []

    for i in tqdm(range(n)):
        pil_img, (label, seg) = ds[i]
        x = img_tf(pil_img).unsqueeze(0).to(C.DEVICE)
        pred = predict_class(net, x)
        raw01 = np.clip(denormalize(x[0]).permute(1, 2, 0).cpu().numpy(), 0, 1)

        trimap = np.array(mask_resize(seg))
        obj_mask = trimap_to_object_mask(trimap)

        for m in methods:
            attr = get_explanation(m, net, x, raw01, pred)
            loc = localization_score(attr, obj_mask)
            per_image[m]["localization"].append(loc)
            rows.append({"idx": i, "method": m, "localization": loc})

    with open(C.OUTPUT_DIR / "localization.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "method", "localization"])
        w.writeheader(); w.writerows(rows)

    stats = S.aggregate(per_image, methods, ["localization"])
    ranks = S.rank_methods(stats, methods, "localization")
    order = sorted(methods, key=lambda m: ranks[m])

    lines = [f"LOCALIZATION TEST  (n={n} images)",
             "Fraction of explanation mass on the true pet (higher = better)",
             "-" * 60]
    for pos, m in enumerate(order, 1):
        mean, std, _ = stats[m]["localization"]
        name = C.METHOD_INFO.get(m, (m, ""))[0]
        lines.append(f"{pos}. {name:<18} {mean:.4f} +/- {std:.4f}")
    lines.append("")
    lines.append("A method faithful by deletion/insertion but LOW here may be")
    lines.append("relying on background context rather than the animal itself.")
    txt = "\n".join(lines)
    (C.OUTPUT_DIR / "localization.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
