"""
Stage 3 driver - run the full multi-method faithfulness comparison.

Run:  python -m src.run_experiment

Evaluates every method in config.METHODS across all test images and writes:
  outputs/results.csv     per-image, per-method metrics (long format)
  outputs/scorecard.txt   mean +/- std table
  outputs/leaderboard.txt overall average-rank standings
  outputs/wilcoxon.txt    pairwise significance tests
  outputs/summary.txt      everything combined (paste this into your report)
"""
import csv
import time
import numpy as np
import torch
from tqdm import tqdm

from . import config as C
from . import scoring as S
from .data import get_datasets, denormalize
from .model import load_trained
from .explain import get_explanation, predict_class
from .faithfulness import deletion_insertion, randomize_model, map_similarity


def _to_raw01(input_tensor):
    img = denormalize(input_tensor[0]).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)


# metrics we collect per image, per method
BASE_METRICS = ["deletion_auc", "insertion_auc", "combined", "sanity_corr", "seconds"]
# metrics used to compute the overall leaderboard (cost excluded on purpose)
RANKING_METRICS = ["deletion_auc", "insertion_auc", "combined", "sanity_corr"]


def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    net = load_trained()
    net.eval()
    rnd_net = randomize_model(net)

    _, test_ds = get_datasets()
    n = min(C.N_EVAL_IMAGES, len(test_ds))
    methods = C.METHODS
    print(f"Evaluating {len(methods)} methods on {n} images "
          f"(device={C.DEVICE})...")

    # per_image[method][metric] -> list
    per_image = {m: {k: [] for k in BASE_METRICS} for m in methods}
    long_rows = []

    for i in tqdm(range(n)):
        x, y = test_ds[i]
        x = x.unsqueeze(0).to(C.DEVICE)
        pred = predict_class(net, x)
        raw01 = _to_raw01(x)

        for m in methods:
            t0 = time.time()
            attr = get_explanation(m, net, x, raw01, pred)
            secs = time.time() - t0

            _, _, del_auc = deletion_insertion(net, x, attr, pred, mode="deletion")
            _, _, ins_auc = deletion_insertion(net, x, attr, pred, mode="insertion")
            combined = ins_auc - del_auc

            if C.SANITY_ALL_METHODS:
                attr_rnd = get_explanation(m, rnd_net, x, raw01, pred)
                sanity = map_similarity(attr, attr_rnd)
            else:
                sanity = float("nan")

            per_image[m]["deletion_auc"].append(del_auc)
            per_image[m]["insertion_auc"].append(ins_auc)
            per_image[m]["combined"].append(combined)
            per_image[m]["sanity_corr"].append(sanity)
            per_image[m]["seconds"].append(secs)

            long_rows.append({
                "idx": i, "label": int(y), "pred": pred, "method": m,
                "deletion_auc": del_auc, "insertion_auc": ins_auc,
                "combined": combined, "sanity_corr": sanity, "seconds": secs,
            })

    # ---- write per-image CSV (long format, easy to pivot in Excel/pandas) ----
    csv_path = C.OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)

    # ---- aggregate + rank ----
    stats = S.aggregate(per_image, methods, BASE_METRICS)
    board, per_metric_ranks = S.overall_leaderboard(stats, methods, RANKING_METRICS)

    scorecard = S.format_scorecard(stats, methods, BASE_METRICS, C.METHOD_INFO)
    leaderboard = S.format_leaderboard(board, per_metric_ranks, RANKING_METRICS, C.METHOD_INFO)

    # ---- Wilcoxon on the combined score vs. Grad-CAM baseline ----
    if C.RUN_WILCOXON:
        wx = S.wilcoxon_pairs(per_image, methods, "combined", baseline="gradcam")
        wilcoxon_txt = S.format_wilcoxon(wx, "combined", "gradcam", C.METHOD_INFO)
    else:
        wilcoxon_txt = "Wilcoxon test disabled in config."

    # ---- write individual files ----
    (C.OUTPUT_DIR / "scorecard.txt").write_text(scorecard)
    (C.OUTPUT_DIR / "leaderboard.txt").write_text(leaderboard)
    (C.OUTPUT_DIR / "wilcoxon.txt").write_text(wilcoxon_txt)

    summary = (
        f"FAITHFULNESS COMPARISON  (n={n} images, {len(methods)} methods)\n"
        f"{'=' * 64}\n\n"
        f"{scorecard}\n\n{'=' * 64}\n{leaderboard}\n\n{'=' * 64}\n{wilcoxon_txt}\n\n"
        "Notes:\n"
        "  - 'combined' = Insertion AUC - Deletion AUC is the single headline\n"
        "    faithfulness number (higher = better).\n"
        "  - The leaderboard averages ranks across deletion, insertion, combined\n"
        "    and sanity, so no single metric dominates.\n"
        "  - 'seconds' is compute cost per explanation; use it to discuss the\n"
        "    faithfulness-vs-speed trade-off (Score-CAM/SHAP/RISE are the slow ones).\n"
    )
    (C.OUTPUT_DIR / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"Saved -> {csv_path} and outputs/*.txt")


if __name__ == "__main__":
    main()
