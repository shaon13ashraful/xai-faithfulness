"""
Rigorous scoring for the method comparison.

Turns raw per-image metrics into:
  - mean +/- std for every (method, metric)
  - a per-metric ranking of all methods
  - an overall average-rank leaderboard
  - a combined faithfulness score (Insertion AUC - Deletion AUC)
  - pairwise Wilcoxon signed-rank tests on the key metric
  - formatted text tables ready to paste into a report

Metric directions (whether higher or lower is "more faithful"):
    deletion_auc      : lower  is better
    insertion_auc     : higher is better
    combined          : higher is better  (insertion - deletion)
    sanity_corr       : lower  is better
    localization      : higher is better  (optional)
    seconds           : lower  is better  (cost, not faithfulness)
"""
import numpy as np

# direction: +1 means higher is better, -1 means lower is better
METRIC_DIRECTION = {
    "deletion_auc": -1,
    "insertion_auc": +1,
    "combined": +1,
    "sanity_corr": -1,
    "localization": +1,
    "seconds": -1,
}

METRIC_LABEL = {
    "deletion_auc": "Deletion AUC (lower=better)",
    "insertion_auc": "Insertion AUC (higher=better)",
    "combined": "Combined = Ins - Del (higher=better)",
    "sanity_corr": "Sanity corr (lower=better)",
    "localization": "Localization (higher=better)",
    "seconds": "Sec / explanation (lower=better)",
}


def aggregate(per_image, methods, metrics):
    """
    per_image: dict method -> dict metric -> list of per-image values
    Returns stats: method -> metric -> (mean, std, n)
    """
    stats = {}
    for m in methods:
        stats[m] = {}
        for k in metrics:
            vals = np.asarray(per_image[m].get(k, []), dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                stats[m][k] = (float("nan"), float("nan"), 0)
            else:
                stats[m][k] = (float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0), len(vals))
    return stats


def rank_methods(stats, methods, metric):
    """
    Rank methods on one metric (1 = best) respecting the metric direction.
    Returns dict method -> rank (ties share the averaged rank).
    """
    direction = METRIC_DIRECTION[metric]
    means = np.array([stats[m][metric][0] for m in methods], dtype=np.float64)
    # sort so that best is first
    order_vals = -means if direction > 0 else means
    # rank with ties averaged
    order = np.argsort(order_vals, kind="mergesort")
    ranks = np.empty(len(methods), dtype=np.float64)
    sorted_vals = order_vals[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return {methods[i]: float(ranks[i]) for i in range(len(methods))}


def overall_leaderboard(stats, methods, ranking_metrics):
    """
    Average rank across the given metrics -> overall standing.
    Returns list of (method, avg_rank) sorted best-first.
    """
    per_metric_ranks = {mt: rank_methods(stats, methods, mt) for mt in ranking_metrics}
    avg = {}
    for m in methods:
        avg[m] = float(np.mean([per_metric_ranks[mt][m] for mt in ranking_metrics]))
    board = sorted(avg.items(), key=lambda kv: kv[1])
    return board, per_metric_ranks


def wilcoxon_pairs(per_image, methods, metric, baseline="gradcam"):
    """
    Pairwise Wilcoxon signed-rank test comparing each method against a baseline
    on a per-image metric. Returns dict method -> (statistic, p_value).
    Requires scipy; degrades gracefully if unavailable.
    """
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    out = {}
    base_vals = np.asarray(per_image[baseline][metric], dtype=np.float64)
    for m in methods:
        if m == baseline:
            continue
        v = np.asarray(per_image[m][metric], dtype=np.float64)
        n = min(len(base_vals), len(v))
        a, b = base_vals[:n], v[:n]
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 5 or np.all(a == b):
            out[m] = (float("nan"), float("nan"))
            continue
        try:
            stat, p = wilcoxon(a, b)
            out[m] = (float(stat), float(p))
        except Exception:
            out[m] = (float("nan"), float("nan"))
    return out


# ----------------------------------------------------------------------
# formatted tables
# ----------------------------------------------------------------------
def _name(method, method_info):
    return method_info.get(method, (method, ""))[0]


def _family(method, method_info):
    return method_info.get(method, (method, ""))[1]


def format_scorecard(stats, methods, metrics, method_info):
    """Big mean +/- std table."""
    header_cols = ["Method", "Family"] + [METRIC_LABEL[k] for k in metrics]
    widths = [16, 12] + [30] * len(metrics)
    lines = []
    head = " | ".join(h.ljust(w) for h, w in zip(header_cols, widths))
    lines.append(head)
    lines.append("-" * len(head))
    for m in methods:
        row = [_name(m, method_info).ljust(widths[0]),
               _family(m, method_info).ljust(widths[1])]
        for k, w in zip(metrics, widths[2:]):
            mean, std, n = stats[m][k]
            if np.isnan(mean):
                cell = "n/a"
            else:
                cell = f"{mean:.4f} +/- {std:.4f}"
            row.append(cell.ljust(w))
        lines.append(" | ".join(row))
    return "\n".join(lines)


def format_leaderboard(board, per_metric_ranks, ranking_metrics, method_info):
    lines = ["OVERALL LEADERBOARD (by average rank across metrics; 1.0 = best)",
             "-" * 64]
    for pos, (m, avg_rank) in enumerate(board, 1):
        detail = ", ".join(f"{mt.split('_')[0]}={per_metric_ranks[mt][m]:.1f}"
                           for mt in ranking_metrics)
        lines.append(f"{pos}. {_name(m, method_info):<18} avg_rank={avg_rank:.2f}   ({detail})")
    return "\n".join(lines)


def format_wilcoxon(wx, metric, baseline, method_info):
    if wx is None:
        return ("Wilcoxon test skipped (scipy not installed).")
    lines = [f"WILCOXON signed-rank vs. {_name(baseline, method_info)}  "
             f"on {METRIC_LABEL[metric]}",
             "(p < 0.05 => difference is statistically significant)",
             "-" * 64]
    for m, (stat, p) in wx.items():
        if np.isnan(p):
            verdict = "n/a"
        elif p < 0.001:
            verdict = "*** highly significant"
        elif p < 0.01:
            verdict = "** significant"
        elif p < 0.05:
            verdict = "* significant"
        else:
            verdict = "not significant"
        lines.append(f"{_name(m, method_info):<18} stat={stat:>10.2f}  p={p:.4g}   {verdict}")
    return "\n".join(lines)
