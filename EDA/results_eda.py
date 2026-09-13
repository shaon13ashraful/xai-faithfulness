import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font="DejaVu Sans")

# Paths are resolved relative to the project root (the parent of EDA/) so the
# script runs from any working directory.
ROOT = Path(__file__).resolve().parent.parent
csv_path = ROOT / "outputs" / "results.csv"
out = ROOT / "outputs" / "eda_results"
os.makedirs(out, exist_ok=True)

if not csv_path.exists():
    raise SystemExit(f"{csv_path} not found - run `python -m src.run_experiment` first.")

names = {
    "gradcam": "Grad-CAM",
    "gradcampp": "Grad-CAM++",
    "scorecam": "Score-CAM",
    "integrated_grad": "Integrated Grad.",
    "shap": "SHAP",
    "rise": "RISE",
    "occlusion": "Occlusion",
}
order = ["Grad-CAM", "Grad-CAM++", "Score-CAM", "Integrated Grad.", "SHAP", "RISE", "Occlusion"]
teal, green, rust, deep = "#1C7293", "#2E8B57", "#B85042", "#065A82"

df = pd.read_csv(csv_path)
df["method_name"] = df["method"].map(names)
n_imgs = df["idx"].nunique()

print(f"rows {len(df)}, images {n_imgs}, methods {df['method'].nunique()}")

summary = (df.groupby("method_name")[["deletion_auc", "insertion_auc", "combined",
           "sanity_corr", "seconds"]].mean().reindex(order))
print(summary.round(4).to_string())
summary.to_csv(out / "summary_by_method.csv")

means = df.groupby("method_name")["combined"].mean().reindex(order)
stds = df.groupby("method_name")["combined"].std().reindex(order)
bar_colors = [green if m in ("Occlusion", "RISE") else rust if m == "Integrated Grad." else teal
              for m in order]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(range(len(order)), means.values, xerr=stds.values, color=bar_colors,
        error_kw=dict(ecolor="#888", capsize=3, lw=1))
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlabel("Combined faithfulness (Insertion - Deletion)")
ax.set_title(f"Mean combined score per method (n={n_imgs} images)\nerror bars = standard deviation")
plt.tight_layout()
plt.savefig(f"{out}/01_combined_bar.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(9, 4.8))
sns.boxplot(data=df, x="combined", y="method_name", order=order, ax=ax,
            color="#DCE7EC", fliersize=0, width=0.6)
sns.stripplot(data=df, x="combined", y="method_name", order=order, ax=ax,
              color=deep, size=6, alpha=0.8, jitter=0.15)
ax.set_xlabel("Combined faithfulness score")
ax.set_ylabel("")
ax.set_title(f"Spread of combined scores across the {n_imgs} images")
plt.tight_layout()
plt.savefig(f"{out}/02_combined_distribution.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 6))
palette = sns.color_palette("husl", len(order))
for m, c in zip(order, palette):
    sub = df[df["method_name"] == m]
    ax.scatter(sub["deletion_auc"], sub["insertion_auc"], label=m, s=55, color=c, edgecolor="white")
    ax.scatter(sub["deletion_auc"].mean(), sub["insertion_auc"].mean(),
               marker="X", s=180, color=c, edgecolor="black", linewidth=1.2, zorder=5)
ax.set_xlabel("Deletion AUC  (lower = better)")
ax.set_ylabel("Insertion AUC  (higher = better)")
ax.set_title("Deletion vs insertion\n(big X = method mean; top-left is the ideal corner)")
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{out}/03_deletion_vs_insertion.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
corr = df[["deletion_auc", "insertion_auc", "combined", "sanity_corr", "seconds"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.8}, ax=ax,
            xticklabels=["del", "ins", "comb", "sanity", "sec"],
            yticklabels=["del", "ins", "comb", "sanity", "sec"])
ax.set_title("Correlation between metrics\n(across all method-image rows)")
plt.tight_layout()
plt.savefig(f"{out}/04_metric_correlation.png", dpi=140)
plt.close()

sec = df.groupby("method_name")["seconds"].mean().reindex(order)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(range(len(order)), sec.values, color=deep)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Seconds per explanation (log scale)")
ax.set_title("Compute cost per method (log scale)")
for i, v in enumerate(sec.values):
    ax.text(v, i, f" {v:.1f}s", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{out}/05_compute_cost.png", dpi=140)
plt.close()

img_mean = df.groupby("idx")["combined"].mean()
img_std = df.groupby("idx")["combined"].std()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(img_mean.index.astype(str), img_mean.values, yerr=img_std.values,
       color=teal, error_kw=dict(ecolor="#888", capsize=4))
ax.set_xlabel("Image index")
ax.set_ylabel("Mean combined score (all methods)")
ax.set_title("Which images were easy to explain?\n(higher = methods agreed the explanation was faithful)")
plt.tight_layout()
plt.savefig(f"{out}/06_image_difficulty.png", dpi=140)
plt.close()

print()
print(f"combined vs insertion  {df['combined'].corr(df['insertion_auc']):.3f}")
print(f"combined vs deletion   {df['combined'].corr(df['deletion_auc']):.3f}")
print(f"combined vs sanity     {df['combined'].corr(df['sanity_corr']):.3f}")
print(f"combined vs seconds    {df['combined'].corr(df['seconds']):.3f}")
print(f"fastest {sec.idxmin()} {sec.min():.2f}s, slowest {sec.idxmax()} {sec.max():.1f}s, "
      f"ratio {sec.max()/sec.min():.0f}x")
print(img_mean.round(3).to_string())
