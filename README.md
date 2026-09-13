# Are Explanations Faithful? A Multi-Method Comparison on Oxford-IIIT Pets

**Course:** CSE710 — Advanced Pattern Recognition
**Author:** Md Ashraful Alam (ID: 16201101)

This project trains an image classifier, explains its predictions with **six**
attribution methods across **three families**, and then rigorously **tests
whether those explanations are faithful** — whether they reflect what the model
truly relies on, rather than just looking convincing.

The contribution is the *evaluation*: not producing heatmaps, but ranking them
with quantitative, statistically-tested metrics.

---

## Methods compared (6 methods, 3 families)

| Family | Methods |
|---|---|
| **CAM-based** | Grad-CAM, Grad-CAM++, Score-CAM |
| **Gradient-based** | Integrated Gradients |
| **Perturbation-based** | SHAP, RISE, Occlusion |

Comparing within and across families is a core part of the analysis: e.g. do
the three CAM variants agree? Do the slow perturbation methods actually earn
their cost in faithfulness?

---

## Faithfulness metrics (the scoring)

Each explanation is reduced to a normalized [0,1] importance map, then scored:

| Metric | What it measures | Faithful direction |
|---|---|---|
| **Deletion AUC** | confidence as important pixels are removed | lower better |
| **Insertion AUC** | confidence as important pixels are added back | higher better |
| **Combined** | Insertion − Deletion (single headline number) | higher better |
| **Sanity correlation** | map similarity on trained vs. randomized model | lower better |
| **Localization** (optional) | share of heatmap mass on the real pet | higher better |
| **Seconds** | compute cost per explanation | lower better |

The scoring goes beyond raw averages:

- **Mean ± standard deviation** for every method/metric (reliability, not just central value).
- **Per-metric ranking** of all methods, with tied ranks averaged.
- **Overall leaderboard** by average rank across deletion, insertion, combined and sanity — so no single metric dominates.
- **Wilcoxon signed-rank tests** comparing each method to a Grad-CAM baseline on the combined score, flagging whether differences are statistically significant (p < 0.05).

Everything is written to `outputs/` as both CSV (for your own analysis) and formatted text tables (to paste into the report).

---

## Setup (laptop, no GPU needed)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The dataset (~800 MB) downloads automatically on first run.

---

## Run it

```bash
# Stage 1 — train the classifier (~15-40 min on CPU). Saves models/resnet18_pets.pt
python -m src.train

# Stage 3 — full multi-method comparison. Writes outputs/summary.txt + more
python -m src.run_experiment

# Optional — localization test for all methods (uses segmentation masks)
python -m src.run_localization

# Figures for the report: all methods overlaid on one image + curves
python -m src.visualize --index 0
```

### Output files
```
outputs/results.csv        per-image, per-method metrics (long format)
outputs/scorecard.txt      mean +/- std table for every method
outputs/leaderboard.txt    overall average-rank standings
outputs/wilcoxon.txt       significance tests vs. Grad-CAM
outputs/summary.txt        all of the above combined (paste into report)
outputs/localization.txt   localization ranking (if you run it)
outputs/fig_*.png          figures
```

---

## This build uses the THOROUGH profile

Settings in `src/config.py` are tuned for accuracy over speed:
`N_EVAL_IMAGES=200`, `DELETION_STEPS=50`, `RISE_N_MASKS=4000`,
`SHAP_NSAMPLES=200`, `IG_STEPS=64`.

**This is deliberately heavy and will take a while on a CPU laptop.** The slow
methods are Score-CAM, SHAP, and RISE. If a full run is too long:

- Drop `N_EVAL_IMAGES` to 40–60 for a first pass (stats are still meaningful).
- Comment out the slowest methods in `METHODS` to iterate, then add them back
  for the final run.
- Lower `RISE_N_MASKS` (e.g. 1000) and `SHAP_NSAMPLES` (e.g. 80).

Tip: run the slow full job once, overnight, and keep `results.csv`. All tables
can be regenerated from that CSV without recomputing explanations.

---

## Project layout

```
xai-faithfulness/
├── README.md
├── requirements.txt
├── REPORT_TEMPLATE.md
├── src/
│   ├── config.py            # paths, hyperparameters, method list, scoring switches
│   ├── data.py              # dataset download, transforms, loaders, masks
│   ├── model.py             # ResNet-18 + transfer-learning head
│   ├── train.py             # Stage 1: training
│   ├── explain.py           # Stage 2: 6 methods behind get_explanation()
│   ├── faithfulness.py      # Stage 3: deletion/insertion, sanity, localization
│   ├── scoring.py           # aggregation, ranking, leaderboard, Wilcoxon, tables
│   ├── run_experiment.py    # Stage 3 driver -> all result files
│   ├── run_localization.py  # optional localization test (all methods)
│   └── visualize.py         # figures (all methods)
├── models/    outputs/    data/
```

---

## Scope and honesty

Comparing attribution methods under faithfulness metrics is an established
direction (Grad-CAM: Selvaraju 2017; Grad-CAM++: Chattopadhay 2018;
Score-CAM: Wang 2020; Integrated Gradients: Sundararajan 2017; SHAP:
Lundberg & Lee 2017; RISE + deletion/insertion: Petsiuk 2018; sanity checks:
Adebayo 2018). This project's contribution is a clean, reproducible,
statistically-tested comparison of all six on the Oxford-IIIT Pet dataset,
with a localization analysis on top. Cite these works and frame the project as
a careful re-evaluation, not a new method.
