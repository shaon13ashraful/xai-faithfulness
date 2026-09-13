# Report Template

Skeleton for your write-up. Fill each section with your own results. Favor
clear prose over jargon.

## 1. Introduction
- Problem: classifiers are black boxes; explanation tools help, but a
  convincing-looking explanation can still be unfaithful to the model.
- Question: across six attribution methods from three families, which produce
  the most faithful explanations on this dataset, and are the differences
  statistically significant?

## 2. Related Work
- CAM family: Grad-CAM (Selvaraju 2017), Grad-CAM++ (Chattopadhay 2018),
  Score-CAM (Wang 2020).
- Gradient family: Integrated Gradients (Sundararajan 2017).
- Perturbation family: SHAP (Lundberg & Lee 2017), RISE (Petsiuk 2018),
  Occlusion (Zeiler & Fergus 2014).
- Evaluation: deletion/insertion (Petsiuk 2018), sanity checks (Adebayo 2018),
  SHAP-vs-Grad-CAM comparison (Tempel 2024).
- State what is known and what you add: a unified, statistically-tested
  six-method comparison on Oxford-IIIT Pets plus a localization analysis.

## 3. Data
- Oxford-IIIT Pet: 37 breeds, ~7,400 images, official split, segmentation masks.

## 4. Method
### 4.1 Classifier
- ResNet-18, ImageNet-pretrained, transfer learning. Report test accuracy.
### 4.2 Explanation methods
- Briefly describe each of the six and which family it belongs to.
- Note all maps are normalized to [0,1] for a fair comparison.
### 4.3 Faithfulness metrics
- Deletion AUC, Insertion AUC, Combined (Ins − Del), Sanity correlation,
  Localization, and compute cost (seconds).
### 4.4 Scoring and statistics
- Mean ± std; per-metric ranks; overall average-rank leaderboard;
  Wilcoxon signed-rank tests vs. Grad-CAM on the combined score.

## 5. Results
- Paste `outputs/scorecard.txt` (the mean ± std table).
- Paste `outputs/leaderboard.txt` (overall standings).
- Paste `outputs/wilcoxon.txt` (significance).
- Paste `outputs/localization.txt` if you ran it.
- Include 2–3 figures from `src/visualize.py` (all-method overlays + curves).
- Summarize the headline: which method won on `combined`, and by how much.

## 6. Discussion
- Do the three CAM variants agree, or does one dominate?
- Do the slow perturbation methods (SHAP, RISE) justify their cost with higher
  faithfulness? Quantify the speed gap from the `seconds` column.
- Any metric disagreements (faithful by deletion/insertion but poorly localized)?
  These are your most interesting findings.
- Which differences were statistically significant vs. Grad-CAM?

## 7. Limitations
- Single architecture; approximation budgets for SHAP/RISE/IG; evaluation set
  size; sanity check uses one randomization seed.

## 8. Conclusion
- One-paragraph, evidence-backed answer to your original question.

## References
- List the papers above in your required citation style.
