"""
Central configuration for the whole project.
Edit values here instead of hard-coding them in the scripts.
"""
from pathlib import Path
import torch

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # dataset is downloaded here automatically
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_pets.pt"

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
# Oxford-IIIT Pet: 37 breeds of cats and dogs, ~7,400 images, with
# segmentation masks (trimaps) that we use for the optional localization test.
NUM_CLASSES = 37
IMG_SIZE = 224                    # ResNet expects 224x224
# ImageNet normalisation (required because we use ImageNet-pretrained weights)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------
# Training (kept light so it finishes on a CPU laptop via transfer learning)
# ----------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 4                        # transfer learning converges fast; raise if you have time
LR = 1e-3                         # only the new head is trained at this LR
FREEZE_BACKBONE = True           # True = fast (train only the final layer)
NUM_WORKERS = 2                  # set to 0 on Windows if you hit multiprocessing errors
SEED = 42

# ----------------------------------------------------------------------
# Faithfulness experiment settings  (THOROUGH profile)
# ----------------------------------------------------------------------
N_EVAL_IMAGES = 6               # first-pass value (~0.8 h on a 4-core CPU laptop).
                                # 6 is the FLOOR for the Wilcoxon test: at n=5 the
                                # smallest attainable two-sided p is 0.0625, so
                                # p<0.05 is unreachable and wilcoxon.txt becomes
                                # meaningless. Do not go lower.
                                # All per-explanation budgets below are left at
                                # their THOROUGH values on purpose, so every
                                # method is compared at an equal budget; only the
                                # number of images is reduced. Set back to 200
                                # for the full run (~26 h).
                                # More images = more reliable stats (slower).
DELETION_STEPS = 50             # finer deletion/insertion curves

# Perturbation-based method budgets (higher = slower, more accurate)
SHAP_BACKGROUND = 50            # background samples for SHAP
SHAP_NSAMPLES = 200             # evaluations per SHAP explanation
RISE_N_MASKS = 4000             # random masks for RISE (thorough)
RISE_MASK_SCALE = 7             # low-res mask grid before upsampling
RISE_PROB = 0.5                 # prob a cell is on, in each RISE mask
OCCLUSION_PATCH = 16            # occlusion window size (px)
OCCLUSION_STRIDE = 8            # occlusion stride (px)
IG_STEPS = 64                   # integration steps for Integrated Gradients
SCORECAM_BATCH = 32             # batch size for Score-CAM forward passes

# Methods to evaluate (comment any out to skip)
METHODS = [
    "gradcam",
    "gradcampp",       # Grad-CAM++
    "scorecam",        # Score-CAM (gradient-free CAM)
    "integrated_grad", # Integrated Gradients
    "shap",            # SHAP (blur masker)
    "rise",            # RISE (perturbation, model-agnostic)
    "occlusion",       # sliding-window occlusion
]

# Human-readable names + family, used in tables
METHOD_INFO = {
    "gradcam":         ("Grad-CAM",         "CAM"),
    "gradcampp":       ("Grad-CAM++",       "CAM"),
    "scorecam":        ("Score-CAM",        "CAM"),
    "integrated_grad": ("Integrated Grad.", "Gradient"),
    "shap":            ("SHAP",             "Perturbation"),
    "rise":            ("RISE",             "Perturbation"),
    "occlusion":       ("Occlusion",        "Perturbation"),
}

# Run the sanity check for every method (True) or just the CAM family (False).
SANITY_ALL_METHODS = True

# Statistical testing
RUN_WILCOXON = True             # pairwise Wilcoxon signed-rank on key metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
