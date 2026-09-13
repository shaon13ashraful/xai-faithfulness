import os
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet

data_dir = "./data"
out = "./eda_dataset_figs"
os.makedirs(out, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

teal, deep = "#1C7293", "#065A82"

cat_breeds = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
}

print("loading dataset (downloads on first run)...")
ds = OxfordIIITPet(root=data_dir, split="trainval",
                   target_types=["category", "segmentation"], download=True)
test = OxfordIIITPet(root=data_dir, split="test", target_types=["category"], download=True)
classes = ds.classes
print(f"breeds {len(classes)}, trainval {len(ds)}, test {len(test)}")

counts = Counter()
for i in range(len(ds)):
    _, (label, _) = ds[i]
    counts[classes[label]] += 1

names_sorted = sorted(counts, key=lambda k: counts[k])
vals = [counts[n] for n in names_sorted]
colors = [teal if n in cat_breeds else deep for n in names_sorted]

fig, ax = plt.subplots(figsize=(9, 9))
ax.barh(range(len(names_sorted)), vals, color=colors)
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=8)
ax.set_xlabel("Number of images (trainval)")
ax.set_title("Images per breed  (teal = cat, dark blue = dog)")
plt.tight_layout()
plt.savefig(f"{out}/01_class_balance.png", dpi=140)
plt.close()

n_cat = sum(v for n, v in counts.items() if n in cat_breeds)
n_dog = sum(v for n, v in counts.items() if n not in cat_breeds)

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([n_cat, n_dog], labels=[f"Cats\n{n_cat}", f"Dogs\n{n_dog}"],
       colors=[teal, deep], autopct="%1.0f%%", startangle=90,
       textprops={"color": "white", "fontsize": 12, "weight": "bold"})
ax.set_title("Cat vs dog split (trainval)")
plt.tight_layout()
plt.savefig(f"{out}/02_cat_dog_split.png", dpi=140)
plt.close()

random.seed(42)
idxs = random.sample(range(len(ds)), min(800, len(ds)))
ws, hs, ars, obj_frac = [], [], [], []
for i in idxs:
    img, (label, seg) = ds[i]
    w, h = img.size
    ws.append(w)
    hs.append(h)
    ars.append(w / h)
    t = np.array(seg)
    obj_frac.append(float(((t == 1) | (t == 3)).mean()))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].hist(ws, bins=30, color=teal)
axes[0].set_title("Width (px)")
axes[0].set_xlabel("pixels")
axes[1].hist(hs, bins=30, color=teal)
axes[1].set_title("Height (px)")
axes[1].set_xlabel("pixels")
axes[2].hist(ars, bins=30, color=deep)
axes[2].set_title("Aspect ratio (w/h)")
axes[2].set_xlabel("ratio")
axes[2].axvline(1.0, color="black", ls="--", lw=1)
plt.suptitle(f"Image geometry (sample of {len(idxs)} images)")
plt.tight_layout()
plt.savefig(f"{out}/03_image_geometry.png", dpi=140)
plt.close()

means_rgb = []
for i in idxs:
    img, _ = ds[i]
    means_rgb.append(np.asarray(img.convert("RGB")).reshape(-1, 3).mean(0))
means_rgb = np.array(means_rgb)

fig, ax = plt.subplots(figsize=(7, 4.5))
for c, name, col in zip(range(3), ["Red", "Green", "Blue"], ["#C0392B", "#27AE60", "#2E5FA3"]):
    ax.hist(means_rgb[:, c], bins=30, alpha=0.6, label=name, color=col)
ax.set_xlabel("Per-image mean channel value (0-255)")
ax.set_ylabel("Number of images")
ax.set_title("Colour distribution across the dataset")
ax.legend()
plt.tight_layout()
plt.savefig(f"{out}/04_colour_stats.png", dpi=140)
plt.close()

first_of = {}
for i in range(len(ds)):
    img, (label, _) = ds[i]
    name = classes[label]
    if name not in first_of:
        first_of[name] = img
    if len(first_of) == len(classes):
        break

fig, axes = plt.subplots(5, 8, figsize=(16, 10))
for ax in axes.ravel():
    ax.axis("off")
for ax, name in zip(axes.ravel(), classes):
    ax.imshow(first_of[name].resize((120, 120)))
    ax.set_title(name, fontsize=7)
    ax.axis("off")
plt.suptitle("One sample per breed", fontsize=14)
plt.tight_layout()
plt.savefig(f"{out}/05_sample_gallery.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(np.array(obj_frac) * 100, bins=30, color=deep)
ax.set_xlabel("Percent of frame filled by the pet")
ax.set_ylabel("Number of images")
ax.set_title(f"How much of the image is the animal?  (sample of {len(idxs)})")
ax.axvline(np.mean(obj_frac) * 100, color="black", ls="--", lw=1,
           label=f"mean {np.mean(obj_frac)*100:.0f}%")
ax.legend()
plt.tight_layout()
plt.savefig(f"{out}/06_object_size.png", dpi=140)
plt.close()

print()
print(f"breeds {len(classes)}")
print(f"trainval {len(ds)}, test {len(test)}")
print(f"cats {n_cat}, dogs {n_dog}")
print(f"images per breed: min {min(vals)}, max {max(vals)}, mean {np.mean(vals):.1f}")
print(f"width mean {np.mean(ws):.0f} ({min(ws)}-{max(ws)})")
print(f"height mean {np.mean(hs):.0f} ({min(hs)}-{max(hs)})")
print(f"aspect ratio mean {np.mean(ars):.2f}")
print(f"pet fills frame mean {np.mean(obj_frac)*100:.0f}%")
