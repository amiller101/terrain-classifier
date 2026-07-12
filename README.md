# Vision-Based Terrain Classification for Robotic Navigation

Classifying outdoor terrain (dirt, grass, asphalt, gravel) from monocular RGB robot video
using classical computer vision and machine learning — no deep learning. The system extracts
hand-engineered HOG + LBP features from image patches, classifies each patch with a classical
model, and reconstructs a full-frame terrain segmentation map from the per-patch predictions.

A solo project built on the [RUGD](http://rugd.vision) (Robot Unstructured Ground Driving)
dataset.

Best result: a strongly-regularized Logistic Regression on HOG+LBP features reached 94.11%
test accuracy on a held-out video, running end-to-end at ~2 FPS on CPU (no GPU).

> **Status: experimental / research code.** This repository is a record of an exploratory
> project, not a packaged, ship-ready application. It is intentionally structured for
> discovery and experimentation: there is no CLI or config file — experiment settings (which
> videos, patch size, strides, model choice, tuning on/off) are edited inline in `main()` in
> `terrain_classifier.py`. Expect to read and modify the code to run your own experiments.
> Paths use Windows-style separators in a few defaults, and some code paths (e.g. video
> export) are partial or commented out.

---

## Repository structure

```
terrain_classifier/
├── terrain_classifier.py     # top-level, runnable script (orchestrates everything)
├── image_preprocessing.py    # RUGD loading, grayscale/gamma, patch tiling + labeling, test/train splits
├── hog.py                    # custom NumPy Histogram of Oriented Gradients descriptor
├── lbp.py                    # Binary Pattern texture descriptor
├── requirements.txt          # Python dependencies
├── reports/                  # project proposal, midterm, and final reports (PDF)
└── data/                     # RUGD dataset (not committed — see "Dataset setup")
```

## How it works

1. **Preprocess** — RUGD frames are converted to grayscale, gamma-normalized (γ = 2.2), and
   tiled into overlapping square patches. Each patch is labeled with the terrain that
   dominates its (Gaussian-weighted) area. Training patches are kept only if one known
   terrain covers ≥ 60% of the patch; test patches are marked "unknown" unless ≥ 90% is
   known terrain. Train/test splits are made at the image or whole-video level to avoid
   leakage.
2. **Features** — each patch is described by a custom NumPy HOG descriptor (Sobel gradients,
   soft-binned orientation histograms, L2-Hys block normalization; validated against
   `skimage.feature.hog`) concatenated with an LBP texture histogram. Extraction is
   CPU-parallelized with `joblib`.
3. **Model** — features are standardized (`StandardScaler`), optionally reduced (PCA/SVD, off
   by default), and classified. Logistic Regression, SVM, and Random Forest are available;
   Logistic Regression (C = 0.01) is the default and best performer.
4. **Reconstruct** — per-patch class probabilities are voted back onto the frame; each pixel
   takes the argmax class and a color from the terrain lookup table, producing a full-frame
   terrain map (optionally exported as a video alongside the original RGB).


## Results summary

| Model | Train → Test video | Patch | Test accuracy |
| --- | --- | --- | --- |
| Logistic Regression | trail-9 → trail-10 | 81×64 | 94.11% |
| Logistic Regression | trail-9 → trail-10 | 64×64 | 91.11% |
| RBF SVM | trail-9 → trail-10 | 64×64 | ~93.5% |
| Random Forest | trail-9 → trail-10 | 64×64 | ~83% |

Adding LBP to HOG lifted accuracy from ~88% to ~92–93%. The final pipeline runs at
~0.531 s/frame (~2 FPS) on a laptop CPU (AMD Ryzen 9 5900HS, no GPU).


## Reports

The `reports/` folder contains the project write-ups, which are the most complete description
of the methodology, experiments, and results:

- `reports/ProjectProposal_Aaron_Miller.pdf`
- `reports/ProjectMidtermReport_Aaron_Miller.pdf`
- `reports/ProjectFinal_Aaron_Miller.pdf`


## Installation

Requires Python 3.x (developed on 3.14).

```bash
# 1. Clone
git clone https://github.com/amiller101/terrain-classifier
cd terrain_classifier

# 2. Create and activate a fresh virtual environment
python -m venv .venv

#    Windows (PowerShell)
.venv\Scripts\Activate.ps1
#    Windows (cmd)
.venv\Scripts\activate.bat
#    macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset setup

The code expects the [RUGD dataset](http://rugd.vision) under `data/`, with frames and their
pixel-level annotation maps in parallel per-video subfolders:

```
data/
├── RUGD_frames-with-annotations/
│   ├── trail-3/    # *.png RGB frames
│   ├── trail-9/
│   ├── trail-10/
│   └── ...
└── RUGD_annotations/
    ├── trail-3/    # *.png color-coded terrain labels (matching filenames)
    ├── trail-9/
    ├── trail-10/
    └── ...
```

The dataset is large and not included in this repo. It was last accessed at http://rugd.vision/.  
The annotation colors
this project recognizes are:  
dirt `(108, 64, 20)`, grass `(0, 102, 0)`, asphalt
`(64, 64, 64)`, gravel `(255, 128, 0)`  
All other classes are treated as unknown.

## Running

```bash
python terrain_classifier.py
```

Configuration is set inline in `main()`. The key configurables are:

- `videos`, `videos_train`, `videos_test` -- Which RUGD videos to use and how to split them.
- `patch_size`, `training_stride`, `testing_stride` -- Patch size and tiling density.
- `model_choice` (`"LogiReg"`, `"SVM"`, `"RF"`) and `tuning` -- Model selection and whether to
  run validation.
- `custom_PCA` -- Toggle the hand-written SVD-based PCA vs. scikit-learn's.

Reconstructed terrain maps are written to `results/<label>/` as PNG frames.

## References

- RUGD dataset — <http://rugd.vision>
- Dalal, N. & Triggs, B. *Histograms of Oriented Gradients for Human Detection.* CVPR, 2005.
- Ojala, T., Pietikäinen, M. & Mäenpää, T. *Multiresolution Gray-Scale and Rotation Invariant
  Texture Classification with Local Binary Patterns.* IEEE TPAMI, 2002.
- Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR, 2011.
