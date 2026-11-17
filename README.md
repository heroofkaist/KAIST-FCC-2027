# KAIST-FCC: Football Possession AI for More Than A Hack 2027

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen.svg)](tests/)

Official submission repository by team **KAIST-FCC** for the **More Than A Hack 2027** competition organized by **Metrica Sports** and **FC Barcelona**.

---

## 1. Challenge Overview

The objective is to determine the possession state for every frame of a full professional football match (**FC Barcelona vs. Athletic Club**, 199,319 video frames at 25 fps, ~2 hours) from noisy 2D player and ball tracking data, optical homography, and tactical phase metadata.

Each frame $t \in [0, 199318]$ must be labeled with one of four classes:
- **`H`**: Home team possession (**FC Barcelona**).
- **`A`**: Away team possession (**Athletic Club**).
- **`D`**: Dead ball / stoppage (ball out of play, foul, referee whistle, celebration).
- **`X`**: Abstain / uninformative (outside graded match halves or during extreme occlusion).

The primary ranking metric is **`spellF1`**: the mean F1 score of one-to-one matched Home (`H`) and Away (`A`) possession intervals at temporal Intersection over Union (IoU) thresholds $\tau \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$. Dead-ball (`D`) intervals are stoppages, not possession spells. The secondary tie-breaking metric is **`macroF1`**.

---

## 2. Our Method: Multiscale Spatial-Temporal Ensemble

Our solution overcomes severe tracking dropouts (~41% missing ball frames) and fast turnovers through three integrated components:

```
Full Tactical ATD Tracking (199,319 frames)
                   │
                   ▼
  92-Dimensional Bilateral Feature Extractor
   ├── 59 Multiscale Temporal Window Features (0.2s - 24.0s)
   └── 33 Spatial Kinematics, Proximity & Team Geometry
                   │
                   ▼
   Dual Spatial LightGBM Ensemble (50 / 50)
   ├── spatial_lgb31 (31 leaves, 350 rounds)
   └── spatial_lgb63 (63 leaves, 350 rounds, L2=10)
                   │
                   ▼
  Age-Expanded Viterbi Dynamic Programming Decoder
   ├── Switch transition penalty & minimum duration priors
   └── Soft guidance from Smart Tagging tactical phases
                   │
                   ▼
 Final Validated Submission: submission.csv (199,319 rows)
```

### Key Technical Contributions
1. **92-Dimensional Spatial-Temporal Features:**
   - **Ball Kinematics (9 dims):** 2D velocity vectors, 2D acceleration, boundary exit flags, normalized pitch coordinates, and height indicators.
   - **Player-Ball Proximity & Density (24 dims):** Euclidean distances to the 1st, 2nd, and 3rd nearest players of each team; congestion counts within 2m, 5m, and 10m pressing spheres; team centroid distances; and longitudinal/lateral team spread.
   - **Multiscale Temporal Context (59 dims):** Class distribution statistics across 7 past and future horizons (0.2s to 24.0s), run-length tracking, and cross-gap continuity indicators.
2. **Exact Bilateral Team Symmetry:**
   - Both training and inference strictly enforce bilateral symmetry: swapping Home and Away player coordinates exactly swaps the Home and Away output probabilities ($P(H) \leftrightarrow P(A)$), preventing team-color or pitch-side bias.
3. **Temporal Dynamic Programming Decoder (Viterbi HMM):**
   - Eliminates per-frame flicker by decoding the joint optimal sequence under explicit transition switch costs and soft short-segment duration penalties, aligned with automatic Smart Tagging phase boundaries.

---

## 3. Strict Compliance with Competition Rules

- **Zero Ground Truth Leakage:** Strictly **NO** human annotations from the provided BAR–ATH competition match were used for training, feature selection, validation, or hyperparameter tuning.
- **Permitted External Data:** The external corpus contains 7 matches from the open-access **DFL/Sportec dataset**, with 6 used for development and model training and 1 held out for evaluation (Bassek et al., *Scientific Data* 2025, CC BY 4.0, Figshare [doi:10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177)).
- **Public 5-Minute Window:** The 5-minute public ground truth snippet (`evaluation/gt/BAR-ATH_possession_gt_public.csv`) was used solely as a one-time sanity check for coordinate alignment and output format validation.
- **Rule Verification:** All external training code, configurations, model weights, and inference pipelines are open, fully inspectable, and reproducible by the organizers.

---

## 4. Empirical Results on External Data

All model comparisons and tuning were performed using **group match-level separation** on the external Sportec corpus:

### Group Cross-Validation (6 Development Matches, 3 Missing-Ball Scenarios)
- **`clean`** (full tracking), **`masked`** (random 1s tracking loss, 35%), and **`long`** (2–8s correlated camera attrition + nearest-player noise):

| Model | Mean `spellF1` | Mean `macroF1` | Mean Dead `F1` | Clean `spellF1` | Masked `spellF1` | Long `spellF1` |
|---|---|---|---|---|---|---|
| Categorical Baseline (`lgb31`) | 0.3486 | 0.7356 | 0.8002 | 0.3783 | 0.3839 | 0.2835 |
| **Spatial Model (`spatial_lgb31`)** | **0.5199** | **0.8818** | **0.9554** | **0.6149** | **0.4958** | **0.4489** |
| **Relative Gain** | **+49.1%** | **+19.9%** | **+19.4%** | **+62.5%** | **+29.1%** | **+58.3%** |

### Independent Untouched Held-Out Test Match (`J03WR9`)
Tested on a match never seen during training or cross-validation:

| Metric | Categorical Baseline | Spatial Ensemble (`spatial_lgb31 + spatial_lgb63`) | Relative Improvement |
|---|---|---|---|
| Clean `spellF1` | 0.3558 | **0.6589** | **+85.2%** |
| Masked `spellF1` | 0.3783 | **0.5456** | **+44.2%** |
| Long Dropout `spellF1` | 0.3066 | **0.4924** | **+60.6%** |
| **Mean `spellF1`** | **0.3469** | **0.5656** | **+63.0%** |
| Overall `macroF1` | 0.7173 | **0.9037** | **+26.0%** |
| Dead-Ball `F1` | 0.7655 | **0.9604** | **+25.5%** |

### Public 5-Minute Sanity Check
- **`macroF1`:** **0.7821** (vs baseline 0.7018)
- **Overall Accuracy:** **81.73%** (error rate reduced from 26.73% to 18.27%)
- **Team Accuracy (both alive):** **90.75%**
- **Transition F1:** **0.3571** (vs baseline 0.2632)
- **Transition Latency:** **-1.3 frames (-0.05 seconds)**
- **Team Possession Share Error:** **0.0083** (< 1% error between Barcelona and Athletic)

---

## 5. Repository Structure

```
.
├── solution/                       # Production inference pipeline
│   ├── pipeline.py                 # Core orchestrator supporting tracking & bootstrap
│   ├── spatial.py                  # 33-dimensional kinematics & proximity extraction
│   ├── features.py                 # 59-dimensional multiscale temporal window features
│   ├── learned.py                  # Model loader and symmetric probability predictor
│   ├── decode.py                   # Age-expanded Viterbi dynamic programming
│   ├── evidence.py                 # Observation rewards and boundary detection
│   ├── io.py                       # Strict, safe readers for ATD, XML, tags, homography
│   └── config.py                   # Dataclass configuration loader
├── configs/                        # Frozen configuration & model artifacts
│   ├── learned-selected.json       # Selected inference parameters (transition costs, margins)
│   ├── learned-spatial-ensemble.json # Spatial ensemble calibration manifest
│   ├── possession.json             # Reference hyperparameter registry
│   └── models/
│       ├── spatial_lgb31/          # Trained LightGBM-31 spatial model (model.json, weights.txt)
│       └── spatial_lgb63/          # Trained LightGBM-63 spatial model (model.json, weights.txt)
├── evaluation/                     # Official evaluation and validation suite
│   ├── validate_submission.py      # Format, row-count, and alphabet validator
│   ├── run_evaluation.py           # Benchmark scorer against public ground truth
│   └── possession_metrics.py       # Official spellF1 and IoU matching algorithms
├── experiments/                    # External training and cross-validation suite
│   ├── corpus.py                   # Sportec & SkillCorner loader & downsampler
│   ├── protocol.py                 # Group cross-validation splits and dropout simulators
│   ├── learners.py                 # Symmetric model trainer (LightGBM, CatBoost, Torch)
│   └── research.py                 # 3-fold CV benchmark runner
├── tests/                          # 73 unit and integration tests
├── games.py                        # Match layout and asset path resolver
├── requirements-solution.txt       # Pinned runtime dependencies
├── TECHNICAL.md                    # In-depth technical report
├── LICENSE                         # MIT License
└── README.md                       # This document
```

---

## 6. Installation & Reproduction

### Prerequisites
- Python 3.10+ (tested under Python 3.14.7 on macOS and Linux).

### 1. Environment Setup
```bash
git clone https://github.com/heroofkaist/KAIST-FCC-2027.git
cd KAIST-FCC-2027

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-solution.txt
```

### 2. Verify Test Suite
Run the full 73-test suite to verify environment integrity:
```bash
python -m unittest discover -s tests -v
```

### 3. Place Match Data
Unpack the official competition assets into `data/tactical/` as described in `data/README.md`:
```
data/tactical/
├── BAR-ATH_tactical_atd.txt
├── BAR-ATH_tactical_atd.xml
├── BAR-ATH_tactical_homography.csv
└── BAR-ATH_tactical_smart_tagging.xml
```

### 4. Generate Official Submission
Run inference using our verified spatial ensemble:
```bash
python -m solution \
  --mode tracking \
  --config configs/learned-selected.json \
  --calibration configs/learned-spatial-ensemble.json \
  --out-dir artifacts/kaist-fcc-spatial-v1
```

### 5. Validate Output Format
```bash
python evaluation/validate_submission.py artifacts/kaist-fcc-spatial-v1/submission.csv
```

**Expected Checksum (SHA-256):**
```
f8b29a965b15d5c23cf8c30e6c0dc456674a87046aa8c94ee3733aebcf6e0acb  artifacts/kaist-fcc-spatial-v1/submission.csv
```

---

## 7. Limitations & Asset Usage

- **Homography:** While optical homography matrices are validated for geometric integrity by `validate_homography()`, they are not used to make possession decisions because the provided tactical ATD coordinates already reside in meters on pitch scale.
- **Broadcast Source:** The broadcast ATD tracking (`BAR-ATH_broadcast_atd.txt`) operates on an independent, non-aligned timeline and was not used for the final submission.
- **Video:** No video decoding is required; all features are derived from tracking coordinates and automatic Smart Tagging metadata.

---

## 8. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
