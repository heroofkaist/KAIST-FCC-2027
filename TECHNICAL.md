# KAIST-FCC Possession AI: Technical Documentation

**More Than A Hack 2027 — Metrica Sports & FC Barcelona**  
**Team:** KAIST-FCC  
**Target Repository:** [https://github.com/heroofkaist/KAIST-FCC-2027](https://github.com/heroofkaist/KAIST-FCC-2027)

---

## 1. Executive Summary

KAIST-FCC introduces a rule-compliant, multiscale spatial-temporal ensemble for automatic football possession tracking under heavy camera attrition, tactical occlusions, and dead-ball ambiguities. 

The approach couples:
1. **A 92-dimensional bilateral-symmetric feature representation** combining 59 temporal multiscale context windows with 33 physical 2D kinematics, player-ball proximity vectors, and spatial compactness metrics.
2. **A dual gradient-boosted spatial ensemble** (`spatial_lgb31` + `spatial_lgb63`) trained strictly on external professional matches (DFL/Sportec, CC BY 4.0) with exact team permutation invariance.
3. **An age-expanded temporal Viterbi dynamic programming decoder** that integrates possession duration dynamics and soft Smart Tagging phase transitions.

On external Group Cross-Validation across 6 development matches under simulated tracking dropouts, our spatial model achieves a mean `spellF1` of **0.5199** (vs 0.3486 baseline, +49.1% relative improvement). On an untouched held-out match (`J03WR9`), it achieves `spellF1 = 0.5656` (+63.0% relative improvement), with mean dead-ball F1 reaching 0.9604.

---

## 2. Zero-Leakage & Regulatory Compliance

- **No Human Annotations on BAR–ATH:** Strictly zero human labels from the competition match were used for training, calibration, model selection, or hyperparameter optimization.
- **External Data Provenance:** The open DFL/Sportec corpus contains 7 matches; 6 development matches were used for supervised training, while J03WR9 was held out for evaluation (Bassek et al., *Scientific Data* 2025, CC BY 4.0, Figshare doi:10.6084/m9.figshare.28196177).
- **Public 5-Minute Ground Truth:** The public ground truth window (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, frames 11,185–18,684) was used exclusively as a post-hoc sanity check for pipeline correctness and format validation, never for model training or tuning.
- **Reproducibility:** The pipeline is 100% deterministic and byte-for-byte reproducible.

---

## 3. Architecture & Methodology

```
┌────────────────────────────────────────────────────────┐
│  Full Tactical ATD Tracking (199,319 frames, 25 Hz)    │
│  BAR-ATH_tactical_atd.txt + BAR-ATH_tactical_atd.xml   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            Feature Extraction (92 Dimensions)          │
│  - 59 Multiscale Temporal Windows (0.2s to 24.0s)      │
│  - 33 Spatial Kinematics, Proximity & Compactness      │
│  - Exact Team Permutation Invariance (Bilateral Swap)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│       Dual Spatial LightGBM Ensemble (50 / 50)         │
│  - spatial_lgb31 (31 leaves, 350 rounds)               │
│  - spatial_lgb63 (63 leaves, 350 rounds, L2=10)        │
│  Output: Frame posterior probabilities P(H, A, D)      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│      Temporal Viterbi Dynamic Programming Decoder      │
│  - Log-likelihood emissions from spatial ensemble      │
│  - Transition cost & soft segment duration penalty     │
│  - Soft alignment with Smart Tagging phases            │
│  - Confidence gating (X abstention for uninformative)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│     Final Submission: submission.csv (199,319 rows)    │
│     SHA256: f8b29a965b15d5c23cf8c30e6c0dc456674a...   │
└────────────────────────────────────────────────────────┘
```

### 3.1 92-Dimensional Spatial-Temporal Features
1. **Kinematics & Ball State (9 features):**
   - Ball detection validity indicator.
   - Normalized pitch coordinates: $|x - L/2| / (L/2)$ and $|y - W/2| / (W/2)$.
   - Boundary exit flag (ball outside $[0, L] \times [0, W]$).
   - Ball speed magnitude (m/s) normalized to a 45 m/s maximum.
   - Ball acceleration magnitude.
   - Ball velocity validity flag.
   - Ball height validity and normalized vertical position.
2. **Player Proximity & Density (12 features per team $\times$ 2 = 24 features):**
   - Fraction of detected outfield players on screen.
   - Exact Euclidean distances to 1st, 2nd, and 3rd nearest players of each team.
   - Player counts within 2m, 5m, and 10m pressing radii.
   - Team centroid distance to ball.
   - Team spatial spread along pitch length and width (dispersion/compactness).
   - Mean player velocity and tracking coverage fraction.
3. **Multiscale Temporal Context (59 features):**
   - Running class densities of nearest-player ownership across 7 past and future horizons: 0.2s, 0.6s, 1.5s, 3.0s, 6.0s, 12.0s, 24.0s.
   - Run-length durations since last possession transition.
   - Elapsed time since last Home/Away observation and until next observation.
   - Gap continuity agreement indicators across missing tracking segments.

### 3.2 Bilateral Team Symmetry Enforcement
To prevent learned models from developing artificial biases towards Home or Away based on team colors or pitch sides, we enforce exact bilateral invariance during both training and inference:
$$P(H | x) = \frac{1}{2} \left( P_{\text{model}}(H | x) + P_{\text{model}}(A | \text{swap}(x)) \right)$$
$$P(A | x) = \frac{1}{2} \left( P_{\text{model}}(A | x) + P_{\text{model}}(H | \text{swap}(x)) \right)$$
where $\text{swap}(x)$ swaps all Home and Away feature coordinates while leaving ball kinematics and dead-ball features invariant.

### 3.3 Age-Expanded Viterbi Decoding
The frame-level log-posteriors are decoded using an exact dynamic program over the state space $\{H, A, D\}$ that penalizes spurious switches and enforces minimum duration priors:
$$\max_{s_{1:N}} \sum_{t=1}^N \log P(s_t | x_t) - \sum_{t=2}^N C_{\text{trans}} \cdot \mathbb{I}(s_t \neq s_{t-1}) - \sum_{\text{segments}} C_{\text{short}} \cdot \max\left(0, 1 - \frac{\text{len}}{T_{\text{mature}}}\right)$$

---

## 4. Empirical Evaluation on External Data

### 4.1 3-Fold Group Cross-Validation (6 Sportec Matches)
Evaluated across three missing-ball scenarios: `clean` (full tracking), `masked` (random 1s tracking loss, 35%), and `long` (correlated 2–8s camera pan-aways + nearest-player noise):

| Model | `mean_spellF1` | `mean_macroF1` | `mean_dead_f1` | Clean `spellF1` | Masked `spellF1` | Long `spellF1` |
|---|---|---|---|---|---|---|
| Categorical Baseline (`lgb31`) | 0.3486 | 0.7356 | 0.8002 | 0.3783 | 0.3839 | 0.2835 |
| **Spatial Model (`spatial_lgb31`)** | **0.5199** | **0.8818** | **0.9554** | **0.6149** | **0.4958** | **0.4489** |
| **Relative Improvement** | **+49.1%** | **+19.9%** | **+19.4%** | **+62.5%** | **+29.1%** | **+58.3%** |

### 4.2 Independent Held-Out Match (`J03WR9`)
Tested on an untouched match never seen during cross-validation:

| Scenario | Baseline `lgb31` (`spellF1`) | Spatial Ensemble (`spellF1`) | Spatial `macroF1` | Spatial `dead_f1` |
|---|---|---|---|---|
| `clean` | 0.3558 | **0.6589** (+85%) | **0.9419** | **0.9979** |
| `masked` | 0.3783 | **0.5456** (+44%) | **0.9019** | **0.9569** |
| `long` | 0.3066 | **0.4924** (+61%) | **0.8674** | **0.9263** |
| **Mean** | **0.3469** | **0.5656** (+63%) | **0.9037** | **0.9604** |

---

## 5. Public 5-Minute Sanity Check

Evaluated strictly for sanity-checking format and timing against `evaluation/gt/BAR-ATH_possession_gt_public.csv`:

- **Macro F1:** 0.7821 (vs baseline 0.7018)
- **Overall Accuracy:** 81.73% (error rate reduced from 26.73% to 18.27%)
- **Team Accuracy (when alive):** 90.75%
- **Transition F1:** 0.3571 (vs baseline 0.2632)
- **Transition Latency:** **-1.3 frames (-0.05 seconds)**
- **Possession Share Error:** **0.0083** (< 1% error between Barcelona and Athletic)

---

## 6. Exact Reproduction

```bash
# 1. Environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-solution.txt

# 2. Run unit test suite (73 tests)
python -m unittest discover -s tests -v

# 3. Generate candidate submission
python -m solution \
  --mode tracking \
  --config configs/learned-selected.json \
  --calibration configs/learned-spatial-ensemble.json \
  --out-dir artifacts/kaist-fcc-spatial-v1

# 4. Validate output format
python evaluation/validate_submission.py artifacts/kaist-fcc-spatial-v1/submission.csv
```

**Verified Output Checksum:**
- `artifacts/kaist-fcc-spatial-v1/submission.csv`: `f8b29a965b15d5c23cf8c30e6c0dc456674a87046aa8c94ee3733aebcf6e0acb`
