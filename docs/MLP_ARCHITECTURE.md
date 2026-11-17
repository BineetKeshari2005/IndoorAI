# IndoorAI — Multitask Deep Learning (MLP) Architecture Specification

**Document Version:** 1.0  
**Phase:** Deep Learning Baselines (Task 12 of 30)  
**Author:** IndoorAI Core Engineering Team  
**Date:** November 2026  

---

## 1. Executive Summary & Architectural Philosophy

Task 12 introduces the first deep learning architecture to the IndoorAI localization stack: a unified **Multitask Multi-Layer Perceptron (MLP)**.

In previous tasks (Tasks 07, 08, 10), tabular machine learning models (Random Forest, $k$-NN, Two-Stage Hybrid Localizer) treated the localization problem as a collection of disjoint estimators:
- Random Forest trained 5 separate ensembles (561 MB on disk).
- $k$-NN maintained spatial nearest neighbor lookups ($k=7$, cosine distance, distance-weighted).
- The Two-Stage Hybrid required running Random Forest for coarse classification followed by floor-partitioned $k$-NN for regression.

The **Multitask MLP** fundamentally rearchitects indoor positioning by unifying discrete topological classification (Building, Floor) and continuous spatial regression (Longitude, Latitude) within a **single deep neural network with a shared latent feature representation**.

```
                           Input: 1,058 Wi-Fi Features
                     (520 RSSI + 520 Visibility + 18 Summary)
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Linear(1058 -> 512)       │
                         │   BatchNorm1d(512)          │
                         │   ReLU Activation           │
                         │   Dropout(p=0.30)           │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Linear(512 -> 256)        │
                         │   BatchNorm1d(256)          │
                         │   ReLU Activation           │
                         │   Dropout(p=0.25)           │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Linear(256 -> 128)        │
                         │   ReLU Activation           │
                         └──────────────┬──────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
               ▼                        ▼                        ▼
       ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
       │ Head 1: Bldg  │        │ Head 2: Floor │        │ Heads 3 & 4:  │
       │ Linear(128, 3)│        │ Linear(128, 5)│        │ Lon: (128, 1) │
       │ CrossEntropy  │        │ CrossEntropy  │        │ Lat: (128, 1) │
       │ Loss (w=1.0)  │        │ Loss (w=2.0)  │        │ SmoothL1(w=1) │
       └───────────────┘        └───────────────┘        └───────────────┘
```

---

## 2. Input Representation & Zero-Leakage Pipeline

The Multitask MLP ingests the audited $1,058$-dimensional feature matrix (`Variant B`) established in Tasks 05 and 06:

1. **520 Normalized RSSI Features (`WAP_001` to `WAP_520`):**
   Continuous signal strengths scaled to $[0.0, 1.0]$ via positive min-max normalization where $1.0$ represents maximum signal strength ($0\text{ dBm}$) and $0.0$ represents complete AP absence (clamped from raw $+100$ and $\le -105\text{ dBm}$).
2. **520 Binary AP Visibility Features (`WAP_001_vis` to `WAP_520_vis`):**
   Binary flags $\{0, 1\}$ indicating whether each AP was detected ($1$) or absent/unreachable ($0$).
3. **18 Engineered Signal Environment Features:**
   AP availability counts, ratios, RSSI mean, median, standard deviation, interquartile ranges, and strongest signals.

### Zero-Leakage Summary Scaling (`SummaryFeatureScaler`)
Because the 18 summary statistics have wide numerical ranges (e.g. `unavailable_ap_count` up to 520, `rssi_range` up to 95), raw gradients would be dominated by summary features. 
- The `SummaryFeatureScaler` fits a `MinMaxScaler(feature_range=(0.0, 1.0))` **strictly on the training feature matrix**.
- Binary visibility flags and normalized RSSI are preserved untouched.
- At inference time, incoming test fingerprints are transformed using the persisted scaler parameters.

### Target Coordinate Standardization (`CoordinateStandardizer`)
UJIIndoorLoc coordinates span large values:
- `LONGITUDE` $\in [-7691.34, -7300.82]$ (mean $\approx -7464.28$, std $\approx 123.40$).
- `LATITUDE` $\in [4864745.75, 4865017.37]$ (mean $\approx 4864871.0$, std $\approx 66.93$).

Raw coordinate regression loss without normalization produces MSE values on the order of $10^2$ to $10^3$, which would overwhelm the CrossEntropy loss ($\approx 0.05 - 1.5$) by several orders of magnitude.

The `CoordinateStandardizer` normalizes coordinates to zero mean and unit variance strictly on the training partition:
$$\tilde{X} = \frac{X - \mu_X}{\sigma_X}, \quad \tilde{Y} = \frac{Y - \mu_Y}{\sigma_Y}$$

During inference, predictions are inverted back to original dataset coordinate units:
$$\hat{X} = \hat{\tilde{X}} \cdot \sigma_X + \mu_X, \quad \hat{Y} = \hat{\tilde{Y}} \cdot \sigma_Y + \mu_Y$$

---

## 3. Multitask Loss Formulation & Balancing

The total objective function combines multi-class cross-entropy and robust coordinate regression:

$$\mathcal{L}_{\text{total}} = w_{\text{bldg}} \mathcal{L}_{\text{bldg}} + w_{\text{floor}} \mathcal{L}_{\text{floor}} + w_{\text{lon}} \mathcal{L}_{\text{lon}} + w_{\text{lat}} \mathcal{L}_{\text{lat}}$$

Where:
- $\mathcal{L}_{\text{bldg}} = -\sum_{c=0}^{2} y_{b,c} \log \hat{p}_{b,c}$ (Cross-Entropy Loss for 3 buildings)
- $\mathcal{L}_{\text{floor}} = -\sum_{c=0}^{4} y_{f,c} \log \hat{p}_{f,c}$ (Cross-Entropy Loss for 5 floors)
- $\mathcal{L}_{\text{lon}} = \text{SmoothL1}(\hat{\tilde{x}}, \tilde{x})$ (Huber loss with $\beta=1.0$ on standardized longitude)
- $\mathcal{L}_{\text{lat}} = \text{SmoothL1}(\hat{\tilde{y}}, \tilde{y})$ (Huber loss with $\beta=1.0$ on standardized latitude)

### Loss Weight Selection
- $w_{\text{bldg}} = 1.0$: Building identification converges rapidly to $>99.8\%$.
- $w_{\text{floor}} = 2.0$: Floor identification is historically the hardest classification task (vertical signal attenuation is severe across concrete slabs). Weighting floor loss higher forces the shared trunk to prioritize floor-discriminative representations.
- $w_{\text{lon}} = 1.0, w_{\text{lat}} = 1.0$: In standardized coordinate space, Smooth L1 residuals have magnitude $\approx 0.02 - 0.20$, balanced with classification loss.

---

## 4. Layer Specifications & Regularization

| Layer | Type | Input Dim | Output Dim | Activation / Normalization | Dropout | Parameter Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Feature Vector | - | 1,058 | - | - | 0 |
| **Trunk 1** | Dense Linear | 1,058 | 512 | ReLU + BatchNorm1d(512) | $p=0.30$ | 542,720 |
| **Trunk 2** | Dense Linear | 512 | 256 | ReLU + BatchNorm1d(256) | $p=0.25$ | 131,584 |
| **Trunk 3** | Dense Linear | 256 | 128 | ReLU | - | 32,896 |
| **Head 1** | Dense Linear | 128 | 3 | Raw Logits (Softmax in loss) | - | 387 |
| **Head 2** | Dense Linear | 128 | 5 | Raw Logits (Softmax in loss) | - | 645 |
| **Head 3** | Dense Linear | 128 | 1 | Linear Coordinate ($\hat{\tilde{X}}$) | - | 129 |
| **Head 4** | Dense Linear | 128 | 1 | Linear Coordinate ($\hat{\tilde{Y}}$) | - | 129 |
| **Total** | | | | | | **708,490** |

### Regularization Mechanisms
1. **Batch Normalization (`BatchNorm1d`):** Stabilizes internal covariate shifts across training batches and speeds up convergence.
2. **Progressive Dropout ($p=0.30 \rightarrow 0.25$):** Prevents individual hidden units from co-adapting to spurious Wi-Fi AP noise.
3. **Weight Decay ($L_2 = 10^{-4}$):** Penalizes large weight vectors in the Adam optimizer.
4. **Early Stopping:** Monitors internal validation loss with patience of 15 epochs, saving and restoring the best checkpoint.
5. **Learning Rate Scheduling:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=5, min_lr=1e-6)`.

---

## 5. Inference & Joint Target Derivation

In addition to the 4 direct model outputs, the inference engine automatically derives the joint **Building-Floor Index** (13 valid combinations across the campus):
$$\text{Building\_Floor\_Str} = \text{str}(\hat{B}) + \text{"\_"} + \text{str}(\hat{F})$$
$$\text{Building\_Floor\_Index} = \text{BUILDING\_FLOOR\_MAP}[\text{Building\_Floor\_Str}]$$

The model returns both hard predictions (`predict()`) and posterior class probabilities (`predict_proba()`), allowing downstream filtering, particle filters, or temporal smoothing.

---

## 6. Zero-Leakage & Benchmark Integrity Protocol

1. **Isolation of Official Validation Set:**
   The official $1,111$-row validation dataset (`X_validation_engineered.parquet` / `y_validation.parquet`) is NEVER touched during training, feature scaling fitting, target standardizer fitting, or hyperparameter selection.
2. **Forbidden Column Enforcement:**
   `IndoorAIMultitaskMLP` actively checks input DataFrames against `FORBIDDEN_COLUMNS` (`BUILDINGID`, `FLOOR`, `USERID`, `PHONEID`, `TIMESTAMP`, etc.) and raises a `ValueError` if any non-feature column enters the network.
3. **Coordinate Reporting Standard:**
   In compliance with project guidelines, all coordinate regression errors are reported strictly in **dataset coordinate units**, never unverified metres.

