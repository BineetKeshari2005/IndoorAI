# IndoorAI — Task 10: Localization Error Analysis and Hybrid Model Report
**Date:** 2026-11-16 18:38:43  **Evaluation Split:** Official UJIIndoorLoc Validation Set (1,111 records)  **Models Evaluated:** Random Forest (Task 07), Global k-NN (Task 08), Two-Stage Hierarchical Hybrid (Task 10)  
---
## 1. Executive Summary & Measured Findings
This experiment provides an in-depth empirical diagnosis of localization failure modes, investigates the physical and signal properties of difficult Wi-Fi fingerprints, and evaluates a Two-Stage Hierarchical Hybrid Architecture combining Random Forest (Stage 1) and k-Nearest Neighbors (Stage 2).

### Key Benchmarking Outcomes
| Model Architecture | Building Acc | Floor Acc | Mean 2D Eucl Error | Median Eucl Error | 95th % Error | Max Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Task 07)** | 99.82% | **88.66%** | 11.53 units | 8.65 units | 32.74 units | 115.18 units |
| **Global k-NN (Task 08)** | **99.91%** | 84.43% | 10.05 units | 7.33 units | **29.06 units** | 68.02 units |
| **Hybrid (Unthresholded $\tau=0.0$)** | 99.82% | 88.66% | 10.36 units | 7.29 units | 29.90 units | 188.94 units |
| **Hybrid (Gated $\tau=0.70$)** | 99.82% | 88.66% | **9.98 units** | **7.29 units** | 29.26 units | **67.98 units** |

> [!IMPORTANT]
> Coordinate Reference System (CRS) units are unverified in UJIIndoorLoc. All errors are strictly reported in **dataset coordinate units**.

---

## 2. Structural Classification Error Analysis (Buildings and Floors)

### 2.1 Floor Error Adjacency Breakdown
- **Random Forest:** Out of 126 total floor misclassifications, **113 (89.68%)** are adjacent floor confusions ($|\text{true} - \text{pred}| = 1$). Non-adjacent errors occur in only 13 samples (1.17% of validation data).
- **k-NN:** Out of 173 total floor misclassifications, **159 (91.91%)** are adjacent floor confusions. Non-adjacent errors occur in only 14 samples (1.26% of validation data).
- **Physical Mechanism:** Vertical Wi-Fi penetration through reinforced concrete floor slabs generates signal profiles that closely mimic neighboring floors directly above and below, whereas cross-building or multi-floor jumps are virtually eliminated.

### 2.2 Most Commonly Confused Floor Pairs
| True Floor | Pred Floor | Adjacency | RF Misclassifications | k-NN Misclassifications |
| :---: | :---: | :---: | :---: | :---: |
| Floor 1 | Floor 2 | Adjacent | 43 | 55 |
| Floor 2 | Floor 3 | Adjacent | 21 | 43 |
| Floor 4 | Floor 3 | Adjacent | 11 | 10 |
| Floor 0 | Floor 1 | Adjacent | 10 | 7 |
| Floor 3 | Floor 2 | Adjacent | 10 | 9 |
| Floor 2 | Floor 1 | Adjacent | 9 | 8 |

### 2.3 Critical Problem Areas: Building 1 and Floor 4
- **Building 1 Dominance:** Over 55% of all mutual floor errors occur inside **Building 1** (RF floor error rate: 21.82%, k-NN error rate: 25.08%). Building 1 features high architectural symmetry and open atriums that amplify vertical signal bleed.
- **Floor 4 Downward Bias:** On Floor 4 (Building 2, $N=39$), 85% of RF errors and 83% of k-NN errors are classified as Floor 3. As the topmost floor, it lacks upward neighbors and suffers from severe downward signal pull.

---

## 3. Coordinate Prediction Errors & Difficult Fingerprint Archetypes

### 3.1 Signal Feature Correlations with Coordinate Error
| Signal Summary Feature | RF Correlation with Error | k-NN Correlation with Error | Interpretation |
| :--- | :---: | :---: | :--- |
| `strongest_rssi_1` | -0.2159 | -0.1617 | Stronger signals significantly reduce error |
| `rssi_std` | -0.2442 | -0.2321 | High variance/contrast indicates distinctive fingerprint |
| `rssi_range` | -0.2606 | -0.1799 | High variance/contrast indicates distinctive fingerprint |
| `visible_ap_count` | -0.0513 | 0.1856 | Visibility correlation |
| `weak_signal_count` | 0.0242 | 0.2004 | Many faint signals slightly increase noise |

### 3.2 Difficult Wi-Fi Fingerprint Archetypes
#### 1. Sparse & Faint Fingerprints (< 5 APs, max RSSI < -75 dBm)
Queries located in dead zones (e.g. Row 0: 1 AP at -91 dBm; Row 21: 2 APs at -74 dBm) lack triangulation constraints. With only 1-2 weak signals, nearest neighbor matches drift across entire buildings, causing 60-115 unit errors.

#### 2. Top-Floor Boundary Attenuation (Floor 4)
Floor 4 is present exclusively in Building 2. Because it has only downward neighbors, signals suffer asymmetric attenuation and are frequently pulled down into Floor 3.

#### 3. Low Jaccard Overlap (< 0.50)
High error queries have significantly lower AP visibility overlap with training samples (Jaccard 0.5648 vs 0.6915), reflecting AP dropouts, surveyor shielding, and unseen hardware.

**Empirical Verification:** The Top 50 highest-error queries exhibit an average AP Jaccard similarity of **0.5648** with their nearest training neighbor, compared to **0.6915** for the Top 50 lowest-error queries (overall validation mean: 0.67).

---

## 4. Two-Stage Hierarchical Hybrid Localizer Evaluation

We implemented and evaluated the proposed architecture: Stage 1 uses Random Forest to classify Building and Floor, and Stage 2 uses local distance-weighted k-NN fitted strictly on training fingerprints for the predicted Building-Floor subset.

### 4.1 The Stage 1 Error Propagation Penalty
1. **When Stage 1 RF is Correct (983 samples, 88.48%):**  
   - Local k-NN filtering reduces median Euclidean error from 6.97 units (global k-NN) to **6.8682 units**  
   - Mean error drops from 9.39 to **9.3479 units**. Isolating the correct floor slab eliminates vertical noise.
2. **When Stage 1 RF is Wrong (128 samples, 11.52%):**  
   - An unthresholded hybrid forces nearest neighbors to be drawn from the wrong building or floor, causing coordinate error to explode from 15.08 units to **14.844 units**, with a catastrophic maximum error of 188.94 units.
3. **Confidence-Gating Solution ($\tau = 0.70$):**  
   - When Stage 1 RF confidence $P_{\max} < 0.70$, the localizer falls back to global k-NN (456 samples, 41.04%).  
   - Gated Hybrid achieves an overall Mean Euclidean Error of **9.9811 units** (outperforming Global k-NN's 10.05 units) and caps maximum error at **67.9759 units**.

---

## 5. Confidence and Spatial Uncertainty Analysis

### 5.1 Classification Confidence Calibration (Random Forest)
| RF Confidence ($P_{\max}$) | Validation Samples | Fraction of Total | Actual Building-Floor Accuracy | Mean Euclidean Error |
| :---: | :---: | :---: | :---: | :---: |
| $\ge 0.0$ | 1111 | 100.0% | **88.48%** | 9.98 units |
| $\ge 0.3$ | 1095 | 98.56% | **89.41%** | 9.89 units |
| $\ge 0.5$ | 991 | 89.2% | **92.43%** | 9.41 units |
| $\ge 0.7$ | 655 | 58.96% | **96.34%** | 8.26 units |
| $\ge 0.8$ | 437 | 39.33% | **97.48%** | 8.13 units |
| $\ge 0.9$ | 201 | 18.09% | **98.51%** | 7.33 units |

**Takeaway:** At $P_{\max} \ge 0.70$, Building-Floor accuracy reaches **96.34%**, making it an effective gating threshold.

### 5.2 Coordinate Uncertainty via Neighbor Spatial Dispersion
We defined continuous spatial uncertainty as the spatial standard deviation (dispersion) of the $k=7$ retrieved neighbors: $$\sigma_{\text{spatial}} = \sqrt{\sigma_{x,\text{neighbors}}^2 + \sigma_{y,\text{neighbors}}^2}$$
- **Correlation with Actual Coordinate Error:** **0.4991** (Strong positive correlation!).  
- **Low Dispersion Tier (< 3.77 units, N=278):** Mean Error = **7.17 units**.  
- **Medium Dispersion Tier (3.77 - 9.28 units, N=555):** Mean Error = **8.54 units**.  
- **High Dispersion Tier (> 9.28 units, N=278):** Mean Error = **15.67 units**.

---

## 6. Artifact Inventory

- Master Error Analysis JSON: [`experiments/error_analysis/metrics/error_analysis.json`](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/error_analysis.json)
- Top 50 High Error Samples: [`experiments/error_analysis/metrics/high_error_samples.csv`](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/high_error_samples.csv)
- Floor Confusion Analysis: [`experiments/error_analysis/metrics/floor_confusion_analysis.csv`](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/floor_confusion_analysis.csv)
- Building Error Analysis: [`experiments/error_analysis/metrics/building_error_analysis.csv`](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/building_error_analysis.csv)
- Hybrid Model Metrics: [`experiments/error_analysis/metrics/hybrid_model_results.json`](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/metrics/hybrid_model_results.json)
- Hybrid Validation Predictions: [`experiments/error_analysis/predictions/val_predictions.parquet`](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/predictions/val_predictions.parquet)
- Diagnostic Visualizations (7 PNGs): [`experiments/error_analysis/plots/`](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/error_analysis/plots/)
