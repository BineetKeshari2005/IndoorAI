# IndoorAI — Task 11.1: Dashboard Accuracy & Consistency Audit Report

**Date:** 2026-11-16  
**Status:** COMPLETED & VERIFIED  
**Scope:** Model configuration alignment, benchmark metric verification, hybrid model audit, preprocessing validation, and live inference verification.  
**Strict Restraints Enforced:** No models trained or retrained; no dataset alterations; Task 12 not started.

---

## 1. Issues Discovered

During a comprehensive audit of the IndoorAI Streamlit dashboard (`frontend/app.py`) and its underlying prediction service (`backend/app/services/prediction_service.py`), five critical discrepancies and consistency issues were uncovered:

1. **k-NN Hyperparameter Configuration Discrepancy:**
   - In Task 08, 5-fold cross-validation selected $k=7$ as the optimal neighborhood size across all metric targets.
   - However, the initial dashboard code displayed $k=5$ in its prediction descriptions, benchmark model type, and architectural callouts.
   - Crucially, `prediction_service.py` had a hardcoded `n_neighbors=5` in its spatial uncertainty dispersion query (`kneighbors()`), while the underlying loaded model suite was trained with $k=7$.
2. **Fabricated / Unverified Benchmark Table Metrics:**
   - The initial static benchmark table in `PredictionService` (`VERIFIED_BENCHMARKS`) displayed numbers that did not match the independently audited and verified results from Task 09:
     - Random Forest Floor accuracy was displayed as $89.47\%$ (audited value: $88.66\%$).
     - Random Forest Mean 2D Error was displayed as $10.97\text{ units}$ (audited value: $11.53\text{ units}$).
     - k-NN Floor accuracy was displayed as $88.57\%$ (audited value: $84.43\%$).
     - k-NN Mean 2D Error was displayed as $9.38\text{ units}$ (audited value: $10.05\text{ units}$).
     - Hybrid Model Mean 2D Error was displayed as $8.44\text{ units}$ and Median as $5.68\text{ units}$ (audited Task 10 value: $9.98\text{ units}$ mean, $7.29\text{ units}$ median).
3. **Improper Conflation of Experimental Prototype with Audited Baselines:**
   - The Two-Stage Hybrid Localizer was presented as an established baseline alongside Random Forest and k-NN without disclosing that its confidence gating threshold ($\tau = 0.70$) was evaluated post-hoc on the validation partition rather than selected using training-only data.
4. **Uncalibrated Model Confidence Presentation:**
   - The maximum ensemble class probability ($P_{\max}$) was displayed as "Model Confidence" without clarification, creating the misleading impression that a high $P_{\max}$ guarantees low Euclidean coordinate error.
5. **Scikit-Learn Feature Name Warnings in Inference:**
   - Scikit-learn emitted `UserWarning` during k-NN neighbor querying due to DataFrame vs NumPy array mismatches between global k-NN and local k-NN regressors.

---

## 2. Root Cause Analysis

1. **Static Placeholder Ingestion:**
   In Task 11, the developer manually authored a dictionary literal (`VERIFIED_BENCHMARKS`) using unverified notes from an exploratory subset rather than programmatically loading the audited metrics from `experiments/model_benchmark/metrics/benchmark_results.json` and `experiments/model_benchmark/metrics/benchmark_comparison.csv`.
2. **Hardcoded Query Parameter:**
   In `PredictionService.predict()`, the author queried nearest neighbors with `n_neighbors=5` for spatial dispersion instead of dynamically referencing `self.knn_model.params.n_neighbors` ($k=7$).
3. **Post-Hoc Thresholding on Validation Data:**
   In Task 10 (`src/analysis/localization_error_analysis.py`), the confidence gating threshold $\tau = 0.70$ was observed to produce $96.34\%$ accuracy on the validation partition (`y_val`). While the local k-NN models were trained strictly on training data, the selection of $\tau=0.70$ was not tuned via nested cross-validation on the training set alone. Presenting this as an established baseline violated benchmark integrity.

---

## 3. Changes Made

1. **Dynamic k-NN Configuration ($k=7$):**
   - Updated `prediction_service.py` to query `self.knn_model.params.n_neighbors` ($k=7$) dynamically.
   - Updated `frontend/app.py` sidebar, prediction headers, and captions to display the verified $k=7$ configuration.
2. **Dynamic Benchmark Table Ingestion:**
   - Refactored `PredictionService.get_benchmark_table()` to programmatically read verified baseline metrics from `experiments/model_benchmark/metrics/benchmark_results.json`.
3. **Strict Separation of Experimental Prototypes:**
   - Created `PredictionService.get_experimental_hybrid_table()`, reading verified metrics from `experiments/error_analysis/metrics/hybrid_model_results.json`.
   - In `frontend/app.py` Section F, segregated the benchmark view into two distinct subsections:
     - **Table 1: Independently Audited Baselines (Task 09 Source of Truth)**
     - **Table 2: Experimental Research Prototype (Task 10 Investigation)**
   - Added an explicit disclaimer detailing the post-hoc threshold selection caveat.
4. **Confidence Calibration Disclaimers:**
   - Labeled confidence metric as `Confidence (Pmax) *` with an explicit footnote: `* Confidence indicates uncalibrated maximum ensemble vote / neighbor consensus proportion (P_max). It does not represent a calibrated probability of correctness.`
   - Added an informational callout warning users that individual predictions can exhibit elevated errors ($> 40\text{ units}$) despite high confidence.
5. **Warning Suppression & Type Alignment:**
   - Aligned array types in `HybridLocalizer.predict()` and `PredictionService.predict()`, eliminating all scikit-learn `UserWarning` messages.
6. **Regenerated Visual Artifacts:**
   - Updated `experiments/dashboard/generate_dashboard_visuals.py` and regenerated all plots in `experiments/dashboard/` and the conversation artifacts directory.

---

## 4. Actual Model Configurations

| Model | Component Type | Hyperparameters & Configuration | Feature Matrix | Artifact Storage |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest Baseline** | 5-Head Ensemble | `n_estimators=200`, `max_features='sqrt'`, `random_state=42`, `n_jobs=-1` | Variant B (1,058 features: 520 RSSI, 520 visibility, 18 unscaled summary stats) | `experiments/random_forest/models/` (561.1 MB) |
| **k-NN Baseline** | 5-Head Metric Regression | **$k=7$**, `metric='cosine'`, `weights='distance'`, `p=2`, `algorithm='auto'` | Variant B (1,058 features with training-fitted `SummaryFeatureScaler` min-max scaled to $[0, 1]$) | `experiments/knn/models/` (17.7 MB) |
| **Two-Stage Hybrid Localizer** | Hierarchical Router | Stage 1: RF Classifier; Stage 2: 13 Local k-NN Regressors ($k=7$); Fallback: Global k-NN; **$\tau=0.70$ (post-hoc)** | Variant B (1,058 features with fitted summary scaler) | `experiments/hybrid/models/` (578.8 MB) |

---

## 5. Benchmark Values Before and After Correction

All baseline values now strictly match the independently verified Task 09 Benchmark Report (`experiments/model_benchmark/reports/model_benchmark_report.md`):

| Evaluation Dimension | Metric | Prior Dashboard (Unverified) | Corrected Dashboard (Task 09 Audit) | Status / Discrepancy Cause |
| :--- | :--- | :---: | :---: | :--- |
| **Random Forest** | Building Acc (%) | 99.82% | **99.82%** | Exact match |
| | Floor Acc (%) | 89.47% | **88.66%** | Corrected (-0.81% prior discrepancy) |
| | Building-Floor Acc (%) | 89.38% | **88.48%** | Corrected (-0.90% prior discrepancy) |
| | Mean 2D Error | 10.97 units | **11.53 units** | Corrected (+0.56 units prior discrepancy) |
| | Median 2D Error | 8.01 units | **8.65 units** | Corrected (+0.64 units prior discrepancy) |
| | 95th Percentile Error | 28.32 units | **32.74 units** | Corrected (+4.42 units prior discrepancy) |
| **k-NN ($k=7$)** | Building Acc (%) | 99.82% | **99.91%** | Corrected (+0.09% prior discrepancy) |
| | Floor Acc (%) | 88.57% | **84.43%** | Corrected (-4.14% prior discrepancy) |
| | Building-Floor Acc (%) | 88.48% | **84.43%** | Corrected (-4.05% prior discrepancy) |
| | Mean 2D Error | 9.38 units | **10.05 units** | Corrected (+0.67 units prior discrepancy) |
| | Median 2D Error | 6.31 units | **7.33 units** | Corrected (+1.02 units prior discrepancy) |
| | 95th Percentile Error | 24.11 units | **29.06 units** | Corrected (+4.95 units prior discrepancy) |
| **Hybrid (Experimental)** | Building Acc (%) | 99.82% | **99.82%** | Exact match |
| | Floor Acc (%) | 89.47% | **88.66%** | Corrected (-0.81% prior discrepancy) |
| | Building-Floor Acc (%) | 89.38% | **88.48%** | Corrected (-0.90% prior discrepancy) |
| | Mean 2D Error | 8.44 units | **9.98 units** | Corrected (+1.54 units prior discrepancy) |
| | Median 2D Error | 5.68 units | **7.29 units** | Corrected (+1.61 units prior discrepancy) |
| | 95th Percentile Error | 22.84 units | **29.26 units** | Corrected (+6.42 units prior discrepancy) |

---

## 6. Sample Live Prediction Verification

Live inference was executed on the four designated audit validation fingerprints across all three models:

| Sample Index | Ground Truth | Model Evaluated | Predicted Building & Floor | Predicted Coordinates $(X, Y)$ | 2D Euclidean Error | Heuristic Confidence ($P_{\max}$) | Spatial Uncertainty ($\sigma_{\text{disp}}$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample 0** | Building 1, Floor 1<br>$(-7515.92, 4864889.66)$ | **Random Forest**<br>**k-NN ($k=7$)**<br>**Hybrid** | B1, F3 (Δ 2 Flr)<br>B1, F3 (Δ 2 Flr)<br>B1, F3 (Δ 2 Flr) | $(-7505.32, 4864889.57)$<br>$(-7452.11, 4864871.49)$<br>$(-7517.85, 4864932.05)$ | **10.601 units**<br>**66.342 units**<br>**42.434 units** | 82.60%<br>57.14%<br>82.60% | ± 6.48 units<br>± 114.56 units<br>± 35.73 units |
| **Sample 150** | Building 0, Floor 0<br>$(-7618.42, 4864974.84)$ | **Random Forest**<br>**k-NN ($k=7$)**<br>**Hybrid** | B0, F0 (MATCH)<br>B0, F0 (MATCH)<br>B0, F0 (MATCH) | $(-7613.00, 4864981.79)$<br>$(-7618.72, 4864980.69)$<br>$(-7618.72, 4864981.16)$ | **8.816 units**<br>**5.860 units**<br>**6.321 units** | 97.50%<br>100.00%<br>97.50% | ± 3.50 units<br>± 9.53 units<br>± 10.42 units |
| **Sample 450** | Building 0, Floor 2<br>$(-7651.83, 4864933.71)$ | **Random Forest**<br>**k-NN ($k=7$)**<br>**Hybrid** | B0, F2 (MATCH)<br>B0, F2 (MATCH)<br>B0, F2 (MATCH) | $(-7652.24, 4864934.69)$<br>$(-7643.95, 4864929.54)$<br>$(-7643.95, 4864929.54)$ | **1.052 units**<br>**8.924 units**<br>**8.924 units** | 86.00%<br>100.00%<br>86.00% | ± 5.80 units<br>± 0.00 units<br>± 0.77 units |
| **Sample 948** | Building 2, Floor 0<br>$(-7396.55, 4864813.56)$ | **Random Forest**<br>**k-NN ($k=7$)**<br>**Hybrid** | B2, F0 (MATCH)<br>B2, F0 (MATCH)<br>B2, F0 (MATCH) | $(-7357.59, 4864784.95)$<br>$(-7344.23, 4864770.16)$<br>$(-7344.23, 4864770.16)$ | **48.336 units**<br>**67.976 units**<br>**67.976 units** | 80.50%<br>100.00%<br>80.50% | ± 6.90 units<br>± 28.92 units<br>± 25.45 units |

### Key Diagnostic Observations from Sample Testing:
1. **Vertical RF Leakage (Sample 0):** Only 1 access point was detected in this sample (`WAP011` at $-91\text{ dBm}$). Due to sparse coverage and vertical RF bleed, all models misclassified Floor 1 as Floor 3. Random Forest maintained a relatively low coordinate error ($10.60\text{ units}$), whereas k-NN interpolated across distant building clusters ($66.34\text{ units}$).
2. **Sub-Meter Tree Precision (Sample 450):** Random Forest achieved exceptional 2D accuracy ($1.052\text{ units}$ error), outperforming k-NN ($8.924\text{ units}$) when distinctive local AP signals align with dominant tree thresholds.
3. **Outlier Error Despite High Confidence (Sample 948):** Although all models correctly classified Building 2 Floor 0 and k-NN reported $100\%$ confidence (with RF at $80.5\%$), the Euclidean error reached $48.34$ to $67.98\text{ units}$. This demonstrates that heuristic vote proportions do **not** imply calibrated spatial certainty.

---

## 7. Verification & Test Suite Execution

All unit tests and project regression tests were executed and passed cleanly:

```bash
# 1. Run dashboard audit test suite
./.venv/bin/python -m unittest tests/test_dashboard.py
```
```
[INFO] Loaded 5 Random Forest models from experiments/random_forest/models (1058 features)
[INFO] Loaded 5 k-NN models from experiments/knn/models (1058 features)
[INFO] Loaded HybridLocalizer from experiments/hybrid/models (13 local subsets)
Ran 11 tests in 3.604s
OK
```

```bash
# 2. Run full IndoorAI regression test suite
./.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```
```
Ran 97 tests in 13.023s
OK
```

---

## 8. Remaining Limitations & Architectural Invariants

1. **Relative Dataset Units:** All spatial coordinates and Euclidean distances are expressed in **dataset coordinate units**, as provided by the UJIIndoorLoc benchmark. They must not be conflated with physical metres.
2. **Single-Surveyor Validation Set:** As established in Task 09, the validation partition was collected by a single surveyor (`USERID 0`), limiting cross-user demographic evaluation.
3. **Uncalibrated Model Confidence:** The displayed confidence percentage represents raw ensemble vote fractions ($P_{\max}$), not calibrated Bayesian posterior probabilities.
4. **Experimental Hybrid Status:** The Two-Stage Hybrid Localizer requires further training-split threshold cross-validation before it can be considered a benchmark baseline.

