# Wi-Fi Feature Engineering Methodology and Specifications

## 1. Overview and Engineering Objectives

Indoor localization via Wi-Fi fingerprinting relies on comparing Received Signal Strength Indication (RSSI) vectors recorded by mobile devices against known reference positions. While the preprocessed baseline representation established in Task 04 provides **1,040 features** (520 continuous normalized RSSI values and 520 binary Access Point visibility flags), direct utilization of raw RSSI arrays poses several challenges for machine learning estimators:

1. **Extreme Sparsity:** Out of 520 potential Access Points, an individual fingerprint observes an average of only **17.99 APs** ($\sim 3.46\%$ visibility ratio) in the training set and **16.48 APs** in the validation set.
2. **Device and Path Diversity:** RSSI values are sensitive to device antenna gain, phone orientation, battery level, human body shielding, and multipath propagation.
3. **Macro-Level Spatial Blindness:** Individual AP RSSI signals provide local spatial anchors, but lack aggregate physical context (such as how dense the Wi-Fi environment is, the overall signal quality, or the proximity to the nearest transmitter).

To resolve these challenges without introducing information leakage or altering raw measurements, this module introduces **18 deterministic signal summary statistics** per fingerprint, AP-level training moments, and training-only feature selection filtering. Under the default baseline configuration, the final feature matrix expands to **1,058 features** ($1,040 \text{ baseline} + 18 \text{ summary statistics}$).

---

## 2. Engineered Feature Taxonomy and Mathematical Definitions

All per-sample summary features are computed strictly over the subset of visible APs:
$$\mathcal{V} = \{ i \in \{1, \dots, 520\} \mid \text{RSSI}_i > \text{Floor} \}$$
where $\text{Floor} = -105.0\text{ dBm}$ (the unobserved noise floor replacement), and the original dataset sentinel $+100$ is mapped to unobserved. Under zero-visible-AP edge cases ($|\mathcal{V}| = 0$), fallbacks are deterministically assigned.

### 2.1 Access Point Visibility Features (3 features)

* **`visible_ap_count`** ($N_{\text{vis}}$):
  $$N_{\text{vis}} = |\mathcal{V}| = \sum_{i=1}^{520} \mathbb{I}(\text{RSSI}_i > -105.0)$$
  *Physical Reasoning:* The number of visible access points correlates strongly with indoor building geometry, proximity to corridors, and central floor locations versus building perimeters or dead zones.

* **`visible_ap_ratio`** ($R_{\text{vis}}$):
  $$R_{\text{vis}} = \frac{N_{\text{vis}}}{520}$$
  *Physical Reasoning:* Normalized density of the radio environment relative to the total known transmitter infrastructure.

* **`unavailable_ap_count`** ($N_{\text{unavail}}$):
  $$N_{\text{unavail}} = 520 - N_{\text{vis}}$$
  *Physical Reasoning:* Explicit representation of radio occlusion and transmitter absence.

---

### 2.2 Signal Strength Statistical Moments (8 features)

Let $S = \{ \text{RSSI}_i \mid i \in \mathcal{V} \}$ be the multiset of detected signal strengths in dBm.

* **`rssi_mean`** ($\mu_{\text{RSSI}}$):
  $$\mu_{\text{RSSI}} = \begin{cases} \frac{1}{|\mathcal{V}|} \sum_{s \in S} s & \text{if } |\mathcal{V}| > 0 \\ -105.0 & \text{if } |\mathcal{V}| = 0 \end{cases}$$
  *Physical Reasoning:* Overall electromagnetic energy received. Open halls and atriums show higher mean RSSI compared to heavily compartmentalized offices.

* **`rssi_median`** ($\tilde{\mu}_{\text{RSSI}}$):
  $$\tilde{\mu}_{\text{RSSI}} = \begin{cases} \text{Median}(S) & \text{if } |\mathcal{V}| > 0 \\ -105.0 & \text{if } |\mathcal{V}| = 0 \end{cases}$$
  *Physical Reasoning:* Robust central tendency resistant to single-AP anomalous high-power bursts or temporary line-of-sight spikes.

* **`rssi_std`** ($\sigma_{\text{RSSI}}$):
  $$\sigma_{\text{RSSI}} = \begin{cases} \sqrt{\frac{1}{|\mathcal{V}|} \sum_{s \in S} (s - \mu_{\text{RSSI}})^2} & \text{if } |\mathcal{V}| > 1 \\ 0.0 & \text{if } |\mathcal{V}| \le 1 \end{cases}$$
  *Physical Reasoning:* Measure of signal diversity. A low standard deviation suggests the mobile receiver is equidistant from multiple perimeter APs, whereas a high standard deviation indicates immediate proximity to a single AP alongside distant background APs.

* **`rssi_min`** ($s_{\text{min}}$):
  $$s_{\text{min}} = \begin{cases} \min(S) & \text{if } |\mathcal{V}| > 0 \\ -105.0 & \text{if } |\mathcal{V}| = 0 \end{cases}$$
  *Physical Reasoning:* The sensitivity limit of the mobile device receiver in the local environment.

* **`rssi_max`** ($s_{\text{max}}$):
  $$s_{\text{max}} = \begin{cases} \max(S) & \text{if } |\mathcal{V}| > 0 \\ -105.0 & \text{if } |\mathcal{V}| = 0 \end{cases}$$
  *Physical Reasoning:* Equivalent to the strongest observed signal, indicating the closest AP transmitter.

* **`rssi_range`** ($\Delta_{\text{RSSI}}$):
  $$\Delta_{\text{RSSI}} = s_{\text{max}} - s_{\text{min}}$$
  *Physical Reasoning:* Dynamic range of the local radio environment.

* **`rssi_q25`** ($Q_1$) and **`rssi_q75`** ($Q_3$):
  $$Q_1 = \text{Percentile}(S, 25), \quad Q_3 = \text{Percentile}(S, 75)$$
  *Physical Reasoning:* Non-parametric boundary descriptors capturing the lower and upper quartiles of local signal distribution.

---

### 2.3 Ordered Signal Peaks (4 features)

Let $S_{(1)} \ge S_{(2)} \ge \dots \ge S_{(|\mathcal{V}|)}$ denote the sorted order statistics of visible signal strengths. If $|\mathcal{V}| < k$, the missing order statistic defaults to $-105.0\text{ dBm}$.

* **`strongest_rssi_1`** ($S_{(1)}$): Peak signal power.
* **`strongest_rssi_2`** ($S_{(2)}$): Secondary peak power.
* **`strongest_rssi_3`** ($S_{(3)}$): Tertiary peak power.
* **`strongest_rssi_5`** ($S_{(5)}$): 5th strongest signal power.

*Physical Reasoning:* In trilateration and fingerprinting, the 3 to 5 strongest APs provide $>90\%$ of localization information. Tracking their exact power gradient establishes strong spatial constraints on room-level and floor-level coordinates.

---

### 2.4 Signal Quality Distribution Bands (3 features)

APs are categorized into physical propagation bands:

* **`strong_signal_count`** ($N_{\text{strong}}$):
  $$N_{\text{strong}} = \sum_{s \in S} \mathbb{I}(s \ge -60.0\text{ dBm})$$
  *Physical Reasoning:* Line-of-sight (LOS) or near-LOS access points within $<5\text{--}8\text{ meters}$. High counts indicate proximity to dense ceiling-mounted enterprise routers.

* **`medium_signal_count`** ($N_{\text{medium}}$):
  $$N_{\text{medium}} = \sum_{s \in S} \mathbb{I}(-80.0\text{ dBm} \le s < -60.0\text{ dBm})$$
  *Physical Reasoning:* Nominal indoor coverage through 1 to 2 drywall partitions within $10\text{--}25\text{ meters}$.

* **`weak_signal_count`** ($N_{\text{weak}}$):
  $$N_{\text{weak}} = \sum_{s \in S} \mathbb{I}(s < -80.0\text{ dBm})$$
  *Physical Reasoning:* Severely attenuated edge-of-cell signals through reinforced concrete floors or external facades.

---

## 3. Data Leakage Prevention Guarantees

To prevent data leakage during training, validation, and benchmarking:

1. **AP Statistics Isolation:** All access point detection rates, means, standard deviations, and zero-variance status are calculated strictly on `trainingData.csv` (19,937 samples). The validation partition is never used to compute AP statistics.
2. **Scaler Isolation:** The `MinMaxScaler` applied to newly engineered summary statistics is fitted exclusively on training summary values and subsequently applied to transform validation summary statistics:
   $$x_{\text{scaled}} = \frac{x - \min(X_{\text{train}})}{\max(X_{\text{train}}) - \min(X_{\text{train}})}$$
3. **Feature Selection Mask Isolation:** Feature retention masks (e.g. dropping constant columns or filtering by detection count) are evaluated strictly on training features and projected identically onto validation features.
4. **Target and Metadata Exclusion:** Features are checked against a forbidden blacklist (`BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`, etc.). Any detected metadata raises a runtime `ValueError`.

---

## 4. Empirical Summary of Engineered Features (Audited & Verified)

All values below have been independently recalculated and verified directly from the raw dataset and processed Parquet files (see [feature_audit_report.md](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/feature_engineering/audit/feature_audit_report.md)):

| Feature Name | Training Mean | Training Std | Training Min | Training Max | Val Mean | Val Std | Val Min | Val Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `visible_ap_count` | 17.99 | 7.33 | 0.0 | 51.0 | 16.48 | 6.89 | 1.0 | 35.0 |
| `visible_ap_ratio` | 0.0346 | 0.0141 | 0.0 | 0.0981 | 0.0317 | 0.0133 | 0.0019 | 0.0673 |
| `unavailable_ap_count` | 502.01 | 7.33 | 469.0 | 520.0 | 503.52 | 6.89 | 485.0 | 519.0 |
| `rssi_mean` (dBm) | -78.80 | 6.02 | -105.0 | -51.05 | -77.67 | 5.10 | -95.0 | -58.21 |
| `rssi_median` (dBm) | **-81.01** | 6.77 | -105.0 | -42.00 | **-79.63** | 6.08 | -95.0 | -55.00 |
| `rssi_std` (dB) | **10.53** | 3.59 | 0.0 | 35.06 | **10.75** | 2.90 | 0.0 | 19.82 |
| `rssi_min` (dBm) | -92.14 | 4.88 | -105.0 | -57.00 | -90.95 | 3.55 | -97.0 | -72.00 |
| `rssi_max` (dBm) | -57.15 | 8.82 | -105.0 | -30.00 | -56.76 | 8.28 | -90.0 | -39.00 |
| `rssi_range` (dB) | 34.99 | 9.94 | 0.0 | 66.00 | 35.17 | 8.92 | 0.0 | 53.00 |
| `rssi_q25` (dBm) | -86.52 | 5.56 | -105.0 | -57.00 | -85.57 | 4.70 | -95.0 | -65.00 |
| `rssi_q75` (dBm) | -73.28 | 7.21 | -105.0 | -44.00 | -71.63 | 6.37 | -92.0 | -48.00 |
| `strongest_rssi_1` (dBm) | -57.15 | 8.82 | -105.0 | -30.00 | -56.76 | 8.28 | -90.0 | -39.00 |
| `strongest_rssi_2` (dBm) | -59.34 | 8.16 | -105.0 | -34.00 | -58.50 | 7.74 | -91.0 | -43.00 |
| `strongest_rssi_3` (dBm) | -67.94 | 7.82 | -105.0 | -35.00 | -68.10 | 7.37 | -92.0 | -47.00 |
| `strongest_rssi_5` (dBm) | -73.88 | 7.55 | -105.0 | -43.00 | -74.08 | 7.14 | -94.0 | -52.00 |
| `strong_signal_count` | **1.76** | 2.71 | 0.0 | 35.0 | **1.68** | 1.65 | 0.0 | 18.0 |
| `medium_signal_count` | **6.64** | 4.37 | 0.0 | 38.0 | **6.89** | 4.41 | 0.0 | 24.0 |
| `weak_signal_count` | **9.59** | 5.81 | 0.0 | 38.0 | **7.91** | 4.72 | 0.0 | 30.0 |

---

## 5. Feature Selection Evaluation and Policy Comparison

The training partition was analyzed to identify zero-variance and low-frequency features. Across all 1,058 candidate features:

* **Constant Zero-Variance Features:** **110 features** (55 continuous WAPs that are constantly unobserved across all 19,937 training samples, plus their corresponding 55 binary visibility indicators).
* **Active Access Points:** **465 WAPs** observed at least once in training.
* **Top 5 Most Detected APs (Verified):**
  1. `WAP496`: 4,750 detections ($23.83\%$, mean RSSI: $-76.30\text{ dBm}$)
  2. `WAP087`: 4,673 detections ($23.44\%$, mean RSSI: $-82.90\text{ dBm}$)
  3. `WAP502`: 4,637 detections ($23.26\%$, mean RSSI: $-76.42\text{ dBm}$)
  4. `WAP517`: 4,543 detections ($22.79\%$, mean RSSI: $-76.02\text{ dBm}$)
  5. `WAP078`: 4,479 detections ($22.47\%$, mean RSSI: $-82.92\text{ dBm}$)

### Policy Matrix

| Policy Identifier | Retained Features | Dropped Features | Retained % | Description |
| :--- | :--- | :--- | :--- | :--- |
| `all` (Default) | **1,058** | 0 | 100.00% | Preserves all features for tree models and schema consistency. |
| `remove_constant` | **948** | 110 | 89.60% | Drops the 110 zero-variance features on training data. |
| `min_detection_count_5` | **848** | 210 | 80.15% | Drops APs with $< 5$ training observations. |
| `min_detection_count_10` | **802** | 256 | 75.80% | Drops APs with $< 10$ training observations. |
| `min_detection_count_20` | **752** | 306 | 71.08% | Drops APs with $< 20$ training observations. |
| `min_detection_count_50` | **686** | 372 | 64.84% | Drops APs with $< 50$ training observations. |
| `variance_threshold_0.0001`| **895** | 163 | 84.59% | Drops features with empirical variance $\le 10^{-4}$. |
| `variance_threshold_0.001` | **732** | 326 | 69.19% | Drops features with empirical variance $\le 10^{-3}$. |
| `variance_threshold_0.01` | **401** | 657 | 37.90% | Drops features with empirical variance $\le 10^{-2}$. |

---

## 6. Execution Instructions and Downstream Integration

### Execution CLI

```bash
# Run the complete feature engineering pipeline
python -m src.features.feature_pipeline

# Run with custom configuration
python -m src.features.feature_pipeline --config configs/feature_config.yaml

# Run automated unit and integration tests
python -m unittest tests/test_feature_engineering.py
```

### Downstream Model Integration Guidelines

1. **Parquet Consumption:** Downstream estimators (e.g. Random Forest, LightGBM, MLP) should directly load:
   * `data/processed/ujiindoorloc/feature_engineered/X_train_engineered.parquet`
   * `data/processed/ujiindoorloc/feature_engineered/X_validation_engineered.parquet`
   * Targets from `data/processed/ujiindoorloc/y_train.parquet` and `y_validation.parquet`.
2. **Schema Invariant:** The feature names and exact column order are cataloged in `data/processed/ujiindoorloc/feature_engineered/feature_names.json`.
3. **Model Choice Considerations:**
   * **Tree Models (Random Forest, XGBoost, LightGBM):** Use `strategy: all` (1,058 features). Decision trees naturally split on high-importance APs and summary features, ignoring uninformative constant columns.
   * **k-Nearest Neighbors (k-NN) / Multi-Layer Perceptron (MLP):** If distance metrics (Euclidean, Manhattan) are computed across all dimensions, removing the 110 constant features (`strategy: remove_constant`, 948 features) eliminates unneeded metric computations.

