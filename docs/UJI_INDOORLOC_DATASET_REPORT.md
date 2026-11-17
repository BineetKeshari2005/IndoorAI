# UJIIndoorLoc Dataset Acquisition & Inspection Report

## 1. Executive Summary

This report documents the acquisition, integrity verification, structural inspection, and exploratory data analysis (EDA) of the **UJIIndoorLoc** Wi-Fi fingerprinting benchmark dataset, completed as part of **Task 02** in the IndoorAI platform roadmap.

The dataset was downloaded directly from the official **UCI Machine Learning Repository** without relying on secondary mirrors or Kaggle copies. All files have been preserved in their original state under `data/raw/ujiindoorloc/` with SHA-256 cryptographic checksums recorded in an immutable provenance manifest.

> [!IMPORTANT]
> **Modality Clarification:** UJIIndoorLoc is exclusively an RF signal (Wi-Fi RSSI) fingerprinting dataset. **It does NOT contain synchronized smartphone sensor sequences (accelerometer, gyroscope, magnetometer) or indoor photographs.** Multimodal training for subsequent IndoorAI phases will integrate complementary data collected via the IndoorAI Android mobile client.

---

## 2. Dataset Provenance and Storage

- **Official Source URL:** [UCI Machine Learning Repository — UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
- **Direct Acquisition Endpoint:** `https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip`
- **Storage Location:** `data/raw/ujiindoorloc/`
- **Acquisition Timestamp:** 2026-11-16T10:08:40 UTC
- **Integrity Verification:**

| File Name | Physical Size (Bytes) | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- |
| `ujiindoorloc.zip` | 1,463,759 | `893512b82dfd7a7c345d84195b1c8019fbca0fa0d7820ce491ce5aa45ec3782f` | Verified Archive |
| `trainingData.csv` | 42,725,381 | `45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168` | Verified Raw File |
| `validationData.csv` | 2,381,949 | `5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0` | Verified Raw File |
| `manifest.json` | 1,324 | — | Metadata Manifest |

---

## 3. Dataset Dimensions & Feature Schema

The dataset provides 21,048 total fingerprint observations across two official splits:

| Metric | Training Set (`trainingData.csv`) | Validation Set (`validationData.csv`) | Total Combined |
| :--- | :--- | :--- | :--- |
| **Row Count (Samples)** | 19,937 | 1,111 | 21,048 |
| **Column Count** | 529 | 529 | 529 |
| **Wi-Fi WAP Features** | 520 (`WAP001` – `WAP520`) | 520 (`WAP001` – `WAP520`) | 520 |
| **Spatial / Context Fields**| 9 | 9 | 9 |
| **Memory Footprint** | ~80.46 MB | ~4.48 MB | ~84.94 MB |
| **Missing Values (NaN/Null)**| **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** |

### Contextual and Spatial Target Fields
The 9 non-WAP columns contain metric coordinates, hierarchical labels, and device metadata:

1. `LONGITUDE`: Metric coordinate in WGS84 projection (UTM-projected EPSG:3857 coordinates: $-7691.33$ to $-7300.82$).
2. `LATITUDE`: Metric coordinate in WGS84 projection ($4864745.75$ to $4865017.37$).
3. `FLOOR`: Integer identifier for the building floor level $\{0, 1, 2, 3, 4\}$.
4. `BUILDINGID`: Categorical identifier for the campus building $\{0, 1, 2\}$.
5. `SPACEID`: Integer identifier of the specific room or corridor segment.
6. `RELATIVEPOSITION`: Location relative to the space ($1 = \text{Inside}$, $2 = \text{Outside / Doorway}$).
7. `USERID`: Anonymized integer ID of the collector ($1\text{ to }18$ in train; $0$ in validation).
8. `PHONEID`: Identifier representing the Android hardware model used during collection ($16$ unique in train; $11$ in validation).
9. `TIMESTAMP`: UNIX epoch timestamp (seconds).

---

## 4. Wi-Fi Access Point Signal Characteristics

### The Sentinel Value 100
In UJIIndoorLoc, unobserved or out-of-range Wi-Fi Access Points are recorded as integer **`+100`**.
- It is physically impossible for an 802.11 receiver to detect a signal at $+100\text{ dBm}$.
- Treating `100` as a continuous numeric RSSI measurement would completely corrupt distance calculations and gradient-based neural networks.
- In our pipeline, genuine signals are strictly $\le 0\text{ dBm}$, with `100` isolated as an unavailable signal sentinel.

### Empirical Signal Statistics (Excluding Sentinel +100)

| Statistic | Training Set | Validation Set |
| :--- | :--- | :--- |
| **Total Valid Signal Observations** | 358,763 | 18,304 |
| **Global Matrix Sparsity** | 96.54% unavailable (`+100`) | 96.83% unavailable (`+100`) |
| **Minimum Observed RSSI** | $-104.0\text{ dBm}$ | $-102.0\text{ dBm}$ |
| **Maximum Observed RSSI** | $0.0\text{ dBm}$ | $-34.0\text{ dBm}$ |
| **Mean Observed RSSI** | $-78.50\text{ dBm}$ | $-77.68\text{ dBm}$ |
| **Median Observed RSSI** | $-82.00\text{ dBm}$ | $-80.00\text{ dBm}$ |
| **Standard Deviation** | $12.62\text{ dBm}$ | $11.97\text{ dBm}$ |

### AP Visibility per Measurement
- **Training Set:** Samples detect an average of **17.99** access points (median: 17, min: 0, max: 51, std: 7.33).
  - Note: A small number of training samples recorded **0 detected APs** (blind measurements).
- **Validation Set:** Samples detect an average of **16.48** access points (median: 15, min: 1, max: 35, std: 6.89).

---

## 5. Dataset Split Distributions & Domain Shifts

### 5.1 Building Distribution

| Building ID | Training Count | Training % | Validation Count | Validation % |
| :--- | :--- | :--- | :--- | :--- |
| **Building 0** | 5,249 | 26.33% | **536** | **48.24%** |
| **Building 1** | 5,196 | 26.06% | 307 | 27.63% |
| **Building 2** | **9,492** | **47.61%** | 268 | 24.12% |

> [!WARNING]
> **Spatial Distribution Shift:** Building 2 accounts for nearly half (47.6%) of the training records, but only 24.1% of validation records. Building 0 dominates the validation set (48.2%). Models evaluated on the validation set will test cross-building generalization rather than identical class proportions.

### 5.2 Floor Distribution

| Floor ID | Training Count | Training % | Validation Count | Validation % | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Floor 0** | 4,369 | 21.91% | 132 | 11.88% | Ground floor |
| **Floor 1** | 5,002 | 25.09% | 462 | 41.58% | First floor |
| **Floor 2** | 4,416 | 22.15% | 306 | 27.54% | Second floor |
| **Floor 3** | 5,048 | 25.32% | 172 | 15.48% | Third floor |
| **Floor 4** | 1,102 | 5.53% | 39 | 3.51% | Exists exclusively in Building 2 |

### 5.3 Collector & Hardware Diversity
- **Training Set:** Collected by **18 distinct human users** (`USERID` 1 to 18) across **16 Android handset models** (`PHONEID` 1 to 24).
- **Validation Set:** Collected by **1 user** (`USERID` = 0) across **11 phone models**.
- **Research Significance:** The presence of User 0 exclusively in the validation split evaluates how well models generalize to an **unseen individual**, simulating real-world deployment where end-users were not part of the calibration team.

---

## 6. Data Quality & Anomaly Findings

A rigorous programmatic audit across all 529 features yielded the following quality insights:

1. **Exact Duplicate Rows:**
   - **Training Set:** Contains **637 exact duplicate rows** (identical WAP signal vector, space, coordinates, and timestamp). These duplicates likely occurred during static rapid-burst fingerprinter captures.
   - **Validation Set:** Contains **0 duplicate rows**.
   - *Recommendation:* Deduplication will be evaluated during Task 04 without prematurely discarding raw files.
2. **Constant Uninformative WAPs:**
   - **Training Set:** 55 WAPs have a constant value of `100` (never detected in any training fingerprint).
   - **Validation Set:** 153 WAPs have a constant value of `100`.
   - **Both Splits:** **0 WAPs are completely unobserved across both splits.** Every single one of the 520 access points is observed at least once in either train or validation.
3. **Signal Value Integrity:**
   - Unexpected signal count (< -104 dBm or > 0 dBm excluding 100): **0 records**.
   - Missing NaN/null values: **0 cells**.
4. **Coordinate Consistency:**
   - All latitude and longitude coordinates fall strictly within the geographical bounding polygon of the Universitat Jaume I campus.

---

## 7. Generated EDA Artifacts & Visualizations

All statistical outputs and high-resolution plots have been generated and saved under [experiments/dataset_eda/](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/):

### Structured Data Tables
- [experiments/dataset_eda/dataset_overview.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/dataset_overview.json): High-level dimensions and metadata.
- [experiments/dataset_eda/column_information.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/column_information.csv): Detailed per-column data types, null counts, and observation counts.
- [experiments/dataset_eda/numerical_statistics.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/numerical_statistics.csv): Min, max, mean, median, std for all features with `100` isolated.
- [experiments/dataset_eda/categorical_statistics.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/categorical_statistics.json): Frequency distributions for Building, Floor, User, and Phone.
- [experiments/dataset_eda/wap_visibility_statistics.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/wap_visibility_statistics.csv): Detection rates and RSSI profiles for all 520 WAPs.
- [experiments/dataset_eda/dataset_split_summary.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/dataset_split_summary.json): Complete split metrics and quality audit log.

### Visualization Plots
- **Building Distribution:** [experiments/dataset_eda/plots/building_distribution.png](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/plots/building_distribution.png)
- **Floor Distribution per Building:** [experiments/dataset_eda/plots/floor_distribution.png](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/plots/floor_distribution.png)
- **Wi-Fi Visibility (APs Detected per Record):** [experiments/dataset_eda/plots/wifi_visibility_distribution.png](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/plots/wifi_visibility_distribution.png)
- **Distribution of Valid RSSI Values ($\le 0\text{ dBm}$):** [experiments/dataset_eda/plots/rssi_distribution_valid_signals.png](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/plots/rssi_distribution_valid_signals.png)
- **Visible APs per Sample Across Buildings:** [experiments/dataset_eda/plots/visible_aps_per_sample.png](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/dataset_eda/plots/visible_aps_per_sample.png)

---

## 8. Limitations & Implications for IndoorAI Development

1. **RF-Only Baseline:** As established, UJIIndoorLoc establishes strong baseline metrics for Wi-Fi positioning, but cannot train vision or IMU models.
2. **Missing AP Representations:** Models must handle the extreme sparsity (~96.6% unavailable WAPs) and avoid treating `+100` as a physical signal.
3. **Temporal Invariance:** Data was collected in 2013 over limited temporal windows, meaning real-world deployments must account for dynamic environmental changes not present in static benchmark splits.
4. **Next Phase:** Task 03 will conduct advanced exploratory modeling analysis and spatial correlation studies prior to data validation (Task 04) and feature engineering (Tasks 05–06).

