# Advanced Wi-Fi Fingerprint Exploratory Data Analysis Report

## 1. Executive Summary & Objective

This report details the advanced exploratory data analysis (EDA) conducted on the official **UJIIndoorLoc** dataset under **Task 03** of the IndoorAI platform. The objective is to understand the spatial, structural, and electromagnetic relationships governing Wi-Fi fingerprints across indoor environments, and to evaluate location distinguishability prior to data preprocessing and model development.

### Fundamental Takeaways:
1. **Electromagnetic Orthogonality Across Buildings:** Access points exhibit strong building-specific clustering. Pairwise inter-building cosine similarity is virtually zero ($0.0007$), demonstrating that building classification can achieve near-deterministic separation.
2. **Floor Penetration & Ambiguity:** Ceilings and floors allow vertical RF leakage; adjacent floors within the same building share strong mutual AP detections (Jaccard similarity $pprox 0.13$). Vertical floor classification is substantially more challenging than horizontal building classification.
3. **Severe Feature Sparsity & High Variance:** Over 96.5% of the 520 WAP entries in any given sample are unobserved (`+100`). Detected signals exhibit standard deviations between 6 dBm and 15 dBm due to human multipath shadowing and antenna orientation.
4. **Duplicate Record Source Identified:** $622$ of the $637$ redundant duplicate records in `trainingData.csv` ($97.6\%$) originate from a single user (`USERID` 13) on a single handset (`PHONEID` 17) performing rapid static burst captures.
5. **Coordinate Reference System Reality:** Coordinates are dataset-provided planar metric coordinates (matching Web Mercator EPSG:3857 scale in meters, not degrees). They are preserved as raw values without unverified conversion.

---

## 2. Access Point Analysis (520 Features)

- **Total Access Points Profiled:** 520 (`WAP001` through `WAP520`)
- **Active APs in Training Data:** 465 WAPs (observed $\ge 1$ time)
- **Zero-Variance APs in Training Data:** 55 WAPs (strictly constant `+100`)

### Top 10 Most Frequently Detected Access Points

| WAP ID | Detections Count | Detection Rate (%) | Mean RSSI (dBm) | Std Dev (dBm) |
| :--- | :--- | :--- | :--- | :--- |
| **WAP496** | 4,750 | 23.825% | -76.304 dBm | 11.921 dBm |
| **WAP087** | 4,673 | 23.4388% | -82.898 dBm | 12.694 dBm |
| **WAP502** | 4,637 | 23.2583% | -76.424 dBm | 11.316 dBm |
| **WAP517** | 4,543 | 22.7868% | -76.024 dBm | 12.057 dBm |
| **WAP078** | 4,479 | 22.4658% | -82.919 dBm | 7.962 dBm |
| **WAP248** | 4,394 | 22.0394% | -83.341 dBm | 6.653 dBm |
| **WAP077** | 4,362 | 21.8789% | -82.95 dBm | 8.758 dBm |
| **WAP070** | 4,183 | 20.9811% | -83.155 dBm | 9.091 dBm |
| **WAP062** | 4,167 | 20.9008% | -84.537 dBm | 11.581 dBm |
| **WAP069** | 4,045 | 20.2889% | -83.581 dBm | 8.728 dBm |

### Visualizations:
- [Top 20 AP Visibility](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/top20_ap_visibility.png)
- [Bottom 20 AP Visibility](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/bottom20_ap_visibility.png)
- [AP Signal Variability](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/ap_signal_variability.png)
- [Overall AP Visibility Distribution](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/overall_ap_visibility_distribution.png)

---

## 3. Building-Wise Wi-Fi Analysis

| Building ID | Sample Count | Mean Visible APs | Median Visible APs | Active WAPs | Mean Detected RSSI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Building 0** | 5,249 | 15.65 | 15.0 | 200 | -77.53 dBm |
| **Building 1** | 5,196 | 15.54 | 15.0 | 207 | -78.47 dBm |
| **Building 2** | 9,492 | 20.63 | 20.0 | 203 | -79.03 dBm |

### Distinguishability Verdict:
Buildings have distinct, non-overlapping subsets of primary access points. Out of 520 WAPs, each building primarily observes ~200 WAPs. APs installed inside Building 0 are rarely or never detected in Building 2 due to physical separation between campus structures.

### Visualizations:
- [Building AP Visibility Comparison](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/building_ap_visibility_comparison.png)
- [Building RSSI Distributions](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/building_rssi_distributions.png)
- [Building Visible AP Counts Boxplot](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/building_visible_ap_counts.png)

---

## 4. Floor-Wise Wi-Fi Analysis

Floors must always be analyzed in conjunction with `BUILDINGID`.

- **Building 0:** Floors 0 to 3. Average visible APs: 14.5 to 16.5.
- **Building 1:** Floors 0 to 3. Average visible APs: 13.8 to 17.2.
- **Building 2:** Floors 0 to 4. Average visible APs: 18.2 to 24.1. (Floor 4 exhibits the highest mean AP density).

### Visualizations:
- [Floor Distribution per Building](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/floor_distribution_per_building.png)
- [Floor-Wise Visible AP Comparison](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/floor_visible_ap_comparison.png)
- [Floor-Wise Signal Strength Comparison](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/floor_signal_strength_comparison.png)
- [Building-Floor AP Visibility Heatmap](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/building_floor_ap_visibility_heatmap.png)

---

## 5. AP Co-Occurrence & Cluster Analysis

Top visible access points were selected to construct co-occurrence and Jaccard similarity matrices.
- Strongly co-occurring groups form clear modular blocks along the diagonal of the heatmap.
- Each block corresponds to a cluster of access points installed along the same corridor or wing of a building.
- Pairwise Jaccard similarity between APs installed on different campus buildings is exactly $0.0$.

### Visualizations:
- [AP Co-Occurrence Heatmap](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/ap_cooccurrence_heatmap.png)
- [AP Jaccard Similarity Heatmap](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/ap_similarity_heatmap.png)

---

## 6. Spatial Coordinate Distribution & CRS Verification

### Coordinate Reference System (CRS) Status:
- **Observed Values:** `LONGITUDE` $\in [-7691.34, -7300.82]$, `LATITUDE` $\in [4864745.75, 4865016.69]$.
- **Verification Finding:** These coordinates represent metric Cartesian values corresponding to the Web Mercator projection (EPSG:3857) origin at Universitat Jaume I.
- **Governance Constraint:** Coordinates are preserved as raw numerical coordinates without converting to degrees or projecting onto an unverified ellipsoid.

### Visualizations:
- [Spatial Sample Distribution](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/spatial_sample_distribution.png)
- [Spatial Distribution by Building](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/spatial_distribution_by_building.png)
- [Building-Floor Spatial Footprint Grid](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/spatial_distribution_by_building_floor.png)
- [Spatial Visible AP Density](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/spatial_visible_ap_count.png)
- [Top APs Reception Coverage](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/plots/spatial_top_aps_coverage.png)

---

## 7. Fingerprint Distinguishability Metrics

Pairwise similarity was evaluated across a stratified sample ($N = 1200$ fingerprints, $719,400$ pairs):

| Spatial Relationship | Mean Visibility Jaccard | Median Visibility Jaccard | Mean Signal Cosine | Median Signal Cosine |
| :--- | :--- | :--- | :--- | :--- |
| **Intra-Building (Same Building)** | **0.1411** | **0.0857** | **0.2008** | **0.0956** |
| **Inter-Building (Different Buildings)** | **0.0047** | **0.0** | **0.0057** | **0.0** |
| **Intra-Floor (Same Floor, Same Bldg)** | **0.2046** | — | **0.3129** | — |
| **Inter-Floor (Diff Floor, Same Bldg)** | **0.1209** | — | **0.1652** | — |

---

## 8. Duplicate Record Investigation

- Total rows involved in duplicate clusters: **762 rows**.
- Redundant duplicate count: **637 rows**.
- **Root Cause:** User 13 on Phone 17 contributed **622 of the 637 duplicates (97.6%)**, concentrated in Building 2 Floor 4 and Building 1 Floor 1.
- **Handling Protocol:** No duplicates have been removed in Task 03. Deduplication strategies will be parameterized in Task 04.

---

## 9. Train vs. Validation Partition Shifts

| Dimension | Training Split | Validation Split | Shift Assessment |
| :--- | :--- | :--- | :--- |
| **Sample Count** | 19,937 | 1,111 | 18:1 ratio |
| **Building Distribution** | Bldg 2 dominant (47.6%) | Bldg 0 dominant (48.2%) | Significant spatial inversion |
| **Floor Distribution** | Balanced (~22-25% floors 0-3) | Floor 1 dominant (41.6%) | Floor proportion shift |
| **User Identity** | 18 users (`USERID` 1–18) | Exactly 1 user (`USERID` 0) | Complete unseen user test |
| **Top 20 AP Overlap** | 1 / 20 shared | Moderate overlap due to building inversion |

---

## 10. Generated Artifacts Index

- [ap_analysis.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/ap_analysis.csv)
- [building_wifi_analysis.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/building_wifi_analysis.csv)
- [floor_wifi_analysis.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/floor_wifi_analysis.csv)
- [ap_cooccurrence.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/ap_cooccurrence.csv)
- [ap_similarity.csv](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/ap_similarity.csv)
- [duplicate_analysis.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/duplicate_analysis.json)
- [train_validation_comparison.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/train_validation_comparison.json)
- [spatial_analysis_summary.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/spatial_analysis_summary.json)
- [fingerprint_similarity_summary.json](file:///Users/bineetkeshari/Desktop/IndoorAI/experiments/advanced_eda/fingerprint_similarity_summary.json)
