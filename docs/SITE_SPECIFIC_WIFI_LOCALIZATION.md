# Site-Specific Wi-Fi Radio Map & Fingerprint Localization (Task 23)

## 1. Overview & Architectural Decoupling

IndoorAI provides two fundamentally distinct Wi-Fi positioning subsystems:

1. **UJIIndoorLoc ML Benchmark Pipeline (Tasks 01–12):**
   - Operates on a fixed academic benchmark schema containing 520 synthetic column features (`WAP001` through `WAP520`) with standardized $-104$ to $0$ dBm values and $100$ as unobserved.
   - Evaluated exclusively on official UJI validation sets.
2. **Site-Specific Wi-Fi Localization Engine (Task 23):**
   - Operates on real-world smartphone survey observations collected by the IndoorAI Android application (Tasks 14 & 17) and validated through the backend ingestion pipeline (Task 18).
   - Handles arbitrary dynamic BSSIDs (e.g. `aa:bb:cc:dd:ee:ff`), multi-band frequencies ($2.4$ GHz, $5$ GHz, $6$ GHz), real physical landmark coordinates $(x, y)$, and versioned site radio maps.
   - **Strict Isolation:** Real BSSIDs are never mapped or forced into UJI benchmark feature columns (`WAP001..WAP520`), and benchmark models are never used for site-specific live positioning.

```
Android Wi-Fi Surveys (Task 14 / 17)
              │
              ▼
Multimodal Dataset Ingestion (Task 18)
              │
              ▼
Multimodal Preprocessing (Task 20)
              │
              ▼
┌────────────────────────────────────────────────────────┐
│ Site-Specific Radio Map Builder (Task 23)              │
│ - BSSID Normalization & RSSI Bounds Gating             │
│ - Multi-Sweep Temporal Grouping                        │
│ - Mean, Median, Std, Visibility Statistics             │
│ - Versioned Radio Map Storage (data/radio_maps/)       │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ Weighted k-NN Localizer   │ │ Probabilistic Localizer   │
│ - Missing AP Penalties    │ │ - Gaussian Log-Likelihood │
│ - Inverse Distance Weight │ │ - Variance Floors         │
│ - Building/Floor Voting   │ │ - Softmax Posterior Score │
│ - Coordinate Interpolation│ │ - Coordinate Interpolation│
└─────────────┬─────────────┘ └─────────────┬─────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
┌────────────────────────────────────────────────────────┐
│ Leakage-Resistant Session Evaluator                    │
│ - Strict Session-Level Train/Test Partition            │
│ - Building / Floor / Location ID Accuracy              │
│ - Mean, Median, P95 Coordinate Errors (meters)         │
│ - Floor Confusion Matrix & Diagnostics                 │
└────────────────────────────────────────────────────────┘
```

---

## 2. Domain Models & Schemas

The engine is encapsulated in `src/wifi_localization/models.py`:

- **`ReferenceLocation`**: A physical surveyed landmark within a site.
  - Fields: `location_id`, `site_id`, `building_id`, `floor_id`, `room_id`, `x_m`, `y_m`, `description`, `is_verified`, `created_at_iso`, `metadata`.
- **`ApObservation`**: Cleaned single AP observation within a scan.
  - Fields: `bssid`, `rssi_dbm`, `ssid`, `frequency_mhz`, `channel`, `channel_band`.
- **`SurveySweep`**: A temporal burst of AP observations captured during a survey session at a reference location.
  - Methods: `to_rssi_dict()` returns `{bssid: rssi_dbm}`.
- **`ApStatistics`**: Radio frequency statistical distribution of an AP at a location.
  - Fields: `bssid`, `observation_count`, `total_sweeps_at_location`, `visibility_rate`, `mean_rssi`, `median_rssi`, `std_rssi`, `min_rssi`, `max_rssi`, `ssid`, `channel_band`.
- **`LocationRadioProfile`**: Aggregated fingerprint profile for a reference location.
  - Fields: `location_id`, `building_id`, `floor_id`, `x_m`, `y_m`, `total_sweeps`, `ap_stats`, `mean_rssi_vector`.
- **`RadioMap` & `RadioMapMetadata`**: Fully compiled and versioned site database.
- **`CandidateMatch`**: Ranked neighbor returned during matching.
  - Fields: `location_id`, `building_id`, `floor_id`, `x_m`, `y_m`, `distance_metric`, `score`, `matched_ap_count`, `total_query_aps`.
- **`LocalizationPrediction`**: Complete prediction output with spatial coordinates, building/floor estimates, confidence score, and warnings.
- **`EvaluationMetrics`**: Comprehensive offline validation metrics under session-level holdout.

---

## 3. Signal Sanitization & Preprocessing

Implemented in `src/wifi_localization/normalization.py`:

1. **BSSID Standardization:**
   - Validated against regular expression `^([0-9a-fA-F]{2}[:-]){5}([0-9a-fA-F]{2})$`.
   - Converted to lowercase, colon-separated format (e.g. `00:1a:2b:3c:4d:5e`).
2. **RSSI Bounds Gating:**
   - Valid indoor range: $[-105.0, -20.0]$ dBm.
   - Out-of-range, positive (e.g. $+100$ dummy values), or non-numeric values are rejected.
3. **Channel Band Classification:**
   - $2400 \le f \le 2500$ MHz $\rightarrow$ `"2.4GHz"`
   - $4900 \le f \le 5900$ MHz $\rightarrow$ `"5GHz"`
   - $5925 \le f \le 7125$ MHz $\rightarrow$ `"6GHz"`
4. **In-Sweep Deduplication:**
   - When dual-band or duplicate scans appear within the same sweep, the observation with the strongest RSSI is retained.

---

## 4. Radio Map Construction & Versioning

Implemented in `src/wifi_localization/radio_map_builder.py` and stored via `src/wifi_localization/storage.py`:

- **Sweep Extraction:** Raw survey records are segmented into distinct sweeps using a temporal boundary (default $500$ ms) and survey location identifiers.
- **Location Aggregation:**
  - Visibility rate: $V_b = \frac{N_{\text{observed}}}{N_{\text{sweeps}}}$.
  - Sample mean: $\mu_b = \frac{1}{k}\sum_{i=1}^k z_i$.
  - Sample standard deviation with Bessel correction: $\sigma_b = \sqrt{\frac{1}{k-1}\sum_{i=1}^k (z_i - \mu_b)^2}$, floored at $\sigma_{\min} = 2.0$ dBm.
- **Storage Layout:**
  - Persisted under `data/radio_maps/<site_id>/v<version>/`:
    - `radio_map.json`: Full radio map (locations, profiles, AP statistics, metadata).
    - `locations.json`: Reference locations dictionary.
    - `metadata.json`: High-level summary for fast indexing.

---

## 5. Fingerprint Matching Algorithms

### A. Weighted k-Nearest Neighbors (`KnnLocalizer`)

Given query $\mathbf{q}$ and reference profile $\mathbf{r}$, the union distance is:

$$D(q, r) = \sqrt{\frac{1}{|\mathcal{A}_q \cup \mathcal{A}_r|} \sum_{b \in \mathcal{A}_q \cup \mathcal{A}_r} \left(\text{RSSI}_q(b) - \text{RSSI}_r(b)\right)^2}$$

Where missing APs are penalized by default floor $P_{\text{missing}} = -105.0$ dBm.

- **Weighting:** $w_i = \frac{1}{D(q, r_i) + \epsilon}$, normalized to $\sum_{i=1}^k \bar{w}_i = 1$.
- **Coordinate Estimation:** $\hat{x} = \sum_{i=1}^k \bar{w}_i x_i, \quad \hat{y} = \sum_{i=1}^k \bar{w}_i y_i$.
- **Building & Floor:** Plurality weighted vote over top-$k$ candidate profiles.

### B. Probabilistic Gaussian Matching (`ProbabilisticLocalizer`)

Models signal distributions at reference location $L$ as Gaussian:

$$\ln p(z_b \mid L) = -\frac{1}{2}\ln(2\pi \sigma_{b, L}^2) - \frac{(z_b - \mu_{b, L})^2}{2\sigma_{b, L}^2}$$

- **Penalties:**
  - Query APs absent at location $L$ incur unobserved log penalty $P_{\text{log\_missing}} = -15.0$.
  - Reference APs with high visibility ($V_b \ge 0.5$) absent in query incur missing penalty proportional to $V_b$.
- **Posterior Score:** Softmax normalization over top candidates $\exp(\mathcal{L}_i - \max \mathcal{L})$ generates uncalibrated relative likelihood weights for coordinate interpolation.

### C. Confidence Estimation & Safeguards

- **Confidence Score:**
  $$\text{conf} = \exp\left(-\frac{D_{\text{top}}}{\text{scale}}\right) \times \min\left(1.0, \frac{M}{\text{min\_matched\_aps}}\right) \in [0.0, 1.0]$$
- Flagged as `is_low_confidence = True` if $\text{conf} < 0.35$ or overlapping AP count $M < 2$.
- Explicit warnings emitted when no APs overlap or confidence falls below threshold.

---

## 6. Leakage-Resistant Offline Evaluation

Implemented in `src/wifi_localization/evaluator.py`:

- **The Leakage Problem:** When multiple scans collected at the same stationary location in a single survey session are randomly split into train and test sets, temporal autocorrelation causes unrealistically low test errors (data leakage).
- **Session-Level Partitioning:** Enforces that all sweeps belonging to a `session_id` are placed **either** in the training partition (used to build the radio map) **or** in the test partition (evaluated against the radio map). Zero session overlap is permitted.
- **Metrics Reported:**
  - Building Accuracy, Floor Accuracy, Building-Floor Exact Match Accuracy.
  - Location ID Top-1, Top-3, and Top-5 Accuracy.
  - Coordinate Error: Mean, Median, 95th Percentile, and Max (meters).
  - Low-confidence prediction rate.
  - Floor classification confusion matrix.

---

## 7. REST API Endpoints

Integrated into FastAPI router `backend/app/api/routes/datasets.py`:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/datasets/{dataset_id}/wifi-survey/locations` | Register/update a reference location |
| `GET` | `/api/datasets/{dataset_id}/wifi-survey/locations` | List reference locations for a site |
| `POST` | `/api/datasets/{dataset_id}/wifi-survey/fingerprints/build` | Compile site radio map from dataset sweeps |
| `GET` | `/api/datasets/{dataset_id}/wifi-survey/radio-map` | Get compiled radio map by site and version |
| `GET` | `/api/datasets/{dataset_id}/wifi-survey/radio-maps` | List compiled radio maps on disk |
| `POST` | `/api/datasets/{dataset_id}/wifi-survey/localize` | Localize a query Wi-Fi scan |
| `POST` | `/api/datasets/{dataset_id}/wifi-survey/evaluate` | Run session-partitioned offline evaluation |

---

## 8. Dashboard Integration

Streamlit Dashboard Section **13. Site-Specific Wi-Fi Localization** (`frontend/components/wifi_localization_view.py`) provides 4 sub-tabs:

1. **Reference Locations & Survey Registry:** Interactive 2D scatter plot of surveyed landmarks, verified status tags, and registration form.
2. **Radio Map Compiler & Signal Profiles:** Settings to compile versioned radio maps, top-15 AP RSSI horizontal bar charts, and visibility rate histograms.
3. **Live Fingerprint Localizer Testbed:** Interactive simulation and manual input testing, KPI cards, candidate matches table, and 2D spatial map with candidate connection lines.
4. **Leakage-Resistant Evaluation & Confusion Matrix:** Session holdout configuration, classification accuracy KPIs, coordinate error percentiles, and floor confusion matrix heatmap.

---

## 9. Test Suite Verification

Comprehensive test suite in `tests/test_wifi_localization.py` covers:

- `TestWifiNormalization`: MAC normalization, RSSI gating, channel band classification, sweep deduplication.
- `TestRadioMapBuilder`: Temporal sweep grouping, multi-sweep mean/median/std aggregation, AP visibility rates.
- `TestLocalizationEngines`: Weighted k-NN exact match, interpolation, missing AP penalty, Gaussian log-likelihood, low-confidence warnings.
- `TestSessionLevelEvaluator`: Strict session separation, data leakage detection exception, coordinate error computation.
- `TestRadioMapStorage`: JSON serialization, saving, loading, listing, deleting.
- `TestWifiSurveyEndpoints`: FastAPI TestClient integration over all 6 endpoints.

**Total Repository Test Suite:** 201 tests passing (`0` failures, `0` errors).
