# Wi-Fi Feature Selection Analysis Report

## 1. Executive Summary

This report evaluates training-only feature filtering policies for the **1058** total features 
across **19,937** training fingerprints. 
All calculations were executed strictly on the training partition without utilizing validation labels or statistics.

## 2. Policy Comparison Matrix

| Strategy Identifier | Retained Features | Dropped Features | Retained % | Description |
| :--- | :--- | :--- | :--- | :--- |
| `all` | **1,058** | 0 | 100.00% | Preserves all features without filtering (Baseline default). |
| `remove_constant` | **948** | 110 | 89.60% | Drops 110 zero-variance features where all training values are identical. |
| `min_detection_count_5` | **848** | 210 | 80.15% | Retains APs detected in at least 5 training samples (and all summary features). |
| `min_detection_count_10` | **802** | 256 | 75.80% | Retains APs detected in at least 10 training samples (and all summary features). |
| `min_detection_count_20` | **752** | 306 | 71.08% | Retains APs detected in at least 20 training samples (and all summary features). |
| `min_detection_count_50` | **686** | 372 | 64.84% | Retains APs detected in at least 50 training samples (and all summary features). |
| `variance_threshold_0.0001` | **895** | 163 | 84.59% | Retains features with empirical variance > 0.0001 on training data. |
| `variance_threshold_0.001` | **731** | 327 | 69.09% | Retains features with empirical variance > 0.001 on training data. |
| `variance_threshold_0.01` | **402** | 656 | 38.00% | Retains features with empirical variance > 0.01 on training data. |

## 3. Zero-Variance Constant Features

Identified **110** features with zero empirical variance on the training partition.
Under the default configuration (`strategy: all`), these features are preserved to maintain cross-partition schema consistency.

```text
WAP003, WAP004, WAP092, WAP093, WAP094, WAP095, WAP152, WAP158, WAP159, WAP160, WAP215, WAP217, WAP226, WAP227, WAP238, WAP239, WAP240, WAP241, WAP242, WAP243, WAP244, WAP245, WAP246, WAP247, WAP254, WAP293, WAP296, WAP301, WAP303, WAP304...
```

## 4. Recommendation for Downstream Modeling

- **Tree-based Baselines (Random Forest, XGBoost):** Can naturally ignore zero-variance features through split-criterion optimization. Use `strategy: all`.
- **Linear Models & Distance-based Methods (k-NN, SVM, MLP):** Benefit from dropping zero-variance features (`remove_constant` or `min_detection_count_10`) to reduce matrix dimension without losing informative variance.
