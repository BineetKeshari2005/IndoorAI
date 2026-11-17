"""Generate visual evaluation artifacts for IndoorAI Web Dashboard (Task 11).

Produces high-resolution figures replicating the dashboard's 2D coordinate map,
signal strength analysis, and model comparison metrics for documentation and reports.
"""

import os
import sys
sys.path.insert(0, os.getcwd())
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.app.services.prediction_service import get_prediction_service

OUTPUT_DIR = "experiments/dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

service = get_prediction_service()
service.load_all()

# 1. 2D Coordinate Map for Validation Sample #0
sample0 = service.get_validation_sample(0)
pred_rf = service.predict(sample0["features"], model_name="Random Forest")
pred_knn = service.predict(sample0["features"], model_name="k-Nearest Neighbors")
pred_hyb = service.predict(sample0["features"], model_name="Hybrid")

val_targets = service.validation_targets
gt = sample0["ground_truth"]
gt_lon = float(gt["LONGITUDE"])
gt_lat = float(gt["LATITUDE"])

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
colors = {0: "#3B82F6", 1: "#F59E0B", 2: "#8B5CF6"}
labels = {0: "Building 0 Ambient", 1: "Building 1 Ambient", 2: "Building 2 Ambient"}

for b_id in sorted(val_targets["BUILDINGID"].unique()):
    subset = val_targets[val_targets["BUILDINGID"] == b_id]
    ax.scatter(
        subset["LONGITUDE"],
        subset["LATITUDE"],
        c=colors[b_id],
        label=labels[b_id],
        alpha=0.2,
        s=20,
        edgecolors="none",
    )

# Ground truth
ax.scatter(
    gt_lon,
    gt_lat,
    c="#10B981",
    s=180,
    marker="s",
    edgecolors="#064E3B",
    linewidth=2,
    label=f"Ground Truth (B{int(gt['BUILDINGID'])} F{int(gt['FLOOR'])})",
    zorder=6,
)

# RF Pred
ax.scatter(
    pred_rf["pred_longitude"],
    pred_rf["pred_latitude"],
    c="#DC2626",
    s=160,
    marker="o",
    edgecolors="#7F1D1D",
    linewidth=1.5,
    label=f"RF Pred (B{pred_rf['pred_building']} F{pred_rf['pred_floor']})",
    zorder=7,
)

# Hybrid Pred
ax.scatter(
    pred_hyb["pred_longitude"],
    pred_hyb["pred_latitude"],
    c="#2563EB",
    s=220,
    marker="*",
    edgecolors="#1E3A8A",
    linewidth=2,
    label=f"Hybrid Pred (B{pred_hyb['pred_building']} F{pred_hyb['pred_floor']})",
    zorder=8,
)

# Draw error vector for hybrid
ax.plot(
    [gt_lon, pred_hyb["pred_longitude"]],
    [gt_lat, pred_hyb["pred_latitude"]],
    color="#2563EB",
    linestyle="--",
    linewidth=1.5,
    alpha=0.85,
    zorder=5,
    label="Hybrid Error Vector",
)

ax.set_title("IndoorAI 2D Coordinate Map — Validation Sample #0 Multi-Model Localization", fontsize=12, fontweight="bold")
ax.set_xlabel("Dataset Longitude (Coordinate X)", fontsize=10)
ax.set_ylabel("Dataset Latitude (Coordinate Y)", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
plt.tight_layout()
map_path = os.path.join(OUTPUT_DIR, "dashboard_coordinate_map_sample0.png")
fig.savefig(map_path)
plt.close(fig)
print(f"Saved {map_path}")

# 2. Top Signal Strengths for Sample #150 (dense AP reading)
sample150 = service.get_validation_sample(150)
detected = sample150["detected_waps"][:12]
if detected:
    names = [w["wap_id"] for w in detected][::-1]
    vals = [w["rssi_dbm"] for w in detected][::-1]
    bar_colors = ["#10B981" if v >= -60 else ("#F59E0B" if v >= -80 else "#EF4444") for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.barh(names, vals, color=bar_colors, height=0.6)
    ax.set_xlim(-105, -30)
    ax.set_xlabel("Received Signal Strength (dBm)", fontsize=10)
    ax.set_ylabel("Access Point Identifier", fontsize=10)
    ax.set_title(f"IndoorAI Wi-Fi Signal Strength Profile — Validation Sample #150 ({sample150['visible_count']} Visible WAPs)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.2, bar.get_y() + bar.get_height() / 2, f"{width:.0f} dBm", va="center", ha="left", fontsize=8, color="#1F2937")

    plt.tight_layout()
    rssi_path = os.path.join(OUTPUT_DIR, "dashboard_signal_profile_sample150.png")
    fig.savefig(rssi_path)
    plt.close(fig)
    print(f"Saved {rssi_path}")

# 3. Model Benchmark Comparison Chart
bench_df = service.get_benchmark_table()
hyb_df = service.get_experimental_hybrid_table()
combined_df = pd.concat([bench_df, hyb_df], ignore_index=True)

fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=150)
mean_errs = combined_df["Mean 2D Error (units)"].tolist()
med_errs = combined_df["Median 2D Error (units)"].tolist()

x = np.arange(len(combined_df))
width = 0.35

rects1 = ax.bar(x - width/2, mean_errs, width, label="Mean 2D Error (units)", color="#3B82F6")
rects2 = ax.bar(x + width/2, med_errs, width, label="Median 2D Error (units)", color="#10B981")

ax.set_ylabel("Coordinate Error (Dataset Units)", fontsize=10)
ax.set_title("IndoorAI Model Benchmarking — Audited 2D Coordinate Error Comparison", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([
    "Random Forest\n(Task 07 Baseline)",
    "k-NN (k=7)\n(Task 08 Baseline)",
    "Two-Stage Hybrid\n(Task 10 Experimental)"
], fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.5)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
bench_path = os.path.join(OUTPUT_DIR, "dashboard_model_benchmarks.png")
fig.savefig(bench_path)
plt.close(fig)
print(f"Saved {bench_path}")
print("ALL VISUAL ARTIFACTS GENERATED SUCCESSFULLY!")
