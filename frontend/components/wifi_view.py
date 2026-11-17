"""Wi-Fi fingerprint and RF signal exploration component for survey datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_wifi_explorer(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders interactive Wi-Fi sweep inspection, AP signals, and RF distribution charts."""
    st.markdown("### 📶 Wi-Fi Fingerprint Explorer")

    if not dataset_id:
        st.info("Please select or upload a dataset in the Overview tab to explore Wi-Fi observations.")
        return

    # Fetch observations
    try:
        rows = client.get_observations(dataset_id, modality="wifi", limit=5000)
    except Exception as e:
        st.error(f"Failed to fetch Wi-Fi observations: {e}")
        return

    if not rows:
        st.info("This dataset does not contain any Wi-Fi observations.")
        return

    df = pd.DataFrame(rows)

    # Group by observation_id
    sweep_ids = df["observation_id"].unique().tolist()
    total_sweeps = len(sweep_ids)
    total_ap_rows = len(df)
    unique_bssids = df["bssid"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Scan Sweeps", total_sweeps)
    with k2:
        st.metric("Total AP Records", total_ap_rows)
    with k3:
        st.metric("Unique BSSIDs", unique_bssids)
    with k4:
        avg_aps = total_ap_rows / total_sweeps if total_sweeps > 0 else 0.0
        st.metric("Avg APs per Sweep", f"{avg_aps:.1f}")

    st.markdown("---")

    # 1. Sweep Selector and Inspector
    st.markdown("#### 📡 Inspect Individual Wi-Fi Sweep")
    selected_sweep_id = st.selectbox(
        "Select Wi-Fi Scan Sweep:",
        options=sweep_ids,
        index=0,
        help="Select a specific scan sweep to view its detected Access Points and signal levels."
    )

    sweep_df = df[df["observation_id"] == selected_sweep_id]
    first_row = sweep_df.iloc[0]

    meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
    with meta_c1:
        st.caption(f"**Sweep ID:** `{selected_sweep_id}`")
        is_fresh = str(first_row.get("is_fresh", "")).lower() == "true"
        fresh_badge = "🟢 Fresh Sweep" if is_fresh else "🟡 Cached Scan"
        st.caption(f"**Status:** {fresh_badge}")
    with meta_c2:
        norm_ts = first_row.get("normalized_epoch_ms") or first_row.get("original_timestamp")
        st.caption(f"**Timestamp:** `{first_row.get('normalized_iso') or norm_ts}`")
        st.caption(f"**Time Source:** `{first_row.get('time_source')}`")
    with meta_c3:
        bldg = first_row.get("building") or "-"
        flr = first_row.get("floor") or "-"
        rm = first_row.get("room") or "-"
        st.caption(f"**Location Label:** Bldg {bldg}, Fl {flr}, Rm {rm}")
        st.caption(f"**Uncertainty:** ±{first_row.get('uncertainty_ms') or '30.0'} ms")
    with meta_c4:
        st.metric("Visible APs in Sweep", len(sweep_df))

    # Table of APs in this sweep
    st.markdown(f"##### Detected Access Points in `{selected_sweep_id}`")
    display_cols = [
        col for col in ["bssid", "ssid", "rssi_dbm", "frequency_mhz", "channel", "channel_band", "is_mock"]
        if col in sweep_df.columns
    ]
    st.dataframe(sweep_df[display_cols], use_container_width=True)

    st.markdown("---")

    # 2. Visualizations
    st.markdown("#### 📊 Wi-Fi Signal Analytics")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("##### RSSI Signal Strength Distribution")
        rssi_series = pd.to_numeric(df["rssi_dbm"], errors="coerce").dropna()
        valid_rssi = rssi_series[(rssi_series >= -115) & (rssi_series <= 0)]
        if not valid_rssi.empty:
            fig_hist = px.histogram(
                valid_rssi,
                nbins=30,
                labels={"value": "RSSI (dBm)"},
                title="Overall RSSI Distribution (dBm)",
                color_discrete_sequence=["#1E40AF"]
            )
            fig_hist.update_layout(showlegend=False, margin=dict(l=20, r=20, t=35, b=20), height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.caption("No numeric RSSI values available.")

    with viz_col2:
        st.markdown("##### Top 15 Most Frequently Detected APs")
        top_bssids = df["bssid"].value_counts().head(15).reset_index()
        top_bssids.columns = ["BSSID", "Observations"]
        if not top_bssids.empty:
            fig_bar = px.bar(
                top_bssids,
                x="Observations",
                y="BSSID",
                orientation="h",
                title="Top Observed BSSIDs",
                color_discrete_sequence=["#0D9488"]
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=35, b=20), height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

    # 3. AP Count per Sweep over Time
    st.markdown("##### AP Count per Scan Sweep")
    sweep_counts = df.groupby("observation_id").size().reset_index(name="AP Count")
    if not sweep_counts.empty:
        fig_timeline = px.line(
            sweep_counts,
            x="observation_id",
            y="AP Count",
            title="AP Visibility per Scan Sweep",
            markers=True,
            color_discrete_sequence=["#4338CA"]
        )
        fig_timeline.update_layout(xaxis_title="Sweep ID", yaxis_title="Visible APs", height=280)
        st.plotly_chart(fig_timeline, use_container_width=True)

    st.caption("⚠️ Authorized survey research data only. No geographic coordinates are inferred from BSSIDs alone.")
