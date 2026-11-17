"""Interactive motion and magnetic sensor time-series visualization component."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import DatasetApiClient


def render_sensor_explorer(client: DatasetApiClient, dataset_id: Optional[str]) -> None:
    """Renders interactive multi-axis time-series visualization for IMU and magnetometer streams."""
    st.markdown("### 🧭 Sensor Data Explorer")

    if not dataset_id:
        st.info("Please select or upload a dataset in the Overview tab to explore sensor observations.")
        return

    # Fetch sensor observations (sample/limit up to 5000 to keep UI responsive)
    try:
        rows = client.get_observations(dataset_id, modality="sensors", limit=5000)
    except Exception as e:
        st.error(f"Failed to fetch sensor observations: {e}")
        return

    if not rows:
        st.info("This dataset does not contain any motion or magnetic sensor observations.")
        return

    df = pd.DataFrame(rows)

    # Convert numeric columns
    for col in ["x", "y", "z", "magnitude", "normalized_epoch_ms", "original_timestamp_ns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Available sensor types
    sensor_types = sorted(df["sensor_type"].dropna().unique().tolist())
    if not sensor_types:
        st.warning("No valid sensor types detected in sensor observations.")
        return

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        selected_sensor = st.selectbox(
            "Select Sensor Stream:",
            options=sensor_types,
            index=0,
            help="Choose between Accelerometer, Gyroscope, Magnetometer, etc."
        )

    sensor_df = df[df["sensor_type"] == selected_sensor].copy()
    if sensor_df.empty:
        st.info(f"No records available for {selected_sensor}.")
        return

    # Sort chronologically by timestamp
    ts_col = "normalized_epoch_ms" if "normalized_epoch_ms" in sensor_df.columns and not sensor_df["normalized_epoch_ms"].isna().all() else "original_timestamp_ns"
    sensor_df = sensor_df.sort_values(by=ts_col).reset_index(drop=True)

    # Relative time in seconds from start
    start_ts = sensor_df[ts_col].iloc[0]
    is_ns = (start_ts > 1_000_000_000_000_000)
    divisor = 1_000_000_000.0 if is_ns else 1000.0
    sensor_df["Time (s)"] = (sensor_df[ts_col] - start_ts) / divisor

    with f_col2:
        axis_options = ["All Axes (X, Y, Z)", "X-Axis Only", "Y-Axis Only", "Z-Axis Only", "Magnitude Only"]
        selected_axis = st.selectbox("Select Axis View:", options=axis_options, index=0)

    with f_col3:
        max_time = float(sensor_df["Time (s)"].max())
        if max_time > 1.0:
            time_range = st.slider(
                "Time Window (seconds):",
                min_value=0.0,
                max_value=max_time,
                value=(0.0, max_time),
                step=0.5
            )
            filtered_df = sensor_df[(sensor_df["Time (s)"] >= time_range[0]) & (sensor_df["Time (s)"] <= time_range[1])]
        else:
            filtered_df = sensor_df

    # Sensor Telemetry Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Stream Samples", len(sensor_df))
    with m2:
        duration_s = max_time
        st.metric("Stream Duration", f"{duration_s:.1f} s")
    with m3:
        mean_hz = len(sensor_df) / duration_s if duration_s > 0 else 0.0
        st.metric("Effective Sampling Rate", f"{mean_hz:.1f} Hz")
    with m4:
        first_row = sensor_df.iloc[0]
        is_mock = str(first_row.get("is_mock", "")).lower() == "true"
        st.metric("Data Source", "MOCK Telemetry" if is_mock else "REAL Hardware")

    st.markdown("---")

    # Time-Series Interactive Plotly Chart
    st.markdown(f"#### 📈 {selected_sensor} Time-Series Signals")

    fig = go.Figure()

    if selected_axis in ("All Axes (X, Y, Z)", "X-Axis Only") and "x" in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df["Time (s)"],
            y=filtered_df["x"],
            mode="lines",
            name="X Axis",
            line=dict(color="#EF4444", width=1.5)
        ))
    if selected_axis in ("All Axes (X, Y, Z)", "Y-Axis Only") and "y" in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df["Time (s)"],
            y=filtered_df["y"],
            mode="lines",
            name="Y Axis",
            line=dict(color="#10B981", width=1.5)
        ))
    if selected_axis in ("All Axes (X, Y, Z)", "Z-Axis Only") and "z" in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df["Time (s)"],
            y=filtered_df["z"],
            mode="lines",
            name="Z Axis",
            line=dict(color="#3B82F6", width=1.5)
        ))
    if selected_axis in ("Magnitude Only", "All Axes (X, Y, Z)") and "magnitude" in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df["Time (s)"],
            y=filtered_df["magnitude"],
            mode="lines",
            name="Magnitude",
            line=dict(color="#8B5CF6", width=2, dash="dash")
        ))

    unit_label = "m/s²" if "ACCEL" in selected_sensor else ("rad/s" if "GYRO" in selected_sensor else "µT")
    fig.update_layout(
        title=f"{selected_sensor} Signals ({unit_label}) vs Time",
        xaxis_title="Time from Stream Start (seconds)",
        yaxis_title=f"Amplitude ({unit_label})",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Check for gaps and missing values
    missing_axes = filtered_df[["x", "y", "z"]].isna().sum().sum()
    if missing_axes > 0:
        st.warning(f"Detected {missing_axes} missing/NaN coordinate values in selected window. Missing readings are not silently interpolated.")

    st.caption(
        "⚠️ Kinematic Notice: Accelerometer and gyroscope readings measure device inertia, acceleration, and angular velocity. "
        "They do not directly provide absolute indoor metric coordinates without sensor fusion filtering (PDR/EKF)."
    )
