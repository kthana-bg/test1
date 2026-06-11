import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.model_loader import load_all_results


# Color palette for the six models
EYE_COLORS = ["#3498db", "#2ecc71", "#9b59b6"]
POSTURE_COLORS = ["#e74c3c", "#f39c12", "#1abc9c"]

EYE_MODELS = ["Custom CNN", "MobileNetV2", "EfficientNetB0"]
POSTURE_MODELS = [
    "Custom LSTM/DNN",
    "MediaPipe Pose (Rule-Based)",
    "YOLOv8-Pose / MoveNet DNN",
]


def build_accuracy_chart(models: list, results: dict, colors: list, title: str):
    acc_values = [results.get(m, {}).get("accuracy", 0) * 100 for m in models]
    f1_values = [results.get(m, {}).get("f1_score", 0) * 100 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy (%)",
        x=models,
        y=acc_values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in acc_values],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="F1-Score (%)",
        x=models,
        y=f1_values,
        marker_color=[
            "rgba(52,152,219,0.53)",
            "rgba(46,204,113,0.53)",
            "rgba(155,89,182,0.53)"
        ],
        text=[f"{v:.1f}%" for v in f1_values],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 105]),
        xaxis_title="Model",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#1e2130",
        paper_bgcolor="#1e2130",
        font=dict(color="#e0e0e0"),
        margin=dict(t=60, b=40, l=40, r=20),
        height=380,
    )
    return fig


def build_latency_chart(models: list, results: dict, colors: list, title: str):
    latencies = [results.get(m, {}).get("latency_ms", 0) for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=latencies,
        y=models,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f} ms" for v in latencies],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Latency (ms)"),
        yaxis_title="Model",
        plot_bgcolor="#1e2130",
        paper_bgcolor="#1e2130",
        font=dict(color="#e0e0e0"),
        margin=dict(t=60, b=40, l=150, r=60),
        height=280,
    )
    return fig


def build_radar_chart(models: list, results: dict, colors: list, title: str):
    categories = ["Accuracy", "F1-Score", "Speed (inv latency)"]

    # Normalize latency to a 0-100 speed score (higher = faster)
    latencies = [results.get(m, {}).get("latency_ms", 10) for m in models]
    max_lat = max(latencies) if max(latencies) > 0 else 1
    speed_scores = [(1 - (lat / max_lat)) * 100 for lat in latencies]

    fig = go.Figure()
    for i, model in enumerate(models):
        r = results.get(model, {})
        values = [
            r.get("accuracy", 0) * 100,
            r.get("f1_score", 0) * 100,
            speed_scores[i],
        ]
        values.append(values[0])  # close the polygon
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill="toself",
            name=model,
            line_color=colors[i],
            fillcolor=colors[i],
        ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            bgcolor="#1e2130",
        ),
        paper_bgcolor="#1e2130",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=60, b=80),
        height=380,
    )
    return fig


def render_model_selector(
    current_eye_model: str,
    current_posture_model: str,
    eye_models_loaded: dict,
    posture_models_loaded: dict,
) -> tuple[str, str]:
    st.subheader("Active Model Selection")
    st.caption(
        "Switch the model used in the Live Monitoring tab. "
        "Changes take effect on the next processed frame."
    )

    col_eye, col_posture = st.columns(2)

    with col_eye:
        st.markdown("**Eye Strain Model**")
        selected_eye = st.radio(
            "Eye model",
            options=EYE_MODELS,
            index=EYE_MODELS.index(current_eye_model)
            if current_eye_model in EYE_MODELS else 0,
            key="eye_model_selector",
            label_visibility="collapsed",
        )
        loaded = eye_models_loaded.get(selected_eye) is not None
        st.caption(
            "Model loaded and ready." if loaded
            else "Model file not found. Using EAR rule-based fallback."
        )

    with col_posture:
        st.markdown("**Posture Model**")
        selected_posture = st.radio(
            "Posture model",
            options=POSTURE_MODELS,
            index=POSTURE_MODELS.index(current_posture_model)
            if current_posture_model in POSTURE_MODELS else 1,
            key="posture_model_selector",
            label_visibility="collapsed",
        )
        is_rule = "Rule-Based" in selected_posture
        loaded = is_rule or (posture_models_loaded.get(selected_posture) is not None)
        st.caption(
            "Active (rule-based, no model file needed)." if is_rule
            else ("Model loaded and ready." if loaded
                  else "Model file not found. Using MediaPipe rule-based fallback.")
        )

    return selected_eye, selected_posture


def render_comparison_table(all_results: dict):
    """Render a summary comparison table of all six models."""
    st.subheader("Summary Comparison Table")

    rows = []
    for model_name, r in all_results.items():
        group = "Eye" if model_name in EYE_MODELS else "Posture"
        rows.append({
            "Group": group,
            "Model": model_name,
            "Accuracy (%)": f"{r.get('accuracy', 0) * 100:.1f}",
            "F1-Score (%)": f"{r.get('f1_score', 0) * 100:.1f}",
            "Latency (ms)": f"{r.get('latency_ms', 0):.1f}",
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_comparison_tab(
    eye_models_loaded: dict,
    posture_models_loaded: dict,
    current_eye_model: str,
    current_posture_model: str,
) -> tuple[str, str]:
    """
    Main render function for the Comparative Analysis tab.
    Returns (selected_eye_model_name, selected_posture_model_name).
    """
    st.header("Comparative Model Analysis")

    all_results = load_all_results()

    # Split into eye and posture result dicts
    eye_results = {m: all_results.get(m, {}) for m in EYE_MODELS}
    posture_results = {m: all_results.get(m, {}) for m in POSTURE_MODELS}

    # --- Accuracy Charts ---
    st.subheader("Accuracy and F1-Score Comparison")
    acc_col1, acc_col2 = st.columns(2)

    with acc_col1:
        fig = build_accuracy_chart(EYE_MODELS, eye_results, EYE_COLORS, "Eye Strain Models")
        st.plotly_chart(fig, use_container_width=True)

    with acc_col2:
        fig = build_accuracy_chart(
            POSTURE_MODELS, posture_results, POSTURE_COLORS, "Posture Models"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Latency Charts ---
    st.subheader("Inference Latency")
    lat_col1, lat_col2 = st.columns(2)

    with lat_col1:
        fig = build_latency_chart(
            EYE_MODELS, eye_results, EYE_COLORS, "Eye Model Latency (lower = faster)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with lat_col2:
        fig = build_latency_chart(
            POSTURE_MODELS, posture_results, POSTURE_COLORS, "Posture Model Latency"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Radar Charts ---
    st.subheader("Multi-Metric Radar")
    radar_col1, radar_col2 = st.columns(2)
    with radar_col1:
        fig = build_radar_chart(EYE_MODELS, eye_results, EYE_COLORS, "Eye Model Comparison")
        st.plotly_chart(fig, use_container_width=True)
    with radar_col2:
        fig = build_radar_chart(
            POSTURE_MODELS, posture_results, POSTURE_COLORS, "Posture Model Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Summary Table ---
    render_comparison_table(all_results)

    st.divider()

    # --- Model Selector ---
    selected_eye, selected_posture = render_model_selector(
        current_eye_model,
        current_posture_model,
        eye_models_loaded,
        posture_models_loaded,
    )

    return selected_eye, selected_posture
