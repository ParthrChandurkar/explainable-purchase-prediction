"""Read-only Streamlit dashboard for the saved Phase 1-6 artifacts.

This app never loads the training CSV, fits preprocessing, or retrains a model.
Every number, table, and image comes from files already written under results/.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


st.set_page_config(
    page_title="RetailIQ | Purchase Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #10243e;
        --muted: #607087;
        --navy: #122a4a;
        --cyan: #19a8a5;
        --mint: #93dfc8;
        --amber: #f4ad55;
        --canvas: #f5f8fc;
    }
    .stApp { background: linear-gradient(180deg, #f8fbff 0%, var(--canvas) 52%, #f8fbff 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #10243e 0%, #18395e 100%); }
    [data-testid="stSidebar"] * { color: #f4f8ff; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #d8e5f3; }
    .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
    .hero {
        position: relative; overflow: hidden; border-radius: 24px; padding: 2.2rem 2.5rem;
        color: white; margin-bottom: 1.15rem;
        background: radial-gradient(circle at 90% 10%, rgba(147,223,200,.45), transparent 28%),
                    linear-gradient(120deg, #10243e 0%, #17476b 58%, #138b8b 100%);
        box-shadow: 0 18px 50px rgba(19, 52, 84, .18);
    }
    .hero h1 { margin: .35rem 0 .4rem; font-size: clamp(2.2rem, 5vw, 4.15rem); line-height: .98; letter-spacing: -.045em; }
    .hero p { margin: 0; max-width: 760px; color: #e6f3f6; font-size: 1.05rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .18em; font-weight: 750; font-size: .73rem; color: #a7f0db; }
    .status-strip { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.3rem; }
    .status-chip { border: 1px solid rgba(255,255,255,.28); background: rgba(255,255,255,.10); border-radius: 999px; padding: .38rem .78rem; font-size: .78rem; }
    .section-title { margin: 1.3rem 0 .2rem; color: var(--ink); font-size: 1.55rem; font-weight: 780; letter-spacing: -.02em; }
    .section-copy { color: var(--muted); margin-bottom: 1rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.92); border: 1px solid #dfe9f3; padding: 1rem 1.1rem; border-radius: 16px; box-shadow: 0 7px 24px rgba(19,52,84,.06); }
    div[data-testid="stMetricLabel"] { color: #63758b; }
    div[data-testid="stMetricValue"] { color: var(--ink); }
    div[data-testid="stTabs"] button { font-weight: 700; }
    div[data-testid="stDataFrame"] { border: 1px solid #dfe8f1; border-radius: 14px; overflow: hidden; }
    .note-card { background: white; border: 1px solid #dfe9f3; border-left: 5px solid var(--cyan); border-radius: 14px; padding: 1rem 1.1rem; color: var(--ink); }
    .warning-card { background: #fff8ee; border: 1px solid #f6d7ad; border-left: 5px solid var(--amber); border-radius: 14px; padding: 1rem 1.1rem; color: #69441c; }
    .footer { text-align: center; color: #7d8a9d; padding-top: 2rem; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load one already-generated result table."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_text(path: Path) -> str:
    """Load one already-generated readable summary."""
    return path.read_text(encoding="utf-8")


def require_files(paths: list[Path]) -> None:
    """Stop with an actionable message when the pipeline has not been run."""
    missing = [path for path in paths if not path.is_file()]
    if missing:
        st.error("The dashboard needs saved pipeline outputs that are currently missing.")
        st.code("\n".join(str(path.relative_to(PROJECT_ROOT)) for path in missing))
        st.info("Run `python main.py` from the project root, then refresh this page.")
        st.stop()


def image_panel(path: Path, caption: str) -> None:
    """Render a saved figure with a consistent caption."""
    st.image(str(path), caption=caption, width="stretch")


def section(title: str, copy: str) -> None:
    """Render a compact section heading."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


required_files = [
    RESULTS_DIR / "dataset_summary.txt",
    RESULTS_DIR / "eda_preprocessing_summary.txt",
    RESULTS_DIR / "modeling_evaluation_summary.txt",
    RESULTS_DIR / "business_insights_summary.txt",
    RESULTS_DIR / "adaptive_experiment_summary.txt",
    TABLES_DIR / "model_comparison.csv",
    TABLES_DIR / "threshold_tuning.csv",
    TABLES_DIR / "business_insights.csv",
    TABLES_DIR / "adaptive_before_after.csv",
    TABLES_DIR / "shap_local_examples.csv",
    PREDICTIONS_DIR / "sample_predictions_with_explanations.csv",
    FIGURES_DIR / "target_class_distribution.png",
    FIGURES_DIR / "correlation_heatmap.png",
    FIGURES_DIR / "shap_global_bar_best_model.png",
    FIGURES_DIR / "shap_global_beeswarm_best_model.png",
    FIGURES_DIR / "adaptive_before_after.png",
]
require_files(required_files)

dataset_summary = load_text(RESULTS_DIR / "dataset_summary.txt")
eda_summary = load_text(RESULTS_DIR / "eda_preprocessing_summary.txt")
model_summary = load_text(RESULTS_DIR / "modeling_evaluation_summary.txt")
business_summary = load_text(RESULTS_DIR / "business_insights_summary.txt")
adaptive_summary = load_text(RESULTS_DIR / "adaptive_experiment_summary.txt")

shape_match = re.search(r"Shape:\s*\((\d+),\s*(\d+)\)", dataset_summary)
duplicate_match = re.search(r"Duplicate row count:\s*(\d+)", dataset_summary)
target_match = re.search(r"CHOSEN TARGET\n-+\n([^:\n]+):", dataset_summary)
row_count = int(shape_match.group(1)) if shape_match else None
column_count = int(shape_match.group(2)) if shape_match else None
duplicate_count = int(duplicate_match.group(1)) if duplicate_match else None
target_name = target_match.group(1).strip() if target_match else "See dataset summary"

model_comparison = load_csv(TABLES_DIR / "model_comparison.csv")
threshold_tuning = load_csv(TABLES_DIR / "threshold_tuning.csv")
business_insights = load_csv(TABLES_DIR / "business_insights.csv")
adaptive_table = load_csv(TABLES_DIR / "adaptive_before_after.csv")
shap_examples = load_csv(TABLES_DIR / "shap_local_examples.csv")
sample_predictions = load_csv(PREDICTIONS_DIR / "sample_predictions_with_explanations.csv")

best_tuned = threshold_tuning.sort_values(["tuned_f1", "roc_auc"], ascending=False).iloc[0]
later_state = adaptive_table.iloc[0]

with st.sidebar:
    st.markdown("## ◆ RetailIQ")
    st.caption("Explainable purchase intelligence")
    st.markdown("---")
    st.markdown("**Dashboard mode**")
    st.success("Saved-results viewer")
    st.caption("No model training · No API calls · No live customer data")
    st.markdown("---")
    st.markdown("**Artifact health**")
    st.markdown(f"✓ {len(list(FIGURES_DIR.glob('*.png')))} saved figures")
    st.markdown(f"✓ {len(list(TABLES_DIR.glob('*.csv')))} result tables")
    st.markdown(f"✓ {len(business_insights):,} insight rows")
    st.markdown("---")
    st.caption("Refresh the browser after running `python main.py` to load regenerated outputs.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Explainable retail analytics</div>
      <h1>RetailIQ</h1>
      <p>A clear view of customer behaviour, purchase predictions, model explanations, retail actions, and the offline adaptive experiment—all from reproducible saved results.</p>
      <div class="status-strip">
        <span class="status-chip">◆ Saved outputs only</span>
        <span class="status-chip">◆ Threshold-aware predictions</span>
        <span class="status-chip">◆ SHAP explanations</span>
        <span class="status-chip">◆ No retraining</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Overview",
        "Customer analytics",
        "Prediction results",
        "SHAP viewer",
        "Business insights",
        "Adaptive experiment",
    ]
)

with tabs[0]:
    section("Dataset at a glance", "The structural facts and project decisions saved during Phase 1.")
    cols = st.columns(4)
    cols[0].metric("Rows", f"{row_count:,}" if row_count is not None else "See summary")
    cols[1].metric("Columns", f"{column_count:,}" if column_count is not None else "See summary")
    cols[2].metric("Chosen target", target_name)
    cols[3].metric("Duplicate rows", f"{duplicate_count:,}" if duplicate_count is not None else "See summary")

    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        image_panel(FIGURES_DIR / "target_class_distribution.png", "Observed target class distribution")
    with right:
        st.markdown("### What this dashboard represents")
        st.markdown(
            '<div class="note-card"><b>End-to-end evidence, not a live scoring service.</b><br><br>'
            "The app reads the exact artifacts produced by the six project phases. It does not open the training CSV, "
            "load the model, or generate new predictions.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Pipeline coverage")
        st.markdown(
            "- Data quality and leakage decisions\n"
            "- Customer-behaviour exploration\n"
            "- Threshold-corrected model evaluation\n"
            "- Global and local SHAP explanations\n"
            "- Transparent retail-action rules\n"
            "- Offline adaptive-model experiment"
        )
    with st.expander("Open the complete Phase 1 dataset summary"):
        st.code(dataset_summary, language=None)

with tabs[1]:
    section("Customer analytics & EDA", "Saved Phase 2 views of behaviour, class balance, association, encoding, and outliers.")
    image_panel(FIGURES_DIR / "correlation_heatmap.png", "Numeric association heatmap")

    st.markdown("### Behaviour versus purchase outcome")
    behaviour_paths = sorted(FIGURES_DIR.glob("behaviour_*_vs_target.png"))
    for start in range(0, len(behaviour_paths), 2):
        pair = st.columns(2, gap="large")
        for column, path in zip(pair, behaviour_paths[start : start + 2]):
            with column:
                label = path.stem.removeprefix("behaviour_").removesuffix("_vs_target").replace("_", " ").title()
                image_panel(path, label)

    with st.expander("Cleaning and feature decisions", expanded=False):
        decision_tabs = st.tabs(["Missing values", "Encoding", "Outliers"])
        with decision_tabs[0]:
            st.dataframe(load_csv(TABLES_DIR / "missing_value_decisions.csv"), hide_index=True, width="stretch")
        with decision_tabs[1]:
            st.dataframe(load_csv(TABLES_DIR / "encoding_decisions.csv"), hide_index=True, width="stretch")
        with decision_tabs[2]:
            st.dataframe(load_csv(TABLES_DIR / "outlier_report.csv"), hide_index=True, width="stretch")

with tabs[2]:
    section("Purchase prediction results", "Held-out metrics, tuned cutoffs, confusion matrices, and explained sample predictions.")
    metrics = st.columns(4)
    metrics[0].metric("Selected model", str(best_tuned["model"]))
    metrics[1].metric("Tuned threshold", f"{best_tuned['tuned_threshold']:.4f}")
    metrics[2].metric("Tuned F1", f"{best_tuned['tuned_f1']:.4f}")
    metrics[3].metric("ROC-AUC", f"{best_tuned['roc_auc']:.4f}")

    compare_tabs = st.tabs(["Tuned comparison", "Default comparison", "Confusion matrices"])
    with compare_tabs[0]:
        tuned_columns = [
            "model", "tuned_threshold", "tuned_accuracy", "tuned_precision",
            "tuned_recall", "tuned_f1", "roc_auc",
        ]
        st.dataframe(
            threshold_tuning[tuned_columns].style.format(precision=4),
            hide_index=True,
            width="stretch",
        )
        st.caption("F1 and ROC-AUC remain the primary comparison metrics because the target is imbalanced.")
    with compare_tabs[1]:
        st.dataframe(model_comparison.style.format(precision=4), hide_index=True, width="stretch")
    with compare_tabs[2]:
        matrix_columns = st.columns(2, gap="large")
        for column, model_slug, title in [
            (matrix_columns[0], "logistic_regression", "Logistic Regression"),
            (matrix_columns[1], "random_forest", "Random Forest"),
        ]:
            with column:
                image_panel(FIGURES_DIR / f"confusion_matrix_{model_slug}_tuned.png", f"{title} · tuned threshold")

    st.markdown("### Sample predictions with explanations attached")
    st.dataframe(sample_predictions, hide_index=True, width="stretch", height=250)
    with st.expander("Open the complete modeling summary"):
        st.code(model_summary, language=None)

with tabs[3]:
    section("SHAP explanation viewer", "Move from global model behaviour to three real local test-row explanations.")
    global_cols = st.columns(2, gap="large")
    with global_cols[0]:
        image_panel(FIGURES_DIR / "shap_global_bar_best_model.png", "Global importance · mean absolute SHAP")
    with global_cols[1]:
        image_panel(FIGURES_DIR / "shap_global_beeswarm_best_model.png", "Global direction and magnitude")

    st.markdown("### Inspect a local explanation")
    example_options = shap_examples["example_type"].astype(str).tolist()
    selected_example = st.selectbox(
        "Choose a real test example",
        example_options,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    selected_row = shap_examples.loc[shap_examples["example_type"].eq(selected_example)].iloc[0]
    local_cols = st.columns([1.25, 0.75], gap="large")
    with local_cols[0]:
        local_path = FIGURES_DIR / f"shap_local_{selected_example}.png"
        image_panel(local_path, f"{selected_example.replace('_', ' ').title()} prediction waterfall")
    with local_cols[1]:
        local_metrics = st.columns(2)
        local_metrics[0].metric("Probability", f"{selected_row['probability']:.4f}")
        local_metrics[1].metric("Threshold", f"{selected_row['threshold']:.4f}")
        local_metrics[0].metric("Prediction", int(selected_row["prediction"]))
        local_metrics[1].metric("Actual label", int(selected_row["actual_label"]))

        contribution_rows = []
        for rank in range(1, 4):
            contribution_rows.extend(
                [
                    {
                        "direction": "Positive",
                        "feature": selected_row[f"positive_feature_{rank}"],
                        "SHAP value": selected_row[f"positive_shap_{rank}"],
                    },
                    {
                        "direction": "Negative",
                        "feature": selected_row[f"negative_feature_{rank}"],
                        "SHAP value": selected_row[f"negative_shap_{rank}"],
                    },
                ]
            )
        st.dataframe(pd.DataFrame(contribution_rows).style.format({"SHAP value": "{:.4f}"}), hide_index=True, width="stretch")

with tabs[4]:
    section("Business insights", "Transparent Phase 5 rules layered over saved probabilities and per-row SHAP drivers.")
    action_counts = business_insights["recommended_action"].value_counts().rename_axis("Action").reset_index(name="Rows")
    promotion_mask = business_insights["recommended_action"].str.contains("promotion", case=False, na=False)
    insight_metrics = st.columns(4)
    insight_metrics[0].metric("Held-out rows", f"{len(business_insights):,}")
    insight_metrics[1].metric("Action categories", business_insights["recommended_action"].nunique())
    insight_metrics[2].metric("Promotion candidates", f"{int(promotion_mask.sum()):,}")
    insight_metrics[3].metric("Mean probability", f"{business_insights['predicted_probability'].mean():.3f}")

    chart_col, note_col = st.columns([1.1, 0.9], gap="large")
    with chart_col:
        st.markdown("#### Action mix")
        st.bar_chart(action_counts.set_index("Action"), color="#19a8a5", horizontal=True)
    with note_col:
        caveat_match = re.search(r"At the tuned threshold, held-out precision[^\n]+", business_summary)
        caveat = caveat_match.group(0) if caveat_match else "See the saved Phase 5 summary for the precision caveat."
        st.markdown("#### Responsible use")
        st.markdown(f'<div class="warning-card"><b>Precision caveat</b><br><br>{caveat}</div>', unsafe_allow_html=True)

    st.markdown("### Explore row-level actions")
    actions = sorted(business_insights["recommended_action"].dropna().unique().tolist())
    selected_actions = st.multiselect("Recommended action", actions, default=actions)
    prediction_filter = st.radio("Threshold prediction", ["All", "Positive", "Negative"], horizontal=True)
    filtered = business_insights[business_insights["recommended_action"].isin(selected_actions)].copy()
    if prediction_filter != "All":
        wanted = 1 if prediction_filter == "Positive" else 0
        filtered = filtered.loc[filtered["tuned_threshold_prediction"].eq(wanted)]
    st.caption(f"Showing {len(filtered):,} of {len(business_insights):,} rows")
    st.dataframe(filtered, hide_index=True, width="stretch", height=430)

with tabs[5]:
    section("Adaptive before/after comparison", "An offline chronological-batch experiment with an explicit F1-drop retraining rule.")
    adaptive_metrics = st.columns(4)
    adaptive_metrics[0].metric("Batch evaluation F1", f"{later_state['f1']:.4f}")
    adaptive_metrics[1].metric("Batch ROC-AUC", f"{later_state['roc_auc']:.4f}")
    adaptive_metrics[2].metric("Evaluation rows", f"{int(later_state['evaluation_rows']):,}")
    adaptive_metrics[3].metric("Retraining triggered", "Yes" if bool(later_state["retrain_triggered"]) else "No")

    mode_match = re.search(r"Batch mode:\s*([^\n]+)", adaptive_summary)
    mode = mode_match.group(1).strip() if mode_match else "See saved summary"
    st.markdown(
        f'<div class="note-card"><b>Observed mode:</b> {mode}<br><br>'
        "This saved run used the confirmed visit-date field and a chronological split. The random STAND-IN "
        "fallback was not used. This remains an offline experiment, not production drift monitoring.</div>",
        unsafe_allow_html=True,
    )
    image_panel(FIGURES_DIR / "adaptive_before_after.png", "Observed adaptive decision comparison")
    st.dataframe(adaptive_table.style.format(precision=4), hide_index=True, width="stretch")
    with st.expander("Open the complete adaptive experiment summary"):
        st.code(adaptive_summary, language=None)

st.markdown(
    '<div class="footer">RetailIQ · Read-only dashboard over reproducible Phase 1–6 artifacts</div>',
    unsafe_allow_html=True,
)
