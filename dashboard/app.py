"""RetailIQ dashboard for saved artifacts and new customer scoring.

The dashboard never retrains a model. It can load the saved best-model artifact
to score user-entered rows or an uploaded CSV using the fitted preprocessing.
All analysis figures and phase summaries still come from saved result files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Streamlit starts this file from dashboard/, while the saved model refers
    # to the project-level src package when it is unpickled.
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


st.set_page_config(
    page_title="RetailIQ | Purchase Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load one already-generated result table."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_text(path: Path) -> str:
    """Load one already-generated readable summary."""
    return path.read_text(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def load_best_model():
    """Load the already-fitted threshold-aware model without retraining."""
    return joblib.load(MODEL_PATH)


def live_prediction_schema(model) -> dict[str, dict[str, object]]:
    """Read required user inputs from the fitted preprocessing pipeline."""
    preprocessor = model.estimator.named_steps["preprocessor"]
    schema: dict[str, dict[str, object]] = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or not isinstance(columns, list):
            continue
        if name == "numeric":
            medians = transformer.named_steps["imputer"].statistics_
            for column, median in zip(columns, medians):
                schema[column] = {"kind": "numeric", "default": float(median)}
        elif name == "nominal":
            categories = transformer.named_steps["one_hot"].categories_
            for column, choices in zip(columns, categories):
                schema[column] = {
                    "kind": "choice",
                    "choices": [str(value) for value in choices],
                    "default": str(choices[0]),
                }
        elif name == "frequency":
            defaults = transformer.named_steps["imputer"].statistics_
            for column, default in zip(columns, defaults):
                schema[column] = {"kind": "text", "default": str(default)}
        elif name == "ordinal":
            categories = transformer.named_steps["ordinal"].categories_
            for column, choices in zip(columns, categories):
                schema[column] = {
                    "kind": "choice",
                    "choices": [str(value) for value in choices],
                    "default": str(choices[0]),
                }
    return schema


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
    """Render a theme-native section heading."""
    st.subheader(title)
    st.caption(copy)


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
    MODEL_PATH,
]
require_files(required_files)

best_model = load_best_model()
prediction_schema = live_prediction_schema(best_model)

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
    st.success("Saved analysis + live scoring")
    st.caption("No retraining · No API calls · No data is stored")
    st.markdown("---")
    st.markdown("**Navigate**")
    view = st.radio(
        "Dashboard section",
        [
            "Overview",
            "Customer analytics",
            "Prediction results",
            "Live prediction",
            "SHAP viewer",
            "Business insights",
            "Adaptive experiment",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Artifact health**")
    st.markdown(f"✓ {len(list(FIGURES_DIR.glob('*.png')))} saved figures")
    st.markdown(f"✓ {len(list(TABLES_DIR.glob('*.csv')))} result tables")
    st.markdown(f"✓ {len(business_insights):,} insight rows")
    st.markdown("---")
    st.caption("Refresh the browser after running `python main.py` to load regenerated outputs.")

st.title("RetailIQ")
st.caption("Explainable retail analytics")
st.write(
    "Explore customer behaviour, purchase predictions, model explanations, retail actions, "
    "and the offline adaptive experiment. Use the sidebar to change sections."
)

st.caption(f"Viewing: **{view}** · Select another section from the left-side navigation.")

if view == "Overview":
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
        st.info(
            "The saved analysis comes from the six project phases. The Live prediction page uses the already-fitted "
            "best-model artifact to score new rows, but it never retrains the model or stores uploaded data."
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

if view == "Customer analytics":
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

if view == "Prediction results":
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

if view == "Live prediction":
    section(
        "Live purchase prediction",
        "Enter one customer profile or upload a CSV. The saved best model scores the data using its fitted preprocessing; no retraining occurs.",
    )
    st.info(
        "Complete the form for one customer, or upload a CSV containing the required input columns. "
        "The dashboard returns a purchase probability and the tuned threshold-based prediction. "
        "Uploaded files are used only in the current browser session."
    )

    required_input_columns = list(prediction_schema)
    with st.form("single_customer_prediction"):
        st.markdown("### Enter customer behaviour and session details")
        form_columns = st.columns(2, gap="large")
        customer_values: dict[str, object] = {}
        for index, (column, details) in enumerate(prediction_schema.items()):
            field = form_columns[index % 2]
            label = column.replace("_", " ").title()
            with field:
                if details["kind"] == "numeric":
                    customer_values[column] = st.number_input(
                        label,
                        value=float(details["default"]),
                        format="%.4f",
                        key=f"live_{column}",
                    )
                elif details["kind"] == "choice":
                    choices = list(details["choices"])
                    customer_values[column] = st.selectbox(
                        label,
                        choices,
                        index=choices.index(str(details["default"])),
                        key=f"live_{column}",
                    )
                else:
                    customer_values[column] = st.text_input(
                        label,
                        value=str(details["default"]),
                        key=f"live_{column}",
                    )
        single_submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

    if single_submitted:
        new_customer = pd.DataFrame([customer_values], columns=required_input_columns)
        probability = float(best_model.predict_proba(new_customer)[0, 1])
        prediction = int(probability >= best_model.threshold)
        metric_columns = st.columns(3)
        metric_columns[0].metric("Purchase probability", f"{probability:.2%}")
        metric_columns[1].metric("Tuned threshold", f"{best_model.threshold:.4f}")
        metric_columns[2].metric("Prediction", "Likely to purchase" if prediction else "Less likely to purchase")
        if prediction:
            st.success("This profile is above the tuned threshold. Treat it as a candidate for a retail action, not a purchase guarantee.")
        else:
            st.info("This profile is below the tuned threshold. Consider an engagement or checkout-friction review where appropriate.")

    st.markdown("### Batch prediction from CSV")
    upload = st.file_uploader("Upload customer rows as CSV", type="csv")
    st.caption("Required columns: " + ", ".join(f"`{column}`" for column in required_input_columns))
    if upload is not None:
        uploaded_rows = pd.read_csv(upload)
        missing_columns = [column for column in required_input_columns if column not in uploaded_rows.columns]
        if missing_columns:
            st.error("The uploaded CSV is missing: " + ", ".join(missing_columns))
        else:
            scored_rows = uploaded_rows.copy()
            probabilities = best_model.predict_proba(uploaded_rows[required_input_columns])[:, 1]
            scored_rows["predicted_purchase_probability"] = probabilities
            scored_rows["tuned_threshold_prediction"] = (probabilities >= best_model.threshold).astype(int)
            st.success(f"Scored {len(scored_rows):,} uploaded row(s) with the saved model.")
            st.dataframe(scored_rows, hide_index=True, width="stretch", height=360)
            st.download_button(
                "Download scored CSV",
                data=scored_rows.to_csv(index=False).encode("utf-8"),
                file_name="retailiq_scored_predictions.csv",
                mime="text/csv",
            )

if view == "SHAP viewer":
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

if view == "Business insights":
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
        st.warning(f"Precision caveat: {caveat}")

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

if view == "Adaptive experiment":
    section("Adaptive before/after comparison", "An offline chronological-batch experiment with an explicit F1-drop retraining rule.")
    adaptive_metrics = st.columns(4)
    adaptive_metrics[0].metric("Batch evaluation F1", f"{later_state['f1']:.4f}")
    adaptive_metrics[1].metric("Batch ROC-AUC", f"{later_state['roc_auc']:.4f}")
    adaptive_metrics[2].metric("Evaluation rows", f"{int(later_state['evaluation_rows']):,}")
    adaptive_metrics[3].metric("Retraining triggered", "Yes" if bool(later_state["retrain_triggered"]) else "No")

    mode_match = re.search(r"Batch mode:\s*([^\n]+)", adaptive_summary)
    mode = mode_match.group(1).strip() if mode_match else "See saved summary"
    st.info(
        f"Observed mode: {mode}. This saved run used the confirmed visit-date field and a chronological split. "
        "The random STAND-IN fallback was not used. This remains an offline experiment, not production drift monitoring."
    )
    image_panel(FIGURES_DIR / "adaptive_before_after.png", "Observed adaptive decision comparison")
    st.dataframe(adaptive_table.style.format(precision=4), hide_index=True, width="stretch")
    with st.expander("Open the complete adaptive experiment summary"):
        st.code(adaptive_summary, language=None)

st.caption("RetailIQ · Dashboard over reproducible Phase 1–6 artifacts")
