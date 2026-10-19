"""PurchaseLens dashboard for saved artifacts and new customer scoring.

The dashboard never retrains a model. It can load the saved best-model artifact
to score user-entered rows or an uploaded CSV using the fitted preprocessing.
All analysis figures and phase summaries still come from saved result files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Streamlit starts this file from dashboard/, while the saved model refers
    # to the project-level src package when it is unpickled.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import RANDOM_SEED
from src.live_predict import (
    build_input_schema,
    explain_live_row,
    prepare_live_inputs,
    required_input_columns,
    score_live_rows,
)
from src.models import prepare_training_data

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


st.set_page_config(
    page_title="PurchaseLens | Purchase Intelligence",
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


def chart_interpretation(copy: str) -> None:
    """Keep every visual accompanied by a short faculty-demo interpretation."""
    st.caption("**What to notice:** " + copy)


def section(title: str, copy: str) -> None:
    """Render a theme-native section heading."""
    st.subheader(title)
    st.caption(copy)


@st.cache_resource(show_spinner="Preparing the saved SHAP explanation context...")
def load_live_shap_explainer(_model):
    """Build a fixed-background explainer for new rows without refitting a model."""
    classifier = _model.estimator.named_steps["classifier"]
    if classifier.__class__.__name__ != "LogisticRegression":
        raise TypeError("Live SHAP explanations are configured for the selected Logistic Regression model.")
    split = prepare_training_data(
        PROJECT_ROOT / "data",
        RESULTS_DIR / "dataset_summary.txt",
    )
    transformed_train = _model.estimator.named_steps["preprocessor"].transform(split["X_train"])
    background_size = min(200, transformed_train.shape[0])
    background = shap.sample(transformed_train, background_size, random_state=RANDOM_SEED)
    return shap.LinearExplainer(classifier, background)


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
prediction_schema = build_input_schema(best_model)

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
    st.markdown("## ◆ PurchaseLens")
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

st.title("PurchaseLens")
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
        chart_interpretation(
            "The purchase target is imbalanced, so F1-score and ROC-AUC were prioritised over accuracy when comparing models."
        )
    with right:
        st.markdown("### About PurchaseLens")
        st.info(
            "PurchaseLens is an explainable e-commerce decision-support prototype. It estimates purchase likelihood "
            "from pre-purchase session behaviour, explains the prediction, and maps the result to a cautious retail action."
        )
        st.markdown("#### What the prototype does")
        st.markdown(
            "- Uses a leakage-aware, pre-purchase feature pipeline\n"
            "- Scores new customer profiles with the saved selected model\n"
            "- Shows global and individual SHAP explanations\n"
            "- Suggests transparent, rule-based retail follow-up\n"
            "- Includes an offline chronological adaptive evaluation"
        )
        st.caption("New entries are processed in the current browser session only. The model is never retrained from dashboard inputs.")
    with st.expander("Open the complete Phase 1 dataset summary"):
        st.code(dataset_summary, language=None)

if view == "Customer analytics":
    section("Customer analytics & EDA", "Saved Phase 2 views of behaviour, class balance, association, encoding, and outliers.")
    image_panel(FIGURES_DIR / "correlation_heatmap.png", "Numeric association heatmap")
    chart_interpretation(
        "This heatmap is an exploratory view of numeric relationships. It supports feature understanding but does not establish causation."
    )

    st.markdown("### Behaviour versus purchase outcome")
    behaviour_paths = sorted(FIGURES_DIR.glob("behaviour_*_vs_target.png"))
    for start in range(0, len(behaviour_paths), 2):
        pair = st.columns(2, gap="large")
        for column, path in zip(pair, behaviour_paths[start : start + 2]):
            with column:
                label = path.stem.removeprefix("behaviour_").removesuffix("_vs_target").replace("_", " ").title()
                image_panel(path, label)
                chart_interpretation("Compare the purchase rate across the observed behavioural groups before interpreting the model output.")

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
                chart_interpretation("Diagonal cells are correct decisions; off-diagonal cells show false positives and false negatives at the selected cutoff.")

    st.markdown("### Sample predictions with explanations attached")
    st.dataframe(sample_predictions, hide_index=True, width="stretch", height=250)
    with st.expander("Open the complete modeling summary"):
        st.code(model_summary, language=None)

if view == "Live prediction":
    section(
        "Live purchase prediction & decision support",
        "Enter one customer profile or upload a CSV. The saved selected model returns a probability, threshold-based decision, actual local SHAP drivers, and a transparent retail-action suggestion.",
    )
    st.info(
        "This is live scoring, not retraining. New values are processed with the fitted preprocessing pipeline; "
        "entered and uploaded data remain in the current browser session."
    )

    input_columns = required_input_columns(prediction_schema)
    template_row = {
        column: details["default"]
        for column, details in prediction_schema.items()
    }
    st.download_button(
        "Download CSV input template",
        data=pd.DataFrame([template_row]).to_csv(index=False).encode("utf-8"),
        file_name="purchaselens_input_template.csv",
        mime="text/csv",
        help="Fill this template with one or more customer rows, then upload it below.",
    )

    with st.form("single_customer_prediction"):
        st.markdown("### Customer profile and pre-purchase session details")
        st.caption("All fields reflect columns retained by the saved preprocessing pipeline. Encoded category values are intentionally shown as dataset codes.")
        customer_values: dict[str, object] = {}
        field_groups = [
            ("Shopping and engagement", ["unit_price", "quantity", "discount_percent", "discount_amount", "pages_viewed", "time_on_site_sec", "added_to_cart"]),
            ("Visit context", ["visit_day", "visit_month", "visit_weekday", "visit_season", "session_duration_bucket"]),
            ("Customer and channel codes", ["device_type", "user_type", "marketing_channel", "product_category", "location"]),
        ]
        for group_name, group_columns in field_groups:
            st.markdown(f"#### {group_name}")
            form_columns = st.columns(3, gap="medium")
            for index, column in enumerate(group_columns):
                if column not in prediction_schema:
                    continue
                details = prediction_schema[column]
                label = column.replace("_", " ").title()
                with form_columns[index % 3]:
                    if details["kind"] == "numeric":
                        customer_values[column] = st.number_input(
                            label,
                            min_value=0.0,
                            value=float(details["default"]),
                            format="%.2f",
                            key=f"live_{column}",
                        )
                    elif details["kind"] == "choice":
                        choices = list(details["choices"])
                        customer_values[column] = st.selectbox(
                            label,
                            choices,
                            index=choices.index(details["default"]),
                            key=f"live_{column}",
                        )
                    elif details["kind"] == "frequency" and bool(details["numeric"]):
                        customer_values[column] = st.number_input(
                            label,
                            min_value=0.0,
                            value=float(details["default"]),
                            step=1.0,
                            format="%.0f",
                            key=f"live_{column}",
                        )
                    else:
                        customer_values[column] = st.text_input(
                            label,
                            value=str(details["default"]),
                            key=f"live_{column}",
                        )
        single_submitted = st.form_submit_button("Generate purchase decision and explanation", type="primary", use_container_width=True)

    if single_submitted:
        entered_row = pd.DataFrame([customer_values], columns=input_columns)
        scored_customer, notes = score_live_rows(best_model, entered_row, prediction_schema)
        prepared_row, preparation_notes = prepare_live_inputs(entered_row, prediction_schema)
        live_explainer = load_live_shap_explainer(best_model)
        contributions = explain_live_row(best_model, prepared_row, live_explainer)
        st.session_state["live_prediction_result"] = {
            "scored": scored_customer,
            "notes": [*notes, *preparation_notes],
            "contributions": contributions,
        }

    if "live_prediction_result" in st.session_state:
        live_result = st.session_state["live_prediction_result"]
        result = live_result["scored"].iloc[0]
        probability = float(result["predicted_purchase_probability"])
        prediction = int(result["tuned_threshold_prediction"])
        action = str(result["recommended_action_hint"])
        metric_columns = st.columns(4)
        metric_columns[0].metric("Purchase probability", f"{probability:.2%}")
        metric_columns[1].metric("Decision", "Likely to purchase" if prediction else "Less likely to purchase")
        metric_columns[2].metric("Threshold position", "Above selected cutoff" if prediction else "Below selected cutoff")
        metric_columns[3].metric("Recommended action", action.replace("Candidate for promotion - ", "Promotion candidate"))

        if prediction:
            st.success("Action guidance: this is a candidate for follow-up, not a guarantee of purchase. Moderate precision means some flagged customers will not purchase.")
        else:
            st.info("Action guidance: the score is below the selected cutoff. Use engagement or checkout-friction follow-up where the driver pattern supports it.")

        explanation_col, action_col = st.columns([1.2, 0.8], gap="large")
        with explanation_col:
            st.markdown("### Customer-level SHAP drivers")
            st.caption("These are actual local SHAP values for the newly entered row. Positive values increase and negative values decrease the model's purchase likelihood on the log-odds scale.")
            contribution_view = live_result["contributions"].head(6).loc[:, ["direction", "feature", "shap_value"]]
            st.dataframe(contribution_view.style.format({"shap_value": "{:+.4f}"}), hide_index=True, width="stretch")
        with action_col:
            st.markdown("### Decision-support recommendation")
            st.write(f"**{action}**")
            st.caption("The recommendation uses simple, inspectable probability and engagement rules. It is not a second machine-learning model.")
            st.markdown("**How to use it**")
            st.markdown(
                "- Use promotion candidates for limited, monitored offers.\n"
                "- Review engagement and cart signals before intervening.\n"
                "- Treat the output as decision support, not an automated guarantee."
            )
        for note in live_result["notes"]:
            st.warning(note)

    st.markdown("### Batch prediction from CSV")
    upload = st.file_uploader("Upload customer rows as CSV", type="csv")
    st.caption("Required columns: " + ", ".join(f"`{column}`" for column in input_columns))
    if upload is not None:
        uploaded_rows = pd.read_csv(upload)
        try:
            scored_rows, notes = score_live_rows(best_model, uploaded_rows, prediction_schema)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"Scored {len(scored_rows):,} uploaded row(s) with the saved model.")
            for note in notes:
                st.warning(note)
            st.dataframe(scored_rows, hide_index=True, width="stretch", height=360)
            st.download_button(
                "Download scored CSV",
                data=scored_rows.to_csv(index=False).encode("utf-8"),
                file_name="purchaselens_scored_predictions.csv",
                mime="text/csv",
            )

if view == "SHAP viewer":
    section("SHAP explanation viewer", "Move from global model behaviour to three real local test-row explanations.")
    global_cols = st.columns(2, gap="large")
    with global_cols[0]:
        image_panel(FIGURES_DIR / "shap_global_bar_best_model.png", "Global importance · mean absolute SHAP")
        chart_interpretation("Features near the top have the greatest average influence on model predictions across the held-out sample.")
    with global_cols[1]:
        image_panel(FIGURES_DIR / "shap_global_beeswarm_best_model.png", "Global direction and magnitude")
        chart_interpretation("Point position indicates whether feature values pushed predictions upward or downward; spread shows variation across customers.")

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
        chart_interpretation("Bars pushing right increase the purchase score; bars pushing left reduce it for this real held-out customer example.")
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
        chart_interpretation("This chart summarises how the transparent Phase 5 action rules distributed the held-out customer records.")
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
    chart_interpretation("The later chronological batch is assessed against an explicit F1-drop trigger; the chart reports the observed result without forcing retraining.")
    st.dataframe(adaptive_table.style.format(precision=4), hide_index=True, width="stretch")
    with st.expander("Open the complete adaptive experiment summary"):
        st.code(adaptive_summary, language=None)

st.caption("PurchaseLens · Dashboard over reproducible Phase 1–6 artifacts")
