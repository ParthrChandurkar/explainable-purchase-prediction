"""Phase 5 transparent retail-action rules over model and SHAP outputs.

This module trains nothing. It loads the Phase 3b artifact, obtains Phase 4-style
SHAP contributions for every held-out row, and maps probability plus confirmed
feature drivers to simple, inspectable action labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

from src.explain import create_shap_explainer, load_explanation_context


PROMOTION_MARGIN = 0.10
STRONG_POSITIVE_SHAP = 0.10
TOP_DRIVER_COUNT = 3

PROMOTION_HIGH = "Candidate for promotion - higher score"
PROMOTION_THRESHOLD = "Candidate for promotion - threshold-qualified"
CHECKOUT_FRICTION = "Investigate checkout/discount friction"
ENGAGEMENT_NUDGE = "Retention/engagement nudge"
PERSONALIZED_RECOMMENDATION = "Personalized recommendation opportunity"
ACTION_ORDER = [
    PERSONALIZED_RECOMMENDATION,
    PROMOTION_HIGH,
    PROMOTION_THRESHOLD,
    CHECKOUT_FRICTION,
    ENGAGEMENT_NUDGE,
]

PRECISION_CAVEAT_TEMPLATE = (
    "At the tuned threshold, held-out precision is {precision:.4f}. Therefore, "
    "candidate-for-promotion flags will include a meaningful share of false "
    "positives and should be treated as candidates for low-cost outreach, not "
    "as guaranteed purchasers."
)


def discover_confirmed_driver_groups(feature_names: np.ndarray) -> dict[str, list[str]]:
    """Build rule groups only from feature names present in the fitted pipeline."""
    actual_names = [str(name) for name in feature_names]
    engagement_tokens = ("cart", "pages", "session", "time_on_site")
    personalization_tokens = ("product_category", "location_frequency")
    return {
        "engagement": [
            name for name in actual_names if any(token in name.lower() for token in engagement_tokens)
        ],
        "personalization": [
            name for name in actual_names if any(token in name.lower() for token in personalization_tokens)
        ],
    }


def compute_all_test_shap(context: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Calculate positive-class SHAP contributions for all held-out rows."""
    explainer, explainer_name = create_shap_explainer(context)
    explanation = explainer(context["X_test_transformed"])
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != context["X_test_transformed"].shape:
        raise ValueError(
            f"SHAP shape {values.shape} does not match transformed test shape "
            f"{context['X_test_transformed'].shape}."
        )
    return values, explainer_name


def _top_indices(shap_values: np.ndarray) -> np.ndarray:
    return np.argsort(np.abs(shap_values))[::-1][:TOP_DRIVER_COUNT]


def _format_top_drivers(
    row_values: np.ndarray,
    feature_names: np.ndarray,
) -> str:
    indices = _top_indices(row_values)
    return " | ".join(
        f"{feature_names[index]} ({row_values[index]:+.4f})" for index in indices
    )


def recommend_action(
    probability: float,
    threshold: float,
    row_values: np.ndarray,
    feature_names: np.ndarray,
    groups: dict[str, list[str]],
) -> str:
    """Apply one deterministic action rule using only probability and top SHAP drivers."""
    top_indices = _top_indices(row_values)
    top_positive = {
        str(feature_names[index]): float(row_values[index])
        for index in top_indices
        if row_values[index] >= STRONG_POSITIVE_SHAP
    }
    personalization_driver = any(
        name in groups["personalization"] for name in top_positive
    )
    engagement_driver = any(name in groups["engagement"] for name in top_positive)

    # Priority is explicit: a strong product/location driver gets the more
    # specific action; otherwise probability controls promotion candidacy.
    if personalization_driver:
        return PERSONALIZED_RECOMMENDATION
    if probability >= threshold + PROMOTION_MARGIN:
        return PROMOTION_HIGH
    if probability >= threshold:
        return PROMOTION_THRESHOLD
    if engagement_driver:
        return CHECKOUT_FRICTION
    return ENGAGEMENT_NUDGE


def build_business_insights(
    context: dict[str, Any],
    shap_values: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Create the required five-column action table for every test row."""
    feature_names = context["feature_names"]
    groups = discover_confirmed_driver_groups(feature_names)
    probabilities = context["probabilities"]
    predictions = context["predictions"]
    row_ids = context["data"]["X_test"].index.to_numpy()
    if len(shap_values) != len(row_ids):
        raise ValueError("SHAP row count differs from the held-out row count.")

    records = []
    for position, row_id in enumerate(row_ids):
        row_values = shap_values[position]
        records.append(
            {
                "row_id": row_id,
                "predicted_probability": float(probabilities[position]),
                "tuned_threshold_prediction": int(predictions[position]),
                "top_drivers": _format_top_drivers(row_values, feature_names),
                "recommended_action": recommend_action(
                    float(probabilities[position]),
                    float(context["threshold"]),
                    row_values,
                    feature_names,
                    groups,
                ),
            }
        )
    return pd.DataFrame.from_records(records), groups


def aggregate_action_summary(
    insights: pd.DataFrame,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Count rule outcomes and calculate the observed tuned-threshold precision caveat."""
    counts = (
        insights["recommended_action"]
        .value_counts()
        .reindex(ACTION_ORDER, fill_value=0)
        .rename_axis("recommended_action")
    )
    aggregate = counts.reset_index(name="row_count")
    aggregate["percent_of_test_rows"] = aggregate["row_count"].div(len(insights)).mul(100.0)
    precision = precision_score(
        context["data"]["y_test"],
        context["predictions"],
        zero_division=0,
    )
    caveat = PRECISION_CAVEAT_TEMPLATE.format(precision=precision)
    return aggregate, caveat


def render_business_insights_summary(
    context: dict[str, Any],
    explainer_name: str,
    groups: dict[str, list[str]],
    aggregate: pd.DataFrame,
    caveat: str,
    table_path: Path,
) -> str:
    """Render the actual rules, confirmed fields, counts, and precision warning."""
    lines = [
        "PHASE 5 - RULE-BASED BUSINESS INSIGHTS SUMMARY",
        "=" * 46,
        f"Rows processed: {len(context['data']['X_test'])}",
        f"Loaded classifier: {type(context['classifier']).__name__}",
        f"Operational threshold: {context['threshold']:.6f}",
        f"SHAP explainer reused: {explainer_name}",
        f"Corrected features: {len(context['feature_names'])}",
        "",
        "CONFIRMED FEATURE GROUPS USED BY RULES",
        "--------------------------------------",
        f"Engagement: {groups['engagement']}",
        f"Personalization: {groups['personalization']}",
        "",
        "INSPECTABLE RULE ORDER",
        "----------------------",
        f"1. Top-3 positive product/category or location-frequency driver >= {STRONG_POSITIVE_SHAP:.2f} SHAP -> {PERSONALIZED_RECOMMENDATION}",
        f"2. Probability >= threshold + {PROMOTION_MARGIN:.2f} -> {PROMOTION_HIGH}",
        f"3. Probability >= threshold -> {PROMOTION_THRESHOLD}",
        f"4. Below threshold with top-3 positive engagement driver >= {STRONG_POSITIVE_SHAP:.2f} SHAP -> {CHECKOUT_FRICTION}",
        f"5. Remaining below-threshold rows -> {ENGAGEMENT_NUDGE}",
        "",
        "ACTION COUNTS",
        "-------------",
        aggregate.to_string(index=False, float_format=lambda value: f"{value:.2f}"),
        "",
        "PRECISION CAVEAT",
        "----------------",
        caveat,
        "",
        f"Full insights table: {table_path.resolve()}",
    ]
    return "\n".join(lines) + "\n"


def run_phase5(
    artifact_path: str | Path = "models/best_model.joblib",
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
) -> dict[str, Any]:
    """Generate all-row SHAP-driven actions and save the required outputs."""
    result_path = Path(results_dir)
    context = load_explanation_context(
        artifact_path,
        data_dir,
        result_path / "dataset_summary.txt",
    )
    shap_values, explainer_name = compute_all_test_shap(context)
    insights, groups = build_business_insights(context, shap_values)
    aggregate, caveat = aggregate_action_summary(insights, context)
    table_path = result_path / "tables" / "business_insights.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    insights.to_csv(table_path, index=False, float_format="%.8f")
    summary = render_business_insights_summary(
        context,
        explainer_name,
        groups,
        aggregate,
        caveat,
        table_path,
    )
    summary_path = result_path / "business_insights_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("Action counts across the held-out test set:")
    print(aggregate.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\n{caveat}")
    print(f"\nFull insights table written to: {table_path.resolve()}")
    return {
        "context": context,
        "shap_values": shap_values,
        "explainer_name": explainer_name,
        "insights": insights,
        "groups": groups,
        "aggregate": aggregate,
        "caveat": caveat,
        "table_path": table_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_phase5()
