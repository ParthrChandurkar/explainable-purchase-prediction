"""Reusable live scoring helpers for the Streamlit PurchaseLens prototype.

The functions in this module load no training data and never fit a model. They
only use the saved fitted pipeline to score new customer rows consistently with
the preprocessing used in Phase 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_input_schema(model: Any) -> dict[str, dict[str, Any]]:
    """Read the required live-input fields from the fitted preprocessor.

    Raw category values are deliberately retained instead of converted to text.
    This matters because a saved encoder trained on integer category codes must
    receive integers, not strings such as ``"1"``.
    """
    preprocessor = model.estimator.named_steps["preprocessor"]
    schema: dict[str, dict[str, Any]] = {}

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
                values = list(choices)
                schema[column] = {
                    "kind": "choice",
                    "choices": values,
                    "default": values[0],
                }

        elif name == "frequency":
            defaults = transformer.named_steps["imputer"].statistics_
            for column, default in zip(columns, defaults):
                schema[column] = {
                    "kind": "frequency",
                    "default": default,
                    "numeric": bool(pd.api.types.is_number(default)),
                }

        elif name == "ordinal":
            categories = transformer.named_steps["ordinal"].categories_
            for column, choices in zip(columns, categories):
                values = list(choices)
                schema[column] = {
                    "kind": "choice",
                    "choices": values,
                    "default": values[0],
                }

    if not schema:
        raise ValueError("The saved model does not expose a usable preprocessing schema.")
    return schema


def required_input_columns(schema: dict[str, dict[str, Any]]) -> list[str]:
    """Return columns in the same deterministic order as the fitted pipeline."""
    return list(schema)


def _coerce_categories(series: pd.Series, choices: list[Any]) -> tuple[pd.Series, list[str]]:
    """Match display/upload text back to the original fitted category type."""
    lookup = {str(value): value for value in choices}
    unknown = sorted(
        {
            str(value)
            for value in series.dropna().unique()
            if str(value) not in lookup
        }
    )
    return series.map(lambda value: lookup.get(str(value), value)), unknown


def prepare_live_inputs(
    rows: pd.DataFrame,
    schema: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and type-align manual or uploaded customer rows for scoring."""
    required = required_input_columns(schema)
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))

    prepared = rows.loc[:, required].copy()
    notes: list[str] = []
    for column, details in schema.items():
        kind = details["kind"]
        if kind == "numeric":
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        elif kind == "choice":
            prepared[column], unknown = _coerce_categories(
                prepared[column], list(details["choices"])
            )
            if unknown:
                notes.append(
                    f"{column}: unseen value(s) {', '.join(unknown[:5])} were passed to the model; "
                    "the fitted encoder handles them as unknown categories."
                )
        elif kind == "frequency" and bool(details["numeric"]):
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared, notes


def recommend_live_action(
    probability: float,
    threshold: float,
    row: pd.Series,
    schema: dict[str, dict[str, Any]],
) -> str:
    """Return a simple, inspectable action hint for a newly scored row.

    This is deliberately rule-based. It is not a second model and it does not
    claim to be a SHAP explanation for the new row.
    """
    if probability >= threshold + 0.10:
        return "Candidate for promotion - higher score"
    if probability >= threshold:
        return "Candidate for promotion - threshold-qualified"

    engagement_fields = ["pages_viewed", "time_on_site_sec", "added_to_cart"]
    above_typical = 0
    for field in engagement_fields:
        if field in row.index and field in schema:
            default = float(schema[field]["default"])
            if pd.notna(row[field]) and float(row[field]) >= default:
                above_typical += 1
    if above_typical >= 2:
        return "Investigate checkout/discount friction"
    return "Retention/engagement nudge"


def score_live_rows(
    model: Any,
    rows: pd.DataFrame,
    schema: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[str]]:
    """Score new rows and attach probability, decision, and action hint columns."""
    prepared, notes = prepare_live_inputs(rows, schema)
    probabilities = model.predict_proba(prepared)[:, 1]
    threshold = float(model.threshold)

    scored = rows.copy()
    scored["predicted_purchase_probability"] = probabilities
    scored["tuned_threshold_prediction"] = (probabilities >= threshold).astype(int)
    scored["recommended_action_hint"] = [
        recommend_live_action(float(probability), threshold, row, schema)
        for probability, (_, row) in zip(probabilities, prepared.iterrows())
    ]
    return scored, notes
