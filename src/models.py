"""Phase 3 model construction and training.

This module reuses the Phase 2 cleaning and preprocessing functions. It does not
redefine target, leakage, or encoding logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_prep import (
    MAX_ONE_HOT_CARDINALITY,
    RANDOM_SEED,
    build_model_preprocessors,
    clean_dataset,
    identify_feature_roles,
    load_dataset,
    load_phase1_decisions,
)


MAX_TOTAL_FEATURES = 300
TEST_SIZE = 0.20


def audit_existing_encoding_decisions(
    df: pd.DataFrame,
    decisions_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Audit the persisted Phase 2 table before any model is trained.

    Existing one-hot fields above the allowed cardinality are documented. Raw
    identifier fields are marked for removal; other high-cardinality categories
    are marked for unsupervised frequency encoding.
    """
    decisions = pd.read_csv(decisions_path)
    required = {"column", "encoding"}
    if not required.issubset(decisions.columns):
        raise ValueError(f"Encoding decision table must contain {sorted(required)}.")

    records = []
    for column in decisions.loc[decisions["encoding"].eq("one-hot"), "column"]:
        if column not in df.columns:
            raise KeyError(f"Encoding decision column {column!r} is absent from the CSV.")
        unique_values = int(df[column].nunique(dropna=True))
        if unique_values <= MAX_ONE_HOT_CARDINALITY:
            continue
        name_parts = set(str(column).lower().replace("-", "_").split("_"))
        if "id" in name_parts:
            correction = "drop"
            reason = "raw high-cardinality identifier has no stable ordinal meaning and may encourage memorisation"
        else:
            correction = "frequency encoding"
            reason = "retain prevalence information in one numeric feature without using the target"
        records.append(
            {
                "column": column,
                "unique_values": unique_values,
                "previous_encoding": "one-hot",
                "previous_output_columns": unique_values,
                "correction": correction,
                "reason": reason,
            }
        )

    audit = pd.DataFrame.from_records(records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if audit.empty and destination.exists():
        # Preserve the original Phase 3 audit on repeat runs after the corrected
        # decisions table has replaced the old Phase 2 table.
        return pd.read_csv(destination)
    audit.to_csv(destination, index=False)
    return audit


def prepare_training_data(
    data_dir: str | Path = "data",
    phase1_summary_path: str | Path = "results/dataset_summary.txt",
) -> dict[str, Any]:
    """Load, clean, and stratify data using the established project decisions."""
    raw, csv_path = load_dataset(data_dir)
    target, leakage = load_phase1_decisions(phase1_summary_path)
    cleaned, missing_decisions, duplicate_report = clean_dataset(raw, target)
    roles = identify_feature_roles(cleaned, target, leakage)
    X = cleaned.drop(columns=[target])
    y = cleaned[target].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return {
        "csv_path": csv_path,
        "cleaned": cleaned,
        "target": target,
        "leakage": leakage,
        "roles": roles,
        "missing_decisions": missing_decisions,
        "duplicate_report": duplicate_report,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def verify_corrected_preprocessing(
    X_train: pd.DataFrame,
    roles: dict[str, Any],
) -> dict[str, int]:
    """Refuse modeling if high-cardinality one-hot or total expansion remains."""
    remaining_large_one_hot = {
        column: int(X_train[column].nunique(dropna=True))
        for column in roles["nominal"]
        if X_train[column].nunique(dropna=True) > MAX_ONE_HOT_CARDINALITY
    }
    if remaining_large_one_hot:
        raise RuntimeError(
            f"High-cardinality one-hot fields remain: {remaining_large_one_hot}. "
            "Correct encoding before training."
        )

    tree_check, linear_check = build_model_preprocessors(roles)
    tree_count = int(tree_check.fit_transform(X_train).shape[1])
    linear_count = int(linear_check.fit_transform(X_train).shape[1])
    if max(tree_count, linear_count) >= MAX_TOTAL_FEATURES:
        raise RuntimeError(
            f"Corrected feature count is still too large: tree={tree_count}, "
            f"linear={linear_count}; limit is below {MAX_TOTAL_FEATURES}."
        )
    return {"tree": tree_count, "linear": linear_count}


def build_model_pipelines(roles: dict[str, Any]) -> dict[str, Pipeline]:
    """Construct the two required class-balanced model pipelines."""
    tree_preprocessor, linear_preprocessor = build_model_preprocessors(roles)
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", linear_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        max_iter=1000,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        # A single worker avoids duplicating the forest's memory
                        # across processes on ordinary student laptops.
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def train_models(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Fit each requested model on the same stratified training partition."""
    trained = {}
    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        trained[name] = pipeline.fit(X_train, y_train)
    return trained


# XGBoost is intentionally not imported or installed. The project-wide allowed
# dependency list is limited to pandas/numpy/scikit-learn/SHAP/plotting packages;
# forcing XGBoost would break that reproducibility constraint. The two requested
# scikit-learn models therefore remain the complete Phase 3 comparison.
