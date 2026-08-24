"""Phase 4 SHAP explanations for the saved threshold-corrected best model.

The fitted preprocessing and classifier are loaded from the Phase 3b artifact.
No model is retrained. Global explanations use a fixed held-out sample, while
local explanations use real high, low, and threshold-borderline test rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_prep import RANDOM_SEED
from src.models import ThresholdedClassifier, prepare_training_data


BACKGROUND_SAMPLE_SIZE = 200
GLOBAL_EXPLANATION_SIZE = 1000
MAX_DISPLAY_FEATURES = 15


def _readable_feature_names(names: np.ndarray) -> np.ndarray:
    """Remove ColumnTransformer prefixes while retaining actual encoded labels."""
    readable = []
    for name in names.astype(str):
        cleaned = name.split("__", maxsplit=1)[-1]
        readable.append(cleaned)
    if len(readable) != len(set(readable)):
        return names.astype(str)
    return np.asarray(readable, dtype=object)


def load_explanation_context(
    artifact_path: str | Path = "models/best_model.joblib",
    data_dir: str | Path = "data",
    phase1_summary_path: str | Path = "results/dataset_summary.txt",
) -> dict[str, Any]:
    """Load the selected fitted model and reconstruct its fixed held-out split."""
    artifact = joblib.load(artifact_path)
    if isinstance(artifact, ThresholdedClassifier):
        pipeline = artifact.estimator
        threshold = float(artifact.threshold)
    elif isinstance(artifact, Pipeline):
        pipeline = artifact
        threshold = 0.5
    else:
        raise TypeError(f"Unsupported saved artifact type: {type(artifact).__name__}")
    if not {"preprocessor", "classifier"}.issubset(pipeline.named_steps):
        raise ValueError("Saved pipeline must contain preprocessor and classifier steps.")

    data = prepare_training_data(data_dir, phase1_summary_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_train_transformed = preprocessor.transform(data["X_train"])
    X_test_transformed = preprocessor.transform(data["X_test"])
    feature_names = _readable_feature_names(preprocessor.get_feature_names_out())
    if X_test_transformed.shape[1] != len(feature_names):
        raise ValueError("Transformed feature count and feature-name count differ.")
    if X_test_transformed.shape[1] >= 300:
        raise RuntimeError(
            f"Refusing SHAP on {X_test_transformed.shape[1]} features; corrected preprocessing was not loaded."
        )

    probabilities = pipeline.predict_proba(data["X_test"])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "artifact": artifact,
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "classifier": classifier,
        "threshold": threshold,
        "data": data,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "feature_names": feature_names,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def select_representative_rows(context: dict[str, Any]) -> pd.DataFrame:
    """Select distinct high, low, and tuned-threshold-borderline test rows."""
    probabilities = context["probabilities"]
    threshold = context["threshold"]
    candidates = {
        "high_probability": int(np.argmax(probabilities)),
        "low_probability": int(np.argmin(probabilities)),
        "borderline": int(np.argmin(np.abs(probabilities - threshold))),
    }
    if len(set(candidates.values())) != 3:
        raise RuntimeError("High, low, and borderline row selection did not produce distinct rows.")

    X_test = context["data"]["X_test"]
    y_test = context["data"]["y_test"]
    records = []
    for example_type, position in candidates.items():
        records.append(
            {
                "example_type": example_type,
                "test_position": position,
                "source_row_index": X_test.index[position],
                "probability": float(probabilities[position]),
                "threshold": threshold,
                "prediction": int(context["predictions"][position]),
                "actual_label": int(y_test.iloc[position]),
            }
        )
    return pd.DataFrame.from_records(records)


def create_shap_explainer(context: dict[str, Any]) -> tuple[Any, str]:
    """Create the model-family-appropriate SHAP explainer without refitting."""
    classifier = context["classifier"]
    if isinstance(classifier, RandomForestClassifier) or classifier.__class__.__module__.startswith("xgboost"):
        return shap.TreeExplainer(classifier), "TreeExplainer"
    if isinstance(classifier, LogisticRegression):
        background = shap.sample(
            context["X_train_transformed"],
            BACKGROUND_SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        )
        return shap.LinearExplainer(classifier, background), "LinearExplainer"
    raise TypeError(
        f"No approved SHAP explainer is configured for {type(classifier).__name__}."
    )


def _positive_class_explanation(explanation: shap.Explanation) -> shap.Explanation:
    """Select positive-class contributions when an explainer returns both classes."""
    if explanation.values.ndim == 3:
        return shap.Explanation(
            values=explanation.values[:, :, 1],
            base_values=explanation.base_values[:, 1],
            data=explanation.data,
            feature_names=explanation.feature_names,
        )
    return explanation


def compute_shap_explanations(
    context: dict[str, Any],
    examples: pd.DataFrame,
) -> dict[str, Any]:
    """Compute fixed-sample global SHAP values and the three local explanations."""
    explainer, explainer_name = create_shap_explainer(context)
    row_count = context["X_test_transformed"].shape[0]
    sample_count = min(GLOBAL_EXPLANATION_SIZE, row_count)
    generator = np.random.default_rng(RANDOM_SEED)
    global_positions = np.sort(generator.choice(row_count, size=sample_count, replace=False))
    global_matrix = context["X_test_transformed"][global_positions]
    local_positions = examples["test_position"].astype(int).to_numpy()
    local_matrix = context["X_test_transformed"][local_positions]

    global_explanation = _positive_class_explanation(explainer(global_matrix))
    local_explanation = _positive_class_explanation(explainer(local_matrix))
    global_explanation.feature_names = context["feature_names"].tolist()
    local_explanation.feature_names = context["feature_names"].tolist()
    return {
        "explainer": explainer,
        "explainer_name": explainer_name,
        "global_positions": global_positions,
        "global_explanation": global_explanation,
        "local_explanation": local_explanation,
    }


def plot_global_shap(
    shap_results: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Save mean-absolute SHAP bar and beeswarm plots for the selected model."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    explanation = shap_results["global_explanation"]

    shap.plots.bar(explanation, max_display=MAX_DISPLAY_FEATURES, show=False)
    plt.title("Global SHAP Importance - Mean Absolute Contribution")
    plt.tight_layout()
    bar_path = destination / "shap_global_bar_best_model.png"
    plt.savefig(bar_path, dpi=170, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(explanation, max_display=MAX_DISPLAY_FEATURES, show=False)
    plt.title("Global SHAP Beeswarm - Direction and Magnitude")
    plt.tight_layout()
    beeswarm_path = destination / "shap_global_beeswarm_best_model.png"
    plt.savefig(beeswarm_path, dpi=170, bbox_inches="tight")
    plt.close()
    return [bar_path, beeswarm_path]


def plot_local_shap(
    shap_results: dict[str, Any],
    examples: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save one SHAP waterfall for each selected real test row."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    local = shap_results["local_explanation"]
    for index, row in examples.reset_index(drop=True).iterrows():
        shap.plots.waterfall(local[index], max_display=12, show=False)
        plt.gcf().suptitle(
            f"{row['example_type'].replace('_', ' ').title()} Test Row\n"
            f"P(purchase)={row['probability']:.4f}, threshold={row['threshold']:.4f}, "
            f"prediction={int(row['prediction'])}, actual={int(row['actual_label'])}",
            y=1.02,
        )
        plt.tight_layout()
        safe_type = re.sub(r"[^a-z0-9]+", "_", row["example_type"].lower()).strip("_")
        path = destination / f"shap_local_{safe_type}.png"
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def build_local_contribution_table(
    context: dict[str, Any],
    shap_results: dict[str, Any],
    examples: pd.DataFrame,
) -> pd.DataFrame:
    """List the three strongest positive and negative SHAP values per example."""
    values = np.asarray(shap_results["local_explanation"].values)
    names = context["feature_names"]
    records = []
    for row_index, example in examples.reset_index(drop=True).iterrows():
        row_values = values[row_index]
        positive = [index for index in np.argsort(row_values)[::-1] if row_values[index] > 0][:3]
        negative = [index for index in np.argsort(row_values) if row_values[index] < 0][:3]
        record = example.to_dict()
        for rank in range(3):
            if rank < len(positive):
                feature_index = positive[rank]
                record[f"positive_feature_{rank + 1}"] = names[feature_index]
                record[f"positive_shap_{rank + 1}"] = float(row_values[feature_index])
            else:
                record[f"positive_feature_{rank + 1}"] = ""
                record[f"positive_shap_{rank + 1}"] = np.nan
            if rank < len(negative):
                feature_index = negative[rank]
                record[f"negative_feature_{rank + 1}"] = names[feature_index]
                record[f"negative_shap_{rank + 1}"] = float(row_values[feature_index])
            else:
                record[f"negative_feature_{rank + 1}"] = ""
                record[f"negative_shap_{rank + 1}"] = np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def render_explanation_summary(
    context: dict[str, Any],
    shap_results: dict[str, Any],
    examples: pd.DataFrame,
    contribution_table: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Render a reproducible record of the Phase 4 explanation run."""
    classifier_name = type(context["classifier"]).__name__
    lines = [
        "PHASE 4 - SHAP EXPLAINABILITY SUMMARY",
        "=" * 37,
        f"Loaded classifier: {classifier_name}",
        f"SHAP explainer: {shap_results['explainer_name']}",
        f"Operational threshold: {context['threshold']:.6f}",
        f"Corrected transformed features: {context['X_test_transformed'].shape[1]}",
        f"Global held-out SHAP sample: {len(shap_results['global_positions'])} rows (random_state={RANDOM_SEED})",
        "For Logistic Regression, SHAP values are contributions on the model's raw log-odds scale.",
        "",
        "SELECTED REAL TEST EXAMPLES",
        "---------------------------",
        examples.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "TOP LOCAL CONTRIBUTIONS",
        "-----------------------",
        contribution_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "SAVED FIGURES",
        "-------------",
        *[f"- {path.resolve()}" for path in figure_paths],
    ]
    return "\n".join(lines) + "\n"


def run_phase4(
    artifact_path: str | Path = "models/best_model.joblib",
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
) -> dict[str, Any]:
    """Load the best model, generate actual SHAP outputs, and save all artifacts."""
    result_path = Path(results_dir)
    context = load_explanation_context(
        artifact_path,
        data_dir,
        result_path / "dataset_summary.txt",
    )
    examples = select_representative_rows(context)
    shap_results = compute_shap_explanations(context, examples)
    global_paths = plot_global_shap(shap_results, result_path / "figures")
    local_paths = plot_local_shap(shap_results, examples, result_path / "figures")
    contribution_table = build_local_contribution_table(context, shap_results, examples)
    table_path = result_path / "tables" / "shap_local_examples.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    contribution_table.to_csv(table_path, index=False, float_format="%.8f")
    summary = render_explanation_summary(
        context,
        shap_results,
        examples,
        contribution_table,
        [*global_paths, *local_paths],
    )
    summary_path = result_path / "shap_explainability_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Loaded best classifier: {type(context['classifier']).__name__}")
    print(f"Explainer: {shap_results['explainer_name']}")
    print(f"Corrected feature count: {context['X_test_transformed'].shape[1]}")
    print("Selected real test rows and top SHAP contributions:")
    print(contribution_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Phase 4 summary written to: {summary_path.resolve()}")
    return {
        "context": context,
        "examples": examples,
        "shap_results": shap_results,
        "contribution_table": contribution_table,
        "global_paths": global_paths,
        "local_paths": local_paths,
        "table_path": table_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_phase4()
