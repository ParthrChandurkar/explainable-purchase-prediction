"""Phase 6 chronological adaptive-model experiment.

The dataset contains a usable visit-date field confirmed in the Phase 1 report,
so this run uses real chronological batches. If no usable date-like field were
found, the fallback would use a fixed random split explicitly labeled as a
STAND-IN simulation—not evidence of real temporal drift.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_prep import (
    RANDOM_SEED,
    clean_dataset,
    identify_feature_roles,
    load_dataset,
    load_phase1_decisions,
)
from src.models import build_model_pipelines


EARLIER_BATCH_FRACTION = 0.70
F1_DROP_TRIGGER = 0.05
COMBINED_HOLDOUT_SIZE = 0.20


def find_usable_time_column(
    df: pd.DataFrame,
    phase1_summary_path: str | Path,
) -> tuple[str | None, pd.Series | None, list[dict[str, Any]]]:
    """Discover a date/time field and confirm that Phase 1 recorded its name."""
    phase1_text = Path(phase1_summary_path).read_text(encoding="utf-8")
    reports = []
    for column in df.columns:
        parts = set(filter(None, re.sub(r"[^a-z0-9]+", "_", str(column).lower()).split("_")))
        if not parts.intersection({"date", "datetime", "timestamp", "time"}):
            continue
        series = df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
        ):
            continue
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        parse_rate = float(parsed.notna().mean())
        unique_dates = int(parsed.nunique(dropna=True))
        confirmed = bool(re.search(rf"(?m)^{re.escape(str(column))}\s*$", phase1_text))
        reports.append(
            {
                "column": str(column),
                "parse_success_percent": 100.0 * parse_rate,
                "unique_timestamps": unique_dates,
                "confirmed_in_phase1_summary": confirmed,
            }
        )
    usable = [
        report
        for report in reports
        if report["parse_success_percent"] >= 90.0
        and report["unique_timestamps"] >= 2
        and report["confirmed_in_phase1_summary"]
    ]
    if not usable:
        return None, None, reports
    selected = sorted(
        usable,
        key=lambda item: (-item["parse_success_percent"], -item["unique_timestamps"], item["column"]),
    )[0]
    return (
        selected["column"],
        pd.to_datetime(df[selected["column"]], errors="coerce", dayfirst=True),
        reports,
    )


def create_adaptive_batches(
    cleaned: pd.DataFrame,
    target: str,
    parsed_time: pd.Series | None,
) -> dict[str, Any]:
    """Create real chronological batches, or a clearly labeled random stand-in."""
    if parsed_time is not None:
        aligned_time = parsed_time.loc[cleaned.index]
        valid_dates = np.sort(aligned_time.dropna().unique())
        cutoff_position = max(1, int(np.floor(len(valid_dates) * EARLIER_BATCH_FRACTION))) - 1
        cutoff = pd.Timestamp(valid_dates[cutoff_position])
        earlier_mask = aligned_time <= cutoff
        later_mask = aligned_time > cutoff
        earlier = cleaned.loc[earlier_mask].copy()
        later = cleaned.loc[later_mask].copy()
        mode = "real_chronological_split"
        description = (
            f"Real date-based split: earlier dates <= {cutoff.date()}, later dates > {cutoff.date()}."
        )
    else:
        # STAND-IN ONLY: this random split simulates incoming batches when no
        # genuine timestamp exists. It must never be described as temporal drift.
        earlier, later = train_test_split(
            cleaned,
            test_size=1.0 - EARLIER_BATCH_FRACTION,
            random_state=RANDOM_SEED,
            stratify=cleaned[target],
        )
        earlier = earlier.copy()
        later = later.copy()
        cutoff = None
        mode = "random_batch_STAND_IN_simulation"
        description = (
            "STAND-IN random batch simulation because no usable time field was available; "
            "this is not evidence of real temporal drift."
        )
    if earlier.empty or later.empty:
        raise ValueError("Adaptive split produced an empty earlier or later batch.")
    return {
        "earlier": earlier,
        "later": later,
        "mode": mode,
        "description": description,
        "cutoff": cutoff,
    }


def evaluate_at_threshold(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Compute the Phase 3 metrics using the fixed Phase 3b cutoff."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def load_phase3b_reference(
    threshold_table_path: str | Path,
) -> tuple[str, float, float]:
    """Read the selected model, operational threshold, and tuned reference F1."""
    table = pd.read_csv(threshold_table_path)
    required = {"model", "tuned_threshold", "tuned_f1"}
    if not required.issubset(table.columns):
        raise ValueError(f"Threshold table must contain {sorted(required)}.")
    selected = table.sort_values(["tuned_f1", "roc_auc"], ascending=False).iloc[0]
    return str(selected["model"]), float(selected["tuned_threshold"]), float(selected["tuned_f1"])


def _comparison_row(
    stage: str,
    model_state: str,
    evaluation_scope: str,
    train_rows: int,
    eval_rows: int,
    threshold: float,
    metrics: dict[str, float],
    triggered: bool,
    trigger_reason: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model_state": model_state,
        "evaluation_scope": evaluation_scope,
        "train_rows": train_rows,
        "evaluation_rows": eval_rows,
        "threshold": threshold,
        **metrics,
        "retrain_triggered": triggered,
        "trigger_reason": trigger_reason,
    }


def run_adaptive_experiment(
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
) -> dict[str, Any]:
    """Run the initial chronological evaluation and conditional retraining path."""
    result_path = Path(results_dir)
    raw, csv_path = load_dataset(data_dir)
    target, leakage = load_phase1_decisions(result_path / "dataset_summary.txt")
    cleaned, _, _ = clean_dataset(raw, target)
    roles = identify_feature_roles(cleaned, target, leakage)
    time_column, parsed_time, time_reports = find_usable_time_column(
        cleaned, result_path / "dataset_summary.txt"
    )
    batches = create_adaptive_batches(cleaned, target, parsed_time)
    model_name, threshold, reference_f1 = load_phase3b_reference(
        result_path / "tables" / "threshold_tuning.csv"
    )
    if model_name != "Logistic Regression":
        raise ValueError(
            f"This run expected the Phase 3b selected Logistic Regression, found {model_name!r}."
        )

    earlier = batches["earlier"]
    later = batches["later"]
    X_earlier = earlier.drop(columns=[target])
    y_earlier = earlier[target]
    X_later = later.drop(columns=[target])
    y_later = later[target]
    initial_model = build_model_pipelines(roles)[model_name]
    initial_model.fit(X_earlier, y_earlier)
    adaptive_feature_count = len(
        initial_model.named_steps["preprocessor"].get_feature_names_out()
    )
    later_metrics = evaluate_at_threshold(initial_model, X_later, y_later, threshold)

    trigger_floor = reference_f1 - F1_DROP_TRIGGER
    triggered = bool(later_metrics["f1"] < trigger_floor)
    trigger_reason = (
        f"later F1 {later_metrics['f1']:.6f} "
        f"{'<' if triggered else '>='} reference F1 {reference_f1:.6f} - "
        f"trigger drop {F1_DROP_TRIGGER:.2f} = {trigger_floor:.6f}"
    )
    before_row = _comparison_row(
        "before_adaptive_decision",
        "initial_model_trained_on_earlier_batch",
        "later_batch",
        len(earlier),
        len(later),
        threshold,
        later_metrics,
        triggered,
        trigger_reason,
    )

    retrained_model = None
    if triggered:
        # The combined chronological pool is re-split so the after metrics come
        # from rows excluded from retraining. This avoids in-sample evaluation.
        X_combined = cleaned.drop(columns=[target])
        y_combined = cleaned[target]
        X_update, X_holdout, y_update, y_holdout = train_test_split(
            X_combined,
            y_combined,
            test_size=COMBINED_HOLDOUT_SIZE,
            random_state=RANDOM_SEED,
            stratify=y_combined,
        )
        retrained_model = build_model_pipelines(roles)[model_name]
        retrained_model.fit(X_update, y_update)
        after_metrics = evaluate_at_threshold(retrained_model, X_holdout, y_holdout, threshold)
        after_row = _comparison_row(
            "after_adaptive_decision",
            "retrained_on_combined_pool_excluding_holdout",
            "combined_stratified_holdout",
            len(X_update),
            len(X_holdout),
            threshold,
            after_metrics,
            True,
            trigger_reason,
        )
    else:
        # No retraining is performed. The after-decision state intentionally
        # repeats the measured later-batch metrics for the retained model.
        after_metrics = later_metrics.copy()
        after_row = _comparison_row(
            "after_adaptive_decision",
            "initial_model_retained_no_retrain",
            "same_later_batch_no_retrain",
            len(earlier),
            len(later),
            threshold,
            after_metrics,
            False,
            trigger_reason,
        )

    comparison = pd.DataFrame([before_row, after_row])
    return {
        "csv_path": csv_path,
        "target": target,
        "roles": roles,
        "time_column": time_column,
        "time_reports": time_reports,
        "batches": batches,
        "model_name": model_name,
        "threshold": threshold,
        "reference_f1": reference_f1,
        "trigger_floor": trigger_floor,
        "triggered": triggered,
        "trigger_reason": trigger_reason,
        "initial_model": initial_model,
        "adaptive_feature_count": adaptive_feature_count,
        "retrained_model": retrained_model,
        "comparison": comparison,
    }


def plot_adaptive_comparison(
    comparison: pd.DataFrame,
    triggered: bool,
    output_path: str | Path,
) -> Path:
    """Save a before/after grouped metric chart without implying improvement."""
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_data = comparison.melt(
        id_vars="stage",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_data, x="metric", y="score", hue="stage", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    decision = "retraining triggered" if triggered else "no retraining triggered; initial model retained"
    ax.set_title(f"Adaptive Experiment Before/After Comparison\n{decision}")
    ax.legend(title="Experiment stage", loc="lower right")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def render_adaptive_summary(
    experiment: dict[str, Any],
    table_path: Path,
    figure_path: Path,
) -> str:
    """Render the actual split, trigger decision, and unmodified metrics."""
    batches = experiment["batches"]
    lines = [
        "PHASE 6 - ADAPTIVE EXPERIMENT SUMMARY",
        "=" * 37,
        f"Source CSV: {experiment['csv_path'].resolve()}",
        f"Batch mode: {batches['mode']}",
        f"Batch description: {batches['description']}",
        f"Phase 1-confirmed time column: {experiment['time_column']}",
        f"Earlier rows: {len(batches['earlier'])}",
        f"Later rows: {len(batches['later'])}",
        f"Model family: {experiment['model_name']}",
        f"Features learned from earlier batch only: {experiment['adaptive_feature_count']}",
        f"Fixed operational threshold: {experiment['threshold']:.6f}",
        f"Phase 3b reference F1: {experiment['reference_f1']:.6f}",
        f"Retrain trigger: later F1 < reference F1 - {F1_DROP_TRIGGER:.2f}",
        f"Trigger floor: {experiment['trigger_floor']:.6f}",
        f"Trigger result: {experiment['triggered']}",
        f"Reason: {experiment['trigger_reason']}",
        "",
        "BEFORE/AFTER TABLE",
        "------------------",
        experiment["comparison"].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "No improvement is assumed: the table reports the observed values, including an unchanged after-state when retraining is not triggered.",
        f"Saved table: {table_path.resolve()}",
        f"Saved chart: {figure_path.resolve()}",
    ]
    if batches["mode"] == "random_batch_STAND_IN_simulation":
        lines.append(
            "IMPORTANT: This random split is a STAND-IN simulation and is not evidence of real temporal drift."
        )
    return "\n".join(lines) + "\n"


def run_phase6(
    data_dir: str | Path = "data",
    results_dir: str | Path = "results",
) -> dict[str, Any]:
    """Run and save the complete Phase 6 adaptive experiment."""
    result_path = Path(results_dir)
    experiment = run_adaptive_experiment(data_dir, result_path)
    table_path = result_path / "tables" / "adaptive_before_after.csv"
    experiment["comparison"].to_csv(table_path, index=False, float_format="%.8f")
    figure_path = plot_adaptive_comparison(
        experiment["comparison"],
        experiment["triggered"],
        result_path / "figures" / "adaptive_before_after.png",
    )
    summary = render_adaptive_summary(experiment, table_path, figure_path)
    summary_path = result_path / "adaptive_experiment_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Batch mode: {experiment['batches']['mode']}")
    print(experiment["batches"]["description"])
    print(f"Phase 1-confirmed time column: {experiment['time_column']}")
    print(f"Retrain trigger: later F1 < reference F1 - {F1_DROP_TRIGGER:.2f}")
    print(f"Trigger result: {experiment['triggered']} ({experiment['trigger_reason']})")
    print("\nObserved before/after comparison:")
    print(
        experiment["comparison"].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\nSummary written to: {summary_path.resolve()}")
    return {
        **experiment,
        "table_path": table_path,
        "figure_path": figure_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_phase6()
