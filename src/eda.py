"""Phase 2: reproducible cleaning, preprocessing design, and EDA figures.

The module uses the target and leakage exclusions saved by Phase 1. It creates
plots and reports, but deliberately trains no predictive model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_prep import (
    RANDOM_SEED,
    build_model_preprocessors,
    clean_dataset,
    identify_feature_roles,
    load_dataset,
    load_phase1_decisions,
)


BEHAVIOUR_TERMS = {
    "cart": 8,
    "pages": 7,
    "page": 7,
    "duration": 7,
    "time": 7,
    "quantity": 6,
    "discount": 5,
    "price": 4,
    "viewed": 4,
}


def _parts(column: object) -> set[str]:
    normalised = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
    return set(filter(None, normalised.split("_")))


def _behaviour_score(column: object) -> int:
    parts = _parts(column)
    matches = [score for term, score in BEHAVIOUR_TERMS.items() if term in parts]
    return max(matches, default=0)


def print_missing_decisions(decisions: list[dict[str, Any]]) -> None:
    """Print one concise, viva-friendly missing-value decision per column."""
    print("Missing-value decisions by column:")
    for item in decisions:
        print(
            f"- {item['column']} [{item['type']}]: {item['action']} - "
            f"{item['reason']}"
        )


def print_encoding_decisions(roles: dict[str, Any]) -> None:
    """Print why every field is encoded, retained, or excluded."""
    print("Feature-role and encoding decisions:")
    for item in roles["decisions"]:
        print(
            f"- {item['column']}: {item['role']}; {item['encoding']} - "
            f"{item['reason']}"
        )


def class_balance_summary(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Calculate exact class counts, percentages, and a practical interpretation."""
    counts = df[target].value_counts(dropna=False).sort_index()
    percentages = counts.div(len(df)).mul(100.0)
    if len(counts) < 2:
        ratio = float("inf")
        interpretation = "Only one target class is present; classification would not be possible."
    else:
        smallest = int(counts.min())
        largest = int(counts.max())
        ratio = largest / smallest if smallest else float("inf")
        minority_percent = float(percentages.min())
        if minority_percent >= 40.0:
            level = "roughly balanced"
        elif minority_percent >= 20.0:
            level = "moderately imbalanced"
        else:
            level = "strongly imbalanced"
        interpretation = (
            f"The target is {level} (majority:minority ratio {ratio:.2f}:1). "
            "Later evaluation should use stratified splits and metrics beyond accuracy."
        )
    return {
        "counts": counts,
        "percentages": percentages,
        "majority_to_minority_ratio": ratio,
        "interpretation": interpretation,
    }


def print_class_balance(balance: dict[str, Any]) -> None:
    """Print the observed class distribution and its modeling implication."""
    print("Target class balance:")
    for label, count in balance["counts"].items():
        print(f"- class {label}: {int(count)} rows ({balance['percentages'].loc[label]:.2f}%)")
    print(balance["interpretation"])


def select_key_numeric_columns(df: pd.DataFrame, roles: dict[str, Any]) -> list[str]:
    """Find quantitative behavioural fields for an IQR outlier audit."""
    scored = []
    for column in roles["numeric"]:
        score = _behaviour_score(column)
        if score and pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() > 2:
            scored.append((column, score))
    return [column for column, _ in sorted(scored, key=lambda item: (-item[1], item[0]))]


def iqr_outlier_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Report 1.5×IQR outliers without deleting or changing any observation."""
    records = []
    for column in columns:
        series = df[column].dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())
        records.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": count,
                "outlier_percent": 100.0 * count / len(series) if len(series) else 0.0,
                "decision": "retain; report only pending domain/model justification",
            }
        )
    return pd.DataFrame.from_records(records)


def select_behaviour_columns(
    df: pd.DataFrame,
    target: str,
    roles: dict[str, Any],
    maximum: int = 4,
) -> list[str]:
    """Select 2–4 actual pre-outcome behavioural fields by semantic score."""
    eligible = roles["numeric"] + roles["nominal"] + roles["ordinal"]
    scored = [(column, _behaviour_score(column)) for column in eligible if column != target]
    selected = [
        column
        for column, score in sorted(scored, key=lambda item: (-item[1], item[0]))
        if score > 0
    ][:maximum]
    if len(selected) < 2:
        raise ValueError("Fewer than two suitable behavioural columns were discovered.")
    return selected


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_target_distribution(
    df: pd.DataFrame,
    target: str,
    output_dir: str | Path,
) -> Path:
    """Save the observed class counts for the fixed target."""
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str), legend=False, ax=ax)
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}\n({value / len(df):.1%})", ha="center", va="bottom")
    ax.set(title=f"Class Distribution: {target}", xlabel=target, ylabel="Row count")
    ax.set_ylim(0, float(counts.max()) * 1.16)
    fig.tight_layout()
    return _save_figure(fig, Path(output_dir) / "target_class_distribution.png")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target: str,
    roles: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Save correlations for non-leakage quantitative fields and the target."""
    columns = list(dict.fromkeys(roles["numeric"] + [target]))
    correlation = df[columns].corr(numeric_only=True)
    size = max(8, min(14, len(columns)))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Leakage Fields Excluded)")
    fig.tight_layout()
    return _save_figure(fig, Path(output_dir) / "correlation_heatmap.png")


def plot_behaviour_vs_target(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    roles: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Save one readable target comparison for each discovered behavioural field."""
    paths = []
    for column in columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        is_categorical = not pd.api.types.is_numeric_dtype(df[column]) or df[column].nunique() <= 10
        if is_categorical:
            plot_data = df[[column, target]].copy()
            if column in roles["ordinal"]:
                position = roles["ordinal"].index(column)
                category_order = roles["ordinal_categories"][position]
                plot_data[column] = pd.Categorical(
                    plot_data[column], categories=category_order, ordered=True
                )
            rates = (
                plot_data.groupby(column, observed=True)[target]
                .agg(["mean", "count"])
                .reset_index()
            )
            sns.barplot(data=rates, x=column, y="mean", hue=column, legend=False, ax=ax)
            for patch, row in zip(ax.patches, rates.itertuples(index=False)):
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_height(),
                    f"{patch.get_height():.1%}\n(n={row.count:,})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax.set_ylabel(f"Mean {target} (observed target rate)")
            ax.set_ylim(0, min(1.0, max(0.05, float(rates["mean"].max()) * 1.25)))
            ax.tick_params(axis="x", rotation=25)
        else:
            sns.boxplot(data=df, x=target, y=column, hue=target, legend=False, ax=ax)
            ax.set_ylabel(column)
        ax.set_title(f"{column} vs {target}")
        fig.tight_layout()
        safe_name = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        paths.append(_save_figure(fig, Path(output_dir) / f"behaviour_{safe_name}_vs_target.png"))
    return paths


def render_phase2_summary(
    csv_path: Path,
    target: str,
    leakage: list[str],
    duplicate_report: dict[str, int],
    missing_decisions: list[dict[str, Any]],
    roles: dict[str, Any],
    encoded_shapes: dict[str, tuple[int, int]],
    outliers: pd.DataFrame,
    balance: dict[str, Any],
    behaviour_columns: list[str],
    figure_paths: list[Path],
) -> str:
    """Render all actual Phase 2 decisions and findings as plain text."""
    lines = [
        "PHASE 2 - EDA AND PREPROCESSING SUMMARY",
        "=" * 39,
        f"Source CSV: {csv_path.resolve()}",
        f"Fixed Phase 1 target: {target}",
        f"Persisted Phase 1 leakage exclusions: {leakage}",
        "",
        "DUPLICATES",
        "----------",
        f"Initially detected and removed: {duplicate_report['duplicates_found_initially']}",
        f"Created by imputation and removed: {duplicate_report['duplicates_created_by_imputation']}",
        f"Final row count: {duplicate_report['final_rows']}",
        "",
        "MISSING-VALUE DECISIONS",
        "-----------------------",
    ]
    lines.extend(
        f"- {item['column']} [{item['type']}]: {item['action']} - {item['reason']}"
        for item in missing_decisions
    )
    lines.extend(["", "ENCODING AND FEATURE-ROLE DECISIONS", "-----------------------------------"])
    lines.extend(
        f"- {item['column']}: {item['role']}; {item['encoding']} - {item['reason']}"
        for item in roles["decisions"]
    )
    lines.extend(
        [
            "",
            "MODEL-SPECIFIC FEATURE VIEWS",
            "----------------------------",
            f"Tree view shape: {encoded_shapes['tree']} (numeric fields not scaled)",
            f"Logistic-regression view shape: {encoded_shapes['linear']} (numeric fields standard-scaled)",
            "The dimension check uses the complete cleaned data only for Phase 2 verification; later modeling must fit preprocessing on training folds.",
            "",
            "IQR OUTLIER CHECK (NO ROWS REMOVED)",
            "-----------------------------------",
            outliers.to_string(index=False) if not outliers.empty else "No eligible numeric columns were found.",
            "",
            "CLASS BALANCE",
            "-------------",
        ]
    )
    for label, count in balance["counts"].items():
        lines.append(f"- class {label}: {int(count)} rows ({balance['percentages'].loc[label]:.2f}%)")
    lines.extend(
        [
            balance["interpretation"],
            "",
            f"Behaviour fields selected for plots: {behaviour_columns}",
            "Saved figures:",
            *[f"- {path.resolve()}" for path in figure_paths],
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase2(
    data_dir: str | Path = "data",
    phase1_summary_path: str | Path = "results/dataset_summary.txt",
    results_dir: str | Path = "results",
) -> dict[str, Any]:
    """Run the complete Phase 2 workflow, print decisions, and save outputs."""
    sns.set_theme(style="whitegrid", context="notebook")
    df, csv_path = load_dataset(data_dir)
    target, leakage = load_phase1_decisions(phase1_summary_path)
    missing_decisions_in_data = [column for column in [target, *leakage] if column not in df.columns]
    if missing_decisions_in_data:
        raise KeyError(f"Phase 1 decision columns absent from CSV: {missing_decisions_in_data}")

    cleaned, missing_decisions, duplicate_report = clean_dataset(df, target)
    roles = identify_feature_roles(cleaned, target, leakage)
    X = cleaned.drop(columns=[target])
    y = cleaned[target].copy()

    # Fit temporary views only to verify that both encoders work and report the
    # real dimensions. Fresh, unfitted preprocessors are returned for modeling.
    tree_demo, linear_demo = build_model_preprocessors(roles)
    tree_matrix = tree_demo.fit_transform(X)
    linear_matrix = linear_demo.fit_transform(X)
    encoded_shapes = {"tree": tree_matrix.shape, "linear": linear_matrix.shape}
    tree_preprocessor, linear_preprocessor = build_model_preprocessors(roles)

    key_numeric = select_key_numeric_columns(cleaned, roles)
    outliers = iqr_outlier_report(cleaned, key_numeric)
    balance = class_balance_summary(cleaned, target)
    behaviour_columns = select_behaviour_columns(cleaned, target, roles)

    result_path = Path(results_dir)
    figure_dir = result_path / "figures"
    figure_paths = [
        plot_target_distribution(cleaned, target, figure_dir),
        plot_correlation_heatmap(cleaned, target, roles, figure_dir),
        *plot_behaviour_vs_target(cleaned, target, behaviour_columns, roles, figure_dir),
    ]

    tables_dir = result_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing_decisions).to_csv(tables_dir / "missing_value_decisions.csv", index=False)
    pd.DataFrame(roles["decisions"]).to_csv(tables_dir / "encoding_decisions.csv", index=False)
    outliers.to_csv(tables_dir / "outlier_report.csv", index=False)

    print(f"Fixed Phase 1 target: {target}")
    print(f"Persisted leakage exclusions: {leakage}")
    print_missing_decisions(missing_decisions)
    print(
        f"Duplicate rows removed: {duplicate_report['duplicates_found_initially']} initially, "
        f"{duplicate_report['duplicates_created_by_imputation']} after imputation."
    )
    print_encoding_decisions(roles)
    print(f"Tree encoded shape (unscaled numeric): {tree_matrix.shape}")
    print(f"Logistic encoded shape (scaled numeric): {linear_matrix.shape}")
    print("IQR outlier findings (rows retained):")
    print(outliers.to_string(index=False))
    print_class_balance(balance)
    print(f"Behaviour columns plotted: {behaviour_columns}")

    summary = render_phase2_summary(
        csv_path,
        target,
        leakage,
        duplicate_report,
        missing_decisions,
        roles,
        encoded_shapes,
        outliers,
        balance,
        behaviour_columns,
        figure_paths,
    )
    summary_path = result_path / "eda_preprocessing_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Phase 2 summary written to: {summary_path.resolve()}")

    return {
        "raw": df,
        "cleaned": cleaned,
        "X": X,
        "y": y,
        "target": target,
        "leakage": leakage,
        "missing_decisions": missing_decisions,
        "duplicate_report": duplicate_report,
        "roles": roles,
        "tree_preprocessor": tree_preprocessor,
        "linear_preprocessor": linear_preprocessor,
        "encoded_shapes": encoded_shapes,
        "outliers": outliers,
        "balance": balance,
        "behaviour_columns": behaviour_columns,
        "figure_paths": figure_paths,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_phase2()
