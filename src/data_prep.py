"""Load, inspect, and prepare the Kaggle e-commerce CSV.

Phase 1 functions inspect the raw data. Phase 2 functions clean it and construct
model-specific preprocessing pipelines. No train/test split or model is fitted
here. Column decisions are inferred from the loaded schema and the persisted
Phase 1 report instead of assuming dataset-specific field names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_SEED = 42
HIGH_MISSING_THRESHOLD = 40.0
HIGH_CARDINALITY_ID_RATIO = 0.05
DURATION_ORDER = ("very short", "short", "medium", "long", "very long")

# Generic language used to find purchase/conversion outcome fields. These are
# concepts, not assumed dataset column names.
TARGET_TERMS = {
    "purchase": 6,
    "purchased": 6,
    "conversion": 6,
    "converted": 6,
    "order": 5,
    "ordered": 5,
    "checkout": 4,
    "sale": 4,
    "sold": 4,
    "transaction": 4,
    "outcome": 3,
    "abandon": 3,
    "abandoned": 3,
    "completed": 3,
    "success": 3,
}

# Generic post-event concepts used only after a target has been selected.
LEAKAGE_RULES = (
    (
        {"revenue", "amount_paid", "paid_amount", "order_value", "sales_value", "gmv"},
        "post-purchase monetary outcome; its value is produced by a completed transaction",
    ),
    (
        {"review", "rating", "feedback", "helpful_vote", "helpful_votes"},
        "post-purchase feedback; it can only be recorded after the customer outcome",
    ),
    (
        {"refund", "returned", "return", "delivered", "delivery", "shipped", "shipping"},
        "post-order fulfilment outcome; it occurs after the prediction point",
    ),
    (
        {"payment", "payment_method", "payment_status"},
        "checkout/payment-stage information; conservatively unavailable before the purchase decision",
    ),
    (
        {"abandon", "abandoned", "cancelled", "canceled"},
        "competing end-of-session outcome; it directly reveals how the session ended",
    ),
    (
        {"purchase_date", "order_date", "transaction_date", "invoice", "order_id", "transaction_id"},
        "post-event record or identifier created by the transaction",
    ),
)


def discover_csv(data_dir: str | Path) -> Path:
    """Return the only CSV under *data_dir*, failing clearly if ambiguous."""
    data_path = Path(data_dir)
    csv_files = sorted(path for path in data_path.glob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV file was found in {data_path.resolve()}.")
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise ValueError(f"Expected one source CSV in {data_path.resolve()}, found: {names}")
    return csv_files[0]


def load_dataset(data_dir: str | Path = "data") -> tuple[pd.DataFrame, Path]:
    """Discover and load the real CSV without changing its values or dtypes."""
    csv_path = discover_csv(data_dir)
    return pd.read_csv(csv_path), csv_path


def load_phase1_decisions(
    summary_path: str | Path = "results/dataset_summary.txt",
) -> tuple[str, list[str]]:
    """Read the fixed target and leakage exclusions persisted by Phase 1.

    Phase 2 must reuse these decisions rather than running target selection again.
    """
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 summary was not found at {path.resolve()}.")
    text = path.read_text(encoding="utf-8")

    target_match = re.search(
        r"CHOSEN TARGET\s*\n-+\s*\n([^:\n]+):",
        text,
    )
    leakage_match = re.search(
        r"LIKELY LEAKAGE EXCLUSION LIST\s*\n-+\s*\n(?P<body>.*)",
        text,
        flags=re.DOTALL,
    )
    if not target_match or not leakage_match:
        raise ValueError("The Phase 1 summary does not contain the expected decision sections.")

    target = target_match.group(1).strip()
    leakage = []
    for line in leakage_match.group("body").splitlines():
        item = re.match(r"^- ([^:]+):", line.strip())
        if item:
            leakage.append(item.group(1).strip())
    return target, leakage


def inspect_dataset(df: pd.DataFrame, sample_size: int = 5) -> dict[str, Any]:
    """Collect the requested structural checks and a reproducible row sample."""
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")
    actual_size = min(sample_size, len(df))
    sample = df.sample(n=actual_size, random_state=RANDOM_SEED)
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str),
        "missing_counts": df.isna().sum(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample": sample,
    }


def clean_dataset(
    df: pd.DataFrame,
    target: str,
    high_missing_threshold: float = HIGH_MISSING_THRESHOLD,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    """Remove duplicates and apply a documented missing-value policy.

    Policy: retain complete columns; drop rows with a missing target; drop a
    feature only when more than 40% is missing; otherwise use the median for
    numeric fields and the mode for categorical/text fields. The raw dataframe
    is never modified.
    """
    if target not in df.columns:
        raise KeyError(f"Fixed Phase 1 target {target!r} is absent from the dataset.")
    if not 0.0 <= high_missing_threshold <= 100.0:
        raise ValueError("high_missing_threshold must be between 0 and 100.")

    working = df.copy()
    duplicate_rows_before = int(working.duplicated().sum())
    working = working.drop_duplicates().copy()
    rows_after_duplicate_removal = len(working)
    decisions: list[dict[str, Any]] = []
    columns_to_drop: list[str] = []

    for column in working.columns:
        missing_count = int(working[column].isna().sum())
        missing_pct = 100.0 * missing_count / len(working) if len(working) else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(working[column])
        column_type = "numeric" if is_numeric else "categorical/text"

        if missing_count == 0:
            action = "retain (no imputation needed)"
            reason = "0.00% missing"
        elif column == target:
            before = len(working)
            working = working.loc[working[column].notna()].copy()
            removed = before - len(working)
            action = f"drop {removed} row(s) with missing target"
            reason = f"{missing_pct:.2f}% missing; target labels must not be imputed"
        elif missing_pct > high_missing_threshold:
            columns_to_drop.append(column)
            action = "drop column"
            reason = (
                f"{missing_pct:.2f}% missing exceeds the "
                f"{high_missing_threshold:.2f}% threshold"
            )
        elif is_numeric:
            median = working[column].median()
            working[column] = working[column].fillna(median)
            action = f"median imputation ({median:g})"
            reason = f"{missing_pct:.2f}% missing; median is robust to numeric outliers"
        else:
            modes = working[column].mode(dropna=True)
            if modes.empty:
                columns_to_drop.append(column)
                action = "drop column"
                reason = f"{missing_pct:.2f}% missing and no observed value is available"
            else:
                mode = modes.iloc[0]
                working[column] = working[column].fillna(mode)
                action = f"mode imputation ({mode!r})"
                reason = f"{missing_pct:.2f}% missing; mode preserves categorical meaning"

        decisions.append(
            {
                "column": str(column),
                "type": column_type,
                "missing_count": missing_count,
                "missing_percent": missing_pct,
                "action": action,
                "reason": reason,
            }
        )

    if columns_to_drop:
        working = working.drop(columns=columns_to_drop)

    # Imputation can theoretically make two rows identical, so check once more.
    duplicates_after_imputation = int(working.duplicated().sum())
    if duplicates_after_imputation:
        working = working.drop_duplicates().copy()

    duplicate_report = {
        "duplicates_found_initially": duplicate_rows_before,
        "rows_after_initial_removal": rows_after_duplicate_removal,
        "duplicates_created_by_imputation": duplicates_after_imputation,
        "final_rows": len(working),
    }
    return working, decisions, duplicate_report


def _looks_like_date(series: pd.Series) -> bool:
    """Return True when most non-null text values parse as dates."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
    return bool(parsed.notna().mean() >= 0.90)


def identify_feature_roles(
    df: pd.DataFrame,
    target: str,
    leakage_columns: list[str],
) -> dict[str, Any]:
    """Assign real columns to numeric, nominal, ordinal, or excluded roles."""
    if target not in df.columns:
        raise KeyError(f"Fixed Phase 1 target {target!r} is absent after cleaning.")

    roles: dict[str, Any] = {
        "numeric": [],
        "nominal": [],
        "ordinal": [],
        "ordinal_categories": [],
        "excluded": [],
        "decisions": [],
    }
    nominal_name_terms = {"type", "category", "channel", "weekday", "season", "month", "location"}

    for column in df.columns:
        if column == target:
            roles["decisions"].append(
                {"column": column, "role": "target", "encoding": "none", "reason": "fixed Phase 1 label"}
            )
            continue
        if column in leakage_columns:
            roles["excluded"].append(column)
            roles["decisions"].append(
                {
                    "column": column,
                    "role": "excluded leakage",
                    "encoding": "none",
                    "reason": "excluded using the persisted Phase 1 leakage decision",
                }
            )
            continue

        normalised, parts = _name_parts(column)
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / len(df) if len(df) else 0.0

        if "date" in parts and _looks_like_date(series):
            roles["excluded"].append(column)
            roles["decisions"].append(
                {
                    "column": column,
                    "role": "excluded raw date",
                    "encoding": "none",
                    "reason": "date-like text would create high-cardinality dummy variables; calendar fields are retained separately",
                }
            )
        elif "id" in parts and unique_ratio > HIGH_CARDINALITY_ID_RATIO:
            roles["excluded"].append(column)
            roles["decisions"].append(
                {
                    "column": column,
                    "role": "excluded identifier",
                    "encoding": "none",
                    "reason": f"identifier-like field with {unique_ratio:.2%} unique values could encourage memorisation",
                }
            )
        elif "bucket" in parts and not pd.api.types.is_numeric_dtype(series):
            value_lookup = {str(value).strip().lower(): value for value in series.dropna().unique()}
            ordered_values = [value_lookup[label] for label in DURATION_ORDER if label in value_lookup]
            if len(ordered_values) == unique_count:
                roles["ordinal"].append(column)
                roles["ordinal_categories"].append(ordered_values)
                roles["decisions"].append(
                    {
                        "column": column,
                        "role": "ordinal categorical",
                        "encoding": "ordinal",
                        "reason": f"labels have a natural duration order: {ordered_values}",
                    }
                )
            else:
                roles["nominal"].append(column)
                roles["decisions"].append(
                    {
                        "column": column,
                        "role": "nominal categorical",
                        "encoding": "one-hot",
                        "reason": "categorical labels were observed but no complete safe order was inferred",
                    }
                )
        elif (
            not pd.api.types.is_numeric_dtype(series)
            or bool(parts & nominal_name_terms)
            or ("id" in parts and unique_ratio <= HIGH_CARDINALITY_ID_RATIO)
        ):
            roles["nominal"].append(column)
            roles["decisions"].append(
                {
                    "column": column,
                    "role": "nominal categorical",
                    "encoding": "one-hot",
                    "reason": f"{unique_count} observed labels/codes have no defensible numeric order",
                }
            )
        else:
            roles["numeric"].append(column)
            roles["decisions"].append(
                {
                    "column": column,
                    "role": "numeric",
                    "encoding": "none for trees; standard scaling for logistic regression",
                    "reason": "quantitative or binary measurement retained as numeric",
                }
            )

    return roles


def build_model_preprocessors(
    roles: dict[str, Any],
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """Create unfitted tree and logistic-regression feature preprocessors.

    Both encode categories. Only the logistic-regression version standardises
    numeric fields; tree models receive their original numeric scale.
    """
    numeric_columns = roles["numeric"]
    nominal_columns = roles["nominal"]
    ordinal_columns = roles["ordinal"]

    numeric_tree = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    numeric_linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    nominal = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    tree_steps: list[tuple[str, Any, list[str]]] = []
    linear_steps: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        tree_steps.append(("numeric", numeric_tree, numeric_columns))
        linear_steps.append(("numeric", numeric_linear, numeric_columns))
    if nominal_columns:
        tree_steps.append(("nominal", nominal, nominal_columns))
        linear_steps.append(("nominal", nominal, nominal_columns))
    if ordinal_columns:
        ordinal = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=roles["ordinal_categories"],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        tree_steps.append(("ordinal", ordinal, ordinal_columns))
        linear_steps.append(("ordinal", ordinal, ordinal_columns))

    return (
        ColumnTransformer(tree_steps, remainder="drop"),
        ColumnTransformer(linear_steps, remainder="drop"),
    )


def _name_parts(column_name: object) -> tuple[str, set[str]]:
    """Normalise a discovered name and split it into comparable words."""
    normalised = re.sub(r"[^a-z0-9]+", "_", str(column_name).lower()).strip("_")
    return normalised, set(filter(None, normalised.split("_")))


def _matched_terms(column_name: object, vocabulary: set[str] | dict[str, int]) -> list[str]:
    """Match whole names, component words, and useful word stems."""
    normalised, parts = _name_parts(column_name)
    matches: list[str] = []
    for term in vocabulary:
        # Multi-word phrases must appear intact. This prevents a generic word
        # such as "amount" from making two different concepts look equivalent.
        if "_" in term:
            if term == normalised or term in normalised:
                matches.append(term)
        elif term == normalised or term in parts:
            matches.append(term)
        elif len(term) >= 5 and any(part.startswith(term) or term.startswith(part) for part in parts):
            matches.append(term)
    return sorted(set(matches))


def shortlist_target_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rank binary/categorical columns whose names describe purchase outcomes."""
    candidates: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        non_null_unique = int(series.nunique(dropna=True))
        is_binary = non_null_unique == 2
        is_categorical = bool(
            is_binary
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_object_dtype(series)
        )
        matches = _matched_terms(column, TARGET_TERMS)
        if not matches or not is_categorical:
            continue

        name_score = max(TARGET_TERMS[term] for term in matches)
        type_score = 3 if is_binary else 1
        kind = "binary" if is_binary else "categorical"
        candidates.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "unique_non_null": non_null_unique,
                "score": name_score + type_score,
                "reason": (
                    f"{kind} field with {non_null_unique} non-null values; "
                    f"name matched outcome term(s): {', '.join(matches)}"
                ),
            }
        )

    return sorted(candidates, key=lambda item: (-item["score"], str(item["column"])))


def choose_target(candidates: list[dict[str, Any]]) -> tuple[object, str]:
    """Choose the strongest reproducible candidate from the ranked shortlist."""
    if not candidates:
        raise ValueError("No binary/categorical purchase-outcome target candidate was found.")
    chosen = candidates[0]
    reason = (
        f"highest programmatic relevance score ({chosen['score']}); "
        f"{chosen['reason']}"
    )
    return chosen["column"], reason


def scan_likely_leakage(df: pd.DataFrame, target: object) -> list[dict[str, str]]:
    """Flag discovered feature names that imply information after the outcome."""
    if target not in df.columns:
        raise KeyError(f"Chosen target {target!r} is not present in the dataframe.")

    exclusions: list[dict[str, str]] = []
    for column in df.columns:
        if column == target:
            continue
        reasons: list[str] = []
        matched: set[str] = set()
        for vocabulary, reason in LEAKAGE_RULES:
            terms = _matched_terms(column, vocabulary)
            if terms:
                matched.update(terms)
                reasons.append(reason)
        if reasons:
            exclusions.append(
                {
                    "column": str(column),
                    "reason": f"{'; '.join(dict.fromkeys(reasons))} (matched: {', '.join(sorted(matched))})",
                }
            )
    return exclusions


def render_summary(
    csv_path: Path,
    inspection: dict[str, Any],
    candidates: list[dict[str, Any]],
    target: object,
    target_reason: str,
    leakage: list[dict[str, str]],
) -> str:
    """Render all Phase 1 findings as a readable plain-text report."""
    lines = [
        "PHASE 1 - DATASET INSPECTION SUMMARY",
        "=" * 38,
        f"Source CSV: {csv_path.resolve()}",
        f"Shape: {inspection['shape']}",
        "",
        "COLUMN NAMES",
        "------------",
        *[str(column) for column in inspection["columns"]],
        "",
        "DTYPES",
        "------",
        inspection["dtypes"].to_string(),
        "",
        "MISSING VALUES PER COLUMN",
        "-------------------------",
        inspection["missing_counts"].to_string(),
        "",
        f"Duplicate row count: {inspection['duplicate_rows']}",
        "",
        "REPRODUCIBLE 5-ROW SAMPLE (random seed 42)",
        "------------------------------------------",
        inspection["sample"].to_string(index=False),
        "",
        "PROGRAMMATIC TARGET SHORTLIST",
        "-----------------------------",
    ]
    if candidates:
        for candidate in candidates:
            lines.append(
                f"- {candidate['column']} [score={candidate['score']}, dtype={candidate['dtype']}]: "
                f"{candidate['reason']}"
            )
    else:
        lines.append("- No suitable candidates found.")

    lines.extend(
        [
            "",
            "CHOSEN TARGET",
            "-------------",
            f"{target}: {target_reason}",
            "",
            "LIKELY LEAKAGE EXCLUSION LIST",
            "-----------------------------",
        ]
    )
    if leakage:
        lines.extend(f"- {item['column']}: {item['reason']}" for item in leakage)
    else:
        lines.append("- No likely post-event leakage columns were detected by the name scan.")
    return "\n".join(lines) + "\n"


def write_summary(summary: str, output_path: str | Path = "results/dataset_summary.txt") -> Path:
    """Write the inspection report, creating its result directory if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")
    return path


def run_phase1(
    data_dir: str | Path = "data",
    output_path: str | Path = "results/dataset_summary.txt",
) -> dict[str, Any]:
    """Run, print, and save the complete Phase 1 inspection."""
    np.random.seed(RANDOM_SEED)
    df, csv_path = load_dataset(data_dir)
    inspection = inspect_dataset(df)
    candidates = shortlist_target_candidates(df)
    target, target_reason = choose_target(candidates)
    leakage = scan_likely_leakage(df, target)
    summary = render_summary(csv_path, inspection, candidates, target, target_reason, leakage)
    saved_path = write_summary(summary, output_path)
    print(summary)
    print(f"Summary written to: {saved_path.resolve()}")
    return {
        "dataframe": df,
        "csv_path": csv_path,
        "inspection": inspection,
        "candidates": candidates,
        "target": target,
        "target_reason": target_reason,
        "leakage": leakage,
        "summary_path": saved_path,
    }


if __name__ == "__main__":
    run_phase1()
