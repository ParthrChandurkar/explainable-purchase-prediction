# Explainable Purchase Prediction

Undergraduate software project built from the fixed Kaggle dataset **Indian E-Commerce Customer Behavior & Purchase**.

Phase 1 contains reproducible dataset loading, structural inspection, programmatic target shortlisting, and likely leakage detection.

Phase 2 adds documented cleaning, categorical encoding, separate tree/logistic-regression preprocessing views, IQR outlier reporting, class-balance checks, and saved EDA figures. Run it from the project root with:

```powershell
python -m src.eda
```

Phase 3 audits high-cardinality encoding, creates a fixed stratified split, trains class-balanced Logistic Regression and Random Forest pipelines, evaluates held-out predictions, and saves the best complete pipeline. Run it with:

```powershell
python -m src.evaluate
```

SHAP explainability is intentionally deferred to a later phase.
