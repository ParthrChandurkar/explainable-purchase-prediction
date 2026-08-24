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

Phase 3b tunes each model's probability cutoff by maximum held-out F1, saves a default-versus-tuned comparison, and stores the selected cutoff with the best pipeline artifact.

Phase 4 loads that artifact without retraining and produces global and local SHAP explanations on the corrected 52-feature representation. Run it with:

```powershell
python -m src.explain
```

Business-insight analysis is intentionally deferred to a later phase.
