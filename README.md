# Explainable Purchase Prediction

This Python software project builds a reproducible purchase-prediction workflow for the Kaggle dataset **Indian E-Commerce Customer Behavior & Purchase**. It inspects and cleans the data, trains class-balanced Logistic Regression and Random Forest models, tunes the probability threshold, generates SHAP explanations, maps predictions to transparent retail-action suggestions, and runs an offline adaptive-model experiment. All reported values are calculated from the downloaded CSV when the code runs.

## Dataset setup

1. Download **Indian E-Commerce Customer Behavior & Purchase** from [Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/indian-e-commerce-customer-behavior-and-purchase).
2. Extract the downloaded archive.
3. Place its CSV file in `data/`. The current Kaggle download is named `Ecommerce.csv`.

The loader discovers the CSV inside `data/`; model code does not fabricate a substitute dataset. Keep only the intended dataset CSV in that directory so discovery is unambiguous.

## Installation and complete run

Python 3.11 is recommended. From the project root:

```powershell
python -m pip install -r requirements.txt
python main.py
```

`main.py` runs the existing phase modules in order:

1. dataset inspection and Phase 1 decisions;
2. cleaning, preprocessing, and EDA;
3. model training, evaluation, and threshold correction;
4. SHAP explanations;
5. rule-based business insights;
6. the adaptive experiment;
7. final artifact validation.

Each module can still be run independently with commands such as `python -m src.data_prep`, `python -m src.eda`, or `python -m src.adaptive`.

## Optional results dashboard

After `python main.py` has populated `results/`, launch the read-only Streamlit dashboard with:

```powershell
streamlit run dashboard/app.py
```

The dashboard reads only the saved summaries, tables, prediction examples, and figures. It does not load the training dataset, retrain a model, call an external service, or modify pipeline outputs. Its six tabs cover the dataset overview, customer analytics, prediction results, SHAP explanations, business insights, and the adaptive comparison.

## Generated outputs

`results/` is rebuilt from the real dataset and contains:

- `dataset_summary.txt`: dataset shape, names, dtypes, missing values, duplicates, sample rows, selected target, and leakage exclusions.
- `eda_preprocessing_summary.txt`: missing-value, encoding, duplicate, outlier, and class-balance decisions.
- `modeling_evaluation_summary.txt`: train/test setup, model metrics, and threshold correction.
- `shap_explainability_summary.txt`: explainer details and global/local SHAP results.
- `business_insights_summary.txt`: action counts, rule description, and precision caveat.
- `adaptive_experiment_summary.txt`: batch construction, trigger decision, and observed before/after metrics.
- `figures/`: EDA plots, default and tuned confusion matrices, SHAP global/local plots, and the adaptive comparison chart.
- `tables/`: cleaning and encoding audits, outlier findings, model comparison, threshold tuning, local SHAP examples, full business insights, and adaptive before/after results.
- `predictions/sample_predictions_with_explanations.csv`: real high-probability, low-probability, and threshold-borderline test predictions with actual labels and their top positive and negative SHAP contributions.

`models/best_model.joblib` stores the complete fitted preprocessing/model pipeline together with its tuned decision threshold.

## Notebook walkthrough

For a faculty demonstration, open these executed notebooks in order:

1. `notebooks/01_data_understanding.ipynb` - loading, inspection, target selection, and leakage checks.
2. `notebooks/02_eda_preprocessing.ipynb` - cleaning, encoding, outliers, class balance, and EDA.
3. `notebooks/03_modeling_evaluation.ipynb` - corrected feature encoding, model training, evaluation, and threshold tuning.
4. `notebooks/04_explainability_shap.ipynb` - global and local SHAP explanations.
5. `notebooks/05_business_insights.ipynb` - transparent action rules and their precision caveat.
6. `notebooks/06_adaptive_experiment.ipynb` - chronological batch evaluation and conditional retraining.

Each notebook imports the corresponding `src/` code and saves real executed output rather than duplicating the implementation as static text.

## Phase 6 scope: real time field versus simulation

The current dataset contains the Phase 1-confirmed `visit_date` field. Therefore, the recorded Phase 6 run uses a **real chronological split**: earlier visit dates train the initial model and later visit dates act as an incoming offline batch. The batch split used for the saved results is not random or simulated.

The module includes a fallback only for a dataset copy where no usable time field can be confirmed. That fallback creates a fixed random batch split and labels it `random_batch_STAND_IN_simulation`. It is a simulation, not evidence of temporal drift.

Even when `visit_date` is used, Phase 6 remains an offline classroom experiment over a fixed historical CSV. It is not a production monitoring system and does not claim to detect live drift.
