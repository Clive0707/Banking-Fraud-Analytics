"""
Fraud-classifier benchmark.

What changed from the original trainer
-------------------------------------
1. **Five model families instead of three.** Logistic Regression, CART and
   Random Forest are joined by XGBoost and LightGBM, the two gradient-boosting
   families a fraud team would actually reach for.

2. **The best model is chosen on PR-AUC, not F1 at an arbitrary 0.5 cut.**
   Under 0.96% prevalence, average precision is the honest ranking summary.

3. **Every estimator is a self-contained `Pipeline`.** Scaling now travels
   inside the artefact, so the serving path no longer needs to remember which
   model wants a scaler -- a special case that was previously hardcoded by model
   name in the Flask service.

4. **One-hot categoricals instead of `LabelEncoder`.** Label encoding imposed a
   meaningless ordering on `location`, which made the 5x international fraud
   lift invisible to the linear model.

5. **Customer baselines are fitted on the training split only.** The deviation
   features ("is this amount unusual *for this customer*") are powerful, but a
   baseline computed over the whole dataset would absorb the test rows. Fitting
   on train only keeps the evaluation clean.

6. **Thresholds are tuned, not assumed**, and the cost curve reports what each
   operating point is worth in rupees.

7. **Cross-validation** on PR-AUC gives each headline number a variance estimate.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src import features as feat_mod
from src.fraud_detection import metrics as metric_mod
from src.fraud_detection.leakage_analysis import run_leakage_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cross-validation is run on a subsample: five folds of five estimators over
# 400k rows adds little precision to the variance estimate for a lot of minutes.
CV_SUBSAMPLE = 120_000

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression",
    "CART Decision Tree": "cart_decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}


def build_model_zoo(scale_pos_weight):
    """
    Define the benchmark. Every entry is a Pipeline so the saved artefact can
    score a raw feature frame without external preprocessing state.
    """
    seed = config.RANDOM_SEED

    zoo = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]),
        "CART Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                           class_weight="balanced", random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=20,
                                           class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr", tree_method="hist",
                random_state=seed, n_jobs=-1,
            )),
        ])
    except ImportError:
        logger.warning("xgboost not installed -- skipping XGBoost from the benchmark.")

    try:
        from lightgbm import LGBMClassifier
        zoo["LightGBM"] = Pipeline([
            ("clf", LGBMClassifier(
                n_estimators=400, num_leaves=63, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
            )),
        ])
    except ImportError:
        logger.warning("lightgbm not installed -- skipping LightGBM from the benchmark.")

    return zoo


def drop_low_support_features(X, y, feature_names):
    """
    Remove binary features with too few examples to estimate a coefficient.

    A 0/1 column that is set on a handful of rows -- especially with no positive
    cases among them -- carries no information the model can use, but logistic
    regression will still assign it a weight. On this dataset `intl_x_high` was
    set on 7 of 499,684 rows with zero frauds, and the fitted coefficient of
    -9.75 swamped every other term for international high-value transactions,
    which the full 15M shows to be the single riskiest segment.

    Continuous features are left alone; the concern is empty indicator cells.
    """
    kept, dropped = [], []
    y = np.asarray(y).astype(int)

    for col in feature_names:
        values = X[col].values
        is_binary = np.isin(np.unique(values), (0.0, 1.0)).all()
        if not is_binary:
            kept.append(col)
            continue

        present = values == 1
        support = int(present.sum())
        positives = int(y[present].sum())
        if support < config.MIN_FEATURE_SUPPORT or positives < config.MIN_FEATURE_POSITIVES:
            dropped.append({
                "feature": col,
                "support": support,
                "positives": positives,
                "reason": (
                    f"only {support} rows set ({positives} fraudulent); needs "
                    f"{config.MIN_FEATURE_SUPPORT} rows and "
                    f"{config.MIN_FEATURE_POSITIVES} positives"
                ),
            })
        else:
            kept.append(col)

    for d in dropped:
        logger.warning(f"Dropping '{d['feature']}': {d['reason']}")
    if dropped:
        logger.info(
            f"{len(dropped)} feature(s) dropped for insufficient support; "
            f"{len(kept)} retained."
        )
    return kept, dropped


def fit_customer_baselines(train_df):
    """
    Per-customer behavioural baseline fitted on the training rows only.

    Returned frame is indexed by customer_id with the four columns the feature
    builder expects. Customers seen only at scoring time fall back to neutral
    values inside `build_features`.
    """
    grp = train_df.groupby("customer_id")
    stats = pd.DataFrame({
        "cust_avg_amount": grp["amount"].mean(),
        "cust_std_amount": grp["amount"].std().fillna(0.0),
        "cust_txn_count": grp["amount"].size(),
        "cust_unique_merchants": grp["merchant"].nunique() if "merchant" in train_df.columns else 0,
    })
    stats.index.name = "customer_id"
    return stats


def extract_feature_importance(name, pipeline, feature_names, X_sample, y_sample):
    """
    Per-feature importance in whatever form the estimator exposes.

    Linear models report standardised coefficients; tree ensembles report
    impurity-based importances. Both are normalised to sum to 1 so the dashboard
    can plot them on one axis.
    """
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        raw = np.abs(np.asarray(clf.feature_importances_, dtype=float))
        kind = "impurity_decrease"
    elif hasattr(clf, "coef_"):
        raw = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
        kind = "abs_standardised_coefficient"
    else:
        return None

    total = raw.sum()
    norm = raw / total if total > 0 else raw
    ranked = sorted(
        ({"feature": f, "importance": round(float(v), 6)} for f, v in zip(feature_names, norm, strict=True)),
        key=lambda r: -r["importance"],
    )
    return {"model": name, "kind": kind, "features": ranked}


def cross_validate_pr_auc(name, pipeline, X, y, folds=None, subsample=CV_SUBSAMPLE):
    """Stratified K-fold PR-AUC, reported as mean +/- std."""
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score

    folds = folds or config.CV_FOLDS

    if len(y) > subsample:
        idx, _ = train_test_split(
            np.arange(len(y)), train_size=subsample,
            random_state=config.RANDOM_SEED, stratify=y
        )
        Xc, yc = X[idx], y[idx]
    else:
        Xc, yc = X, y

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
    scores = []
    for tr, te in skf.split(Xc, yc):
        m = clone(pipeline)
        m.fit(Xc[tr], yc[tr])
        scores.append(float(average_precision_score(yc[te], m.predict_proba(Xc[te])[:, 1])))

    return {
        "folds": folds,
        "rows_used": int(len(yc)),
        "pr_auc_mean": round(float(np.mean(scores)), 6),
        "pr_auc_std": round(float(np.std(scores)), 6),
        "pr_auc_per_fold": [round(s, 6) for s in scores],
    }


def train_fraud_models(models_dir=None, sample_path=None, run_cv=True):
    """Train, evaluate and persist the full benchmark."""
    models_path = Path(models_dir) if models_dir else config.MODELS_DIR
    models_path.mkdir(parents=True, exist_ok=True)
    sample_path = Path(sample_path) if sample_path else config.TRANSACTIONS_SAMPLE_PARQUET

    # --- Load -------------------------------------------------------------
    if sample_path.exists():
        logger.info(f"Loading stratified training sample from {sample_path}...")
        df = pd.read_parquet(sample_path)
    elif config.RAW_TRANSACTIONS_CSV.exists():
        logger.info(f"Sample not found; reading {config.ML_SAMPLE_TARGET:,} rows from raw CSV...")
        df = pd.read_csv(config.RAW_TRANSACTIONS_CSV, nrows=config.ML_SAMPLE_TARGET)
    else:
        raise FileNotFoundError(
            f"No training data. Expected {sample_path} or {config.RAW_TRANSACTIONS_CSV}."
        )

    logger.info(f"Training dataset: {len(df):,} rows")
    y = df[config.TARGET_COL].astype(int).values

    # --- Leakage audit ----------------------------------------------------
    # Run first: it decides which columns are legal to use downstream.
    leakage = run_leakage_analysis(df)

    # --- Split, then fit customer baselines on train only -----------------
    # Three-way split: fit / calibrate / test.
    #
    # class_weight="balanced" re-weights the 1% positive class about 100x, which
    # shifts the log-odds by a constant and inflates every predicted
    # probability -- uncalibrated, rows scored 0.75 carried a true fraud rate
    # near 3%. A held-out calibration split maps the scores back onto real
    # probabilities without the test set ever being used for fitting.
    idx_fit_calib, idx_te = train_test_split(
        np.arange(len(y)), test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED, stratify=y,
    )
    idx_tr, idx_cal = train_test_split(
        idx_fit_calib, test_size=config.CALIBRATION_SIZE,
        random_state=config.RANDOM_SEED, stratify=y[idx_fit_calib],
    )
    train_df, calib_df, test_df = df.iloc[idx_tr], df.iloc[idx_cal], df.iloc[idx_te]

    logger.info("Fitting per-customer baselines on the training split only...")
    customer_stats = fit_customer_baselines(train_df)

    logger.info("Assembling feature matrices...")
    X_train_df, feature_names = feat_mod.assemble_matrix(train_df, customer_stats=customer_stats)
    X_test_df, _ = feat_mod.assemble_matrix(
        test_df, customer_stats=customer_stats, fit=False,
        encoder_columns=[c for c in feature_names
                         if c not in config.NUMERIC_FEATURES + config.FLAG_FEATURES + config.CUSTOMER_FEATURES],
    )
    X_test_df = X_test_df.reindex(columns=feature_names, fill_value=0.0)

    X_calib_df, _ = feat_mod.assemble_matrix(
        calib_df, customer_stats=customer_stats, fit=False
    )
    X_calib_df = X_calib_df.reindex(columns=feature_names, fill_value=0.0)

    # Drop binary features the training split cannot support. Keeping them
    # lets the optimiser fit an arbitrary coefficient to a near-empty cell.
    feature_names, dropped_features = drop_low_support_features(
        X_train_df, y[idx_tr], feature_names
    )
    X_train_df = X_train_df[feature_names]
    X_test_df = X_test_df[feature_names]
    X_calib_df = X_calib_df[feature_names]

    X_train, X_test = X_train_df.values, X_test_df.values
    X_calib = X_calib_df.values
    y_train, y_test = y[idx_tr], y[idx_te]
    y_calib = y[idx_cal]
    test_amounts = test_df["amount"].astype(float).values

    pos, neg = int(y_train.sum()), int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos else 1.0
    logger.info(
        f"Feature matrix {X_train.shape}; train fraud={pos:,} legit={neg:,} "
        f"(scale_pos_weight={scale_pos_weight:.1f})"
    )
    logger.info(
        f"Split: {len(y_train):,} fit / {len(y_calib):,} calibrate / {len(y_test):,} test"
    )

    baseline = metric_mod.no_skill_baseline(y_test)
    logger.info(
        f"No-skill baseline: predicting 'never fraud' scores "
        f"{baseline['always_negative']['accuracy']:.4f} accuracy -- any model "
        f"reporting less than this on accuracy is worse than a constant."
    )

    # --- Train ------------------------------------------------------------
    zoo = build_model_zoo(scale_pos_weight)
    results, importances = {}, []

    for name, pipeline in zoo.items():
        logger.info(f"Training {name} on {len(y_train):,} rows...")
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_secs = time.perf_counter() - t0

        # Record how far the raw scores sit from real probabilities, then fix it.
        raw_test_score = pipeline.predict_proba(X_test)[:, 1]
        raw_calibration = metric_mod.calibration_report(y_test, raw_test_score)

        # Platt scaling. The balanced-weight distortion is a constant offset in
        # log-odds, which a sigmoid fit recovers almost exactly, and it is far
        # more stable than isotonic at this positive count. Being monotonic it
        # leaves PR-AUC and ROC-AUC untouched -- the ranking is unchanged, only
        # the numbers attached to it become meaningful.
        # FrozenEstimator replaces the cv="prefit" argument, which scikit-learn
        # 1.9 removed. Wrapping the already-fitted pipeline keeps it frozen so
        # only the calibrator is fitted on the held-out split.
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
        calibrated.fit(X_calib, y_calib)

        t1 = time.perf_counter()
        y_score = calibrated.predict_proba(X_test)[:, 1]
        predict_secs = time.perf_counter() - t1

        evaluation = metric_mod.evaluate_model(y_test, y_score, amounts=test_amounts)
        evaluation["calibration_before"] = raw_calibration
        evaluation["calibration_method"] = "sigmoid (Platt) on a held-out split"
        evaluation["timing"] = {
            "train_seconds": round(train_secs, 3),
            "predict_seconds": round(predict_secs, 4),
            "rows_per_second_inference": int(len(y_test) / predict_secs) if predict_secs else None,
        }

        if run_cv:
            logger.info(f"Cross-validating {name}...")
            evaluation["cross_validation"] = cross_validate_pr_auc(
                name, pipeline, X_train, y_train
            )

        stem = MODEL_FILENAMES[name]
        # The calibrated estimator is what gets served, so the probability the
        # dashboard shows is the probability the evaluation measured.
        joblib.dump(calibrated, models_path / f"{stem}_pipeline.pkl", compress=3)
        evaluation["artifact"] = f"{stem}_pipeline.pkl"
        results[name] = evaluation

        imp = extract_feature_importance(name, pipeline, feature_names, X_train, y_train)
        if imp:
            importances.append(imp)

        r = evaluation["ranking"]
        pk = evaluation["precision_at_k"]
        at1pct = next((row for row in pk if row["capacity_fraction"] == 0.01), pk[0])
        cal = evaluation["calibration"]
        logger.info(
            f"  {name}: PR-AUC={r['pr_auc']:.4f} (random={baseline['prevalence']:.4f}) "
            f"ROC-AUC={r['roc_auc']:.4f} | top-1% lift={at1pct['lift']:.2f}x "
            f"| probability inflation {raw_calibration['inflation_factor']}x -> "
            f"{cal['inflation_factor']}x | {train_secs:.1f}s"
        )

    # --- Select the best model on PR-AUC ----------------------------------
    best_model_name = max(results, key=lambda n: results[n]["ranking"]["pr_auc"])
    logger.info(
        f"Best model by PR-AUC: {best_model_name} "
        f"({results[best_model_name]['ranking']['pr_auc']:.4f})"
    )

    # Shared artefacts for the serving path.
    joblib.dump(feature_names, models_path / "feature_cols.pkl", compress=3)
    joblib.dump(customer_stats, models_path / "customer_baselines.pkl", compress=3)

    comparison = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "dataset_total_records": 15_000_000,
        "training_sample_size": int(len(df)),
        "train_rows": int(len(y_train)),
        "calibration_rows": int(len(y_calib)),
        "test_rows": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": feature_names,
        "dropped_features": dropped_features,
        "selection_metric": "pr_auc",
        "best_model": best_model_name,
        "no_skill_baseline": baseline,
        "models": results,
        "leakage_summary": {
            "excluded_columns": config.LEAKING_COLS,
            "verdict": leakage.get("status_leakage", {}).get("verdict"),
            "zero_model_rule": leakage.get("status_leakage", {}).get("zero_model_rule"),
            "inflation_if_included": leakage.get("leaky_vs_clean", {}).get("inflation"),
        },
        "methodology_notes": [
            "Model selection uses PR-AUC (average precision), not accuracy: at "
            "0.96% prevalence a constant 'never fraud' predictor already scores "
            "99.04% accuracy.",
            "Per-customer baseline features are fitted on the training split only, "
            "so the test rows do not contribute to their own baselines.",
            "transaction_status is excluded from every feature set because it "
            "records the bank's own fraud verdict (precision 1.000, recall 0.743 "
            "as a standalone rule) and is unavailable at scoring time.",
            "Operating thresholds are tuned per model for F1 and for expected "
            "net saving rather than left at the 0.5 default.",
            "Binary features with too few examples to estimate are dropped "
            "rather than fitted. The stratified sample preserves the overall "
            "fraud rate but not rare feature combinations, so an interaction "
            "that is genuinely strong across 15M rows can arrive with single "
            "digit support and no positives -- enough for the optimiser to "
            "assign it a large arbitrary weight.",
            "Scores are calibrated with Platt scaling on a held-out split. "
            "class_weight='balanced' makes the raw output a good ranking but a "
            "poor probability: uncalibrated, rows scored 0.75 carried a true "
            "fraud rate near 3%. Calibration is monotonic, so PR-AUC and ROC-AUC "
            "are unchanged while the displayed probability becomes meaningful.",
        ],
    }

    with open(models_path / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    with open(config.FEATURE_IMPORTANCE_JSON, "w") as f:
        json.dump({"models": importances}, f, indent=2)

    threshold_payload = {
        "generated_at": comparison["generated_at"],
        "cost_assumptions": {
            "cost_per_review_inr": config.COST_PER_REVIEW,
            "recovery_rate": config.RECOVERY_RATE,
        },
        "models": {
            name: {
                "at_default_threshold": res["at_default_threshold"],
                "at_best_f1_threshold": res["at_best_f1_threshold"],
                "at_cost_optimal_threshold": res.get("at_cost_optimal_threshold"),
                "cost_curve": res.get("cost_analysis", {}).get("curve", []),
                "cost_optimal": res.get("cost_analysis", {}).get("cost_optimal"),
                "precision_at_k": res["precision_at_k"],
            }
            for name, res in results.items()
        },
    }
    with open(config.THRESHOLD_ANALYSIS_JSON, "w") as f:
        json.dump(threshold_payload, f, indent=2)

    logger.info(f"Benchmark complete. Artefacts written to {models_path}")
    return comparison


if __name__ == "__main__":
    train_fraud_models()
