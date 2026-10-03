"""
Tests for the Spark MLlib training module.

A real SparkSession is slow to start, so these cover the pure-Python logic that
can be verified without one: the feature contract, the leakage exclusion, and
the comparison block that decides whether 15M rows beat the 500k sample. The
end-to-end distributed run is exercised by scripts/smoke_test.py instead.
"""

import json

import pytest

from src import config


@pytest.fixture(scope="module")
def spark_ml():
    """Import lazily so a machine without a JVM can still collect the rest."""
    return pytest.importorskip(
        "src.fraud_detection.spark_ml",
        reason="pyspark unavailable",
    )


@pytest.fixture(scope="module")
def spark_session():
    """
    Minimal local session.

    Spark ML estimators are thin wrappers over JVM objects, so constructing one
    requires a live SparkContext. A single-core session keeps this to a few
    seconds; the test is skipped where no JVM is available.
    """
    pyspark = pytest.importorskip("pyspark", reason="pyspark unavailable")
    from src import config as cfg

    cfg.configure_spark_environment()
    try:
        session = (
            pyspark.sql.SparkSession.builder
            .master("local[1]")
            .appName("BankingFraudAnalytics-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
    except Exception as exc:  # no JVM on this machine
        pytest.skip(f"could not start Spark: {exc}")
    yield session
    session.stop()


class TestFeatureContract:
    def test_matches_the_sklearn_feature_families(self, spark_ml):
        """
        The Spark and scikit-learn paths must describe the same features, or the
        two benchmarks are not comparable.
        """
        assert spark_ml.NUMERIC_COLS == config.NUMERIC_FEATURES
        assert spark_ml.FLAG_COLS == config.FLAG_FEATURES

    def test_customer_columns_cover_the_sklearn_set(self, spark_ml):
        assert set(config.CUSTOMER_FEATURES) <= set(spark_ml.CUSTOMER_COLS)

    def test_location_is_not_one_hot_encoded(self, spark_ml):
        """
        location is represented by is_international plus the measured per-location
        lift, not as 14 dummy columns.
        """
        assert "location" not in spark_ml.CATEGORICAL_COLS

    def test_no_leaking_column_in_any_feature_list(self, spark_ml):
        every = (
            spark_ml.NUMERIC_COLS + spark_ml.FLAG_COLS
            + spark_ml.CUSTOMER_COLS + spark_ml.CATEGORICAL_COLS
        )
        for leaking in config.LEAKING_COLS:
            assert leaking not in every
        assert config.TARGET_COL not in every

    def test_capacity_fractions_match_the_sklearn_benchmark(self, spark_ml):
        assert spark_ml.CAPACITY_FRACTIONS == config.REVIEW_CAPACITY_FRACTIONS


class TestModelZoo:
    def test_every_estimator_uses_the_class_weight_column(self, spark_ml, spark_session):
        """
        Spark has no class_weight="balanced"; imbalance must be handled with an
        explicit weight column on every estimator.
        """
        for name, est in spark_ml.build_model_zoo().items():
            assert est.getOrDefault("weightCol") == "class_weight", name

    def test_every_estimator_targets_the_right_label(self, spark_ml, spark_session):
        for name, est in spark_ml.build_model_zoo().items():
            assert est.getOrDefault("labelCol") == config.TARGET_COL, name

    def test_expensive_models_can_be_skipped(self, spark_ml, spark_session):
        lean = spark_ml.build_model_zoo(include_gbt=False, include_rf=False)
        assert len(lean) == 1
        assert "Spark Logistic Regression" in lean


class TestSklearnComparison:
    """
    The comparison block is what answers "does 15M actually beat 500k?", so its
    verdict logic is worth pinning down.
    """

    def _payload(self, pr_auc, train_rows=12_000_000):
        return {
            "best_model": "Spark Logistic Regression",
            "train_rows": train_rows,
            "models": {
                "Spark Logistic Regression": {
                    "ranking": {"pr_auc": pr_auc, "roc_auc": 0.75},
                    "precision_at_k": [{"capacity_fraction": 0.01, "lift": 4.3}],
                }
            },
        }

    def _write_sklearn(self, tmp_path, monkeypatch, pr_auc=0.03):
        path = tmp_path / "model_comparison.json"
        path.write_text(json.dumps({
            "best_model": "Logistic Regression",
            "train_rows": 400_000,
            "models": {
                "Logistic Regression": {
                    "ranking": {"pr_auc": pr_auc, "roc_auc": 0.754},
                    "precision_at_k": [{"capacity_fraction": 0.01, "lift": 4.29}],
                }
            },
        }))
        monkeypatch.setattr(config, "MODEL_COMPARISON_JSON", path)
        return path

    def test_reports_unavailable_without_the_sklearn_run(self, spark_ml, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_COMPARISON_JSON", tmp_path / "missing.json")
        out = spark_ml._compare_with_sklearn(self._payload(0.03))
        assert out["available"] is False

    def test_flat_result_is_reported_as_no_material_gain(self, spark_ml, tmp_path, monkeypatch):
        self._write_sklearn(tmp_path, monkeypatch, pr_auc=0.0300)
        out = spark_ml._compare_with_sklearn(self._payload(0.0302))
        assert out["verdict"] == "no material gain from the extra data"
        assert abs(out["pr_auc_change_pct"]) < 5

    def test_clear_improvement_is_detected(self, spark_ml, tmp_path, monkeypatch):
        self._write_sklearn(tmp_path, monkeypatch, pr_auc=0.0300)
        out = spark_ml._compare_with_sklearn(self._payload(0.0400))
        assert out["verdict"] == "more data helps"

    def test_clear_regression_is_detected(self, spark_ml, tmp_path, monkeypatch):
        self._write_sklearn(tmp_path, monkeypatch, pr_auc=0.0300)
        out = spark_ml._compare_with_sklearn(self._payload(0.0200))
        assert out["verdict"] == "more data hurts"

    def test_reports_the_training_data_multiple(self, spark_ml, tmp_path, monkeypatch):
        self._write_sklearn(tmp_path, monkeypatch)
        out = spark_ml._compare_with_sklearn(self._payload(0.031, train_rows=12_000_000))
        assert out["training_data_multiple"] == pytest.approx(30.0, abs=0.1)
