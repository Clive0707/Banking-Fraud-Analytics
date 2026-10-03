"""
Flask API tests.

These run against whatever artefacts are present. When the pipeline has not been
run, endpoints must report that honestly with a 503 rather than returning
plausible-looking numbers, so the suite asserts "200 or 503" and checks the
shape of a 200.
"""


import pytest

from backend.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def has_models(client):
    return bool(client.get("/api/health").get_json().get("models_loaded"))


READ_ENDPOINTS = [
    "/api/summary", "/api/fraud", "/api/fraud/trends", "/api/segment-lift",
    "/api/leakage", "/api/time-analytics", "/api/geo-analytics",
    "/api/payment-device-analytics", "/api/clusters", "/api/anomalies",
    "/api/model-performance", "/api/thresholds", "/api/feature-importance",
    "/api/data-quality", "/api/processing", "/api/alerts", "/api/report",
    "/api/transactions",
]


class TestEndpointsRespond:
    @pytest.mark.parametrize("path", READ_ENDPOINTS)
    def test_returns_json_with_a_sane_status(self, client, path):
        res = client.get(path)
        assert res.status_code in (200, 503), f"{path} returned {res.status_code}"
        assert res.get_json() is not None

    def test_health_always_available(self, client):
        body = client.get("/api/health").get_json()
        assert body["status"] == "ok"
        assert "artefacts" in body

    def test_every_loaded_model_can_actually_score(self, client, has_models):
        """
        A model that unpickles but cannot predict must not be advertised.

        scikit-learn pickles are not forward compatible: artefacts written by
        1.9 load under 1.7 and then raise at predict time. The loader probes
        each estimator, so anything listed here is genuinely usable.
        """
        if not has_models:
            pytest.skip("no models in this environment")
        for name in client.get("/api/health").get_json()["models_loaded"]:
            res = client.post(f"/api/predict?model={name}", json={
                "amount": 1000, "balance_before": 5000, "balance_after": 4000,
                "transaction_time": "12:00:00", "transaction_date": "2025-06-15",
                "transaction_type": "UPI", "account_type": "Savings",
                "payment_method": "UPI", "device_type": "Android",
                "location": "Mumbai", "merchant": "Amazon", "customer_id": 100001,
            })
            assert res.status_code == 200, f"{name} is advertised but cannot score"
            assert "fraud_probability" in res.get_json(), name

    def test_unknown_endpoint_is_404(self, client):
        assert client.get("/api/does-not-exist").status_code == 404

    def test_index_is_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"Banking Intelligence" in res.data


class TestNoFabricatedValues:
    """
    Guards against the placeholders the service used to emit: a constant
    fraud_probability of 0.945 on every suspicious transaction, and a hardcoded
    processing-metadata block.
    """

    def test_suspicious_probabilities_are_not_constant(self, client, has_models):
        if not has_models:
            pytest.skip("models not trained in this environment")
        body = client.get("/api/fraud").get_json()
        probs = [t.get("fraud_probability") for t in body.get("suspicious_transactions", [])]
        probs = [p for p in probs if p is not None]
        if len(probs) < 2:
            pytest.skip("not enough scored transactions")
        assert len(set(probs)) > 1, "every transaction carries the same probability"
        assert 0.945 not in probs or len(set(probs)) > 1

    def test_processing_metadata_comes_from_a_real_run(self, client):
        body = client.get("/api/processing").get_json()
        if "error" in body:
            pytest.skip("pipeline benchmark not generated")
        assert "stages" in body and body["stages"]
        assert body["total_seconds"] > 0

    def test_data_quality_is_measured(self, client):
        body = client.get("/api/data-quality").get_json()
        if "error" in body:
            pytest.skip("pipeline not run")
        assert body.get("measured") is True
        assert "consistency_check" in body


class TestPrediction:
    def test_rejects_an_empty_body(self, client):
        res = client.post("/api/predict", data="", content_type="application/json")
        assert res.status_code == 400

    def test_scores_a_valid_transaction(self, client, has_models):
        if not has_models:
            pytest.skip("models not trained in this environment")
        res = client.post("/api/predict", json={
            "amount": 2500, "balance_before": 80000, "balance_after": 77500,
            "transaction_time": "14:30:00", "transaction_date": "2025-06-15",
            "transaction_type": "UPI", "account_type": "Savings",
            "payment_method": "UPI", "device_type": "Android",
            "location": "Mumbai", "merchant": "Amazon", "customer_id": 100001,
        })
        assert res.status_code == 200
        body = res.get_json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["prediction"] in ("Fraud", "Legitimate")
        assert "operating_threshold" in body

    def test_risky_transaction_scores_above_a_benign_one(self, client, has_models):
        """Overnight, international, high-value and drained should rank higher."""
        if not has_models:
            pytest.skip("models not trained in this environment")
        base = {
            "transaction_date": "2025-06-15", "account_type": "Savings",
            "merchant": "Amazon", "customer_id": 100001,
        }
        benign = client.post("/api/predict", json={
            **base, "amount": 2000, "balance_before": 90000, "balance_after": 88000,
            "transaction_time": "13:00:00", "transaction_type": "UPI",
            "payment_method": "UPI", "device_type": "Android", "location": "Mumbai",
        }).get_json()
        risky = client.post("/api/predict", json={
            **base, "amount": 150000, "balance_before": 150000, "balance_after": 0,
            "transaction_time": "03:12:00", "transaction_type": "Online Purchase",
            "payment_method": "Credit Card", "device_type": "Windows", "location": "Dubai",
        }).get_json()
        assert risky["fraud_probability"] > benign["fraud_probability"]

    def test_risk_factors_are_explained(self, client, has_models):
        if not has_models:
            pytest.skip("models not trained in this environment")
        body = client.post("/api/predict", json={
            "amount": 150000, "balance_before": 150000, "balance_after": 0,
            "transaction_time": "03:12:00", "transaction_date": "2025-06-15",
            "transaction_type": "Online Purchase", "account_type": "Savings",
            "payment_method": "Credit Card", "device_type": "Windows",
            "location": "Dubai", "merchant": "Amazon", "customer_id": 100001,
        }).get_json()
        joined = " ".join(body["triggered_risk_factors"]).lower()
        assert "overnight" in joined
        assert "international" in joined


class TestTransactionExplorer:
    def test_pagination_shape(self, client):
        body = client.get("/api/transactions?page=1&per_page=5").get_json()
        if "error" in body:
            pytest.skip("transaction data not loaded")
        assert len(body["data"]) <= 5
        assert body["page"] == 1
        assert "total_records" in body

    def test_fraud_filter_returns_only_fraud(self, client):
        body = client.get("/api/transactions?is_fraud=1&per_page=20").get_json()
        if "error" in body:
            pytest.skip("transaction data not loaded")
        assert all(r["is_fraud"] == 1 for r in body["data"])

    def test_invalid_filter_is_rejected(self, client):
        res = client.get("/api/transactions?min_amount=not-a-number")
        assert res.status_code in (400, 503)

    def test_missing_transaction_is_404(self, client):
        res = client.get("/api/fraud/investigation/TXN-DOES-NOT-EXIST")
        assert res.status_code in (404, 503)

    def test_export_returns_csv(self, client):
        res = client.get("/api/transactions/export?is_fraud=1&limit=10")
        if res.status_code == 503:
            pytest.skip("transaction data not loaded")
        assert res.status_code == 200
        assert "text/csv" in res.content_type
        assert b"transaction_id" in res.data
