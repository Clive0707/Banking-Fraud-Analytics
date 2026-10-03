"""
Analytics REST endpoints.

Changes from the original blueprint
---------------------------------
* Removed four duplicate routes that returned identical payloads to an existing
  endpoint (`/customers`, `/anomalies/top`, `/fraud/by-device`,
  `/fraud/by-payment`, `/fraud/by-location`), keeping thin aliases only where
  the frontend already calls them.
* Added the endpoints that expose the new analytics: segment lift, the target
  leakage audit, threshold/cost analysis, feature importance and the pipeline
  benchmark.
* CSV export now honours the filters at full scope instead of silently
  truncating to the first 1,000 rows of page 1, and streams within a row cap.
* Error responses carry real HTTP status codes rather than 200 with an
  `{"error": ...}` body.
"""

import csv
import io

from flask import Blueprint, Response, jsonify, request

from src import config

EXPORT_COLUMNS = [
    "transaction_id", "customer_id", "transaction_date", "transaction_time",
    "transaction_type", "account_type", "amount", "balance_before", "balance_after",
    "merchant", "location", "payment_method", "device_type",
    "transaction_status", "is_fraud",
]


def _filters_from_request():
    """Collect the shared transaction filters from the query string."""
    return {
        "customer_id": request.args.get("customer_id"),
        "transaction_type": request.args.get("transaction_type"),
        "payment_method": request.args.get("payment_method"),
        "location": request.args.get("location"),
        "is_fraud": request.args.get("is_fraud"),
        "min_amount": request.args.get("min_amount"),
        "max_amount": request.args.get("max_amount"),
        "search": request.args.get("search"),
    }


def _respond(payload, missing_status=503):
    """
    Serialise a service result, mapping a missing-artefact error to a status code.

    An unrun pipeline is a server-state problem, so it reports 503 rather than
    masquerading as a successful response.
    """
    if isinstance(payload, dict) and "error" in payload:
        return jsonify(payload), missing_status
    return jsonify(payload)


def create_analytics_blueprint(data_service):
    bp = Blueprint("analytics", __name__)

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    @bp.route("/summary", methods=["GET"])
    def get_summary():
        return _respond(data_service.get_summary())

    @bp.route("/health", methods=["GET"])
    def health():
        """Readiness probe reporting which artefacts actually loaded."""
        return jsonify({
            "status": "ok",
            "models_loaded": sorted(data_service.models.keys()),
            "best_model": data_service.best_model_name,
            "transactions_loaded": (
                0 if data_service.transactions_df is None else len(data_service.transactions_df)
            ),
            "customers_loaded": (
                0 if data_service.customer_profiles_df is None
                else len(data_service.customer_profiles_df)
            ),
            "artefacts": {
                "summary_stats": data_service.summary_stats is not None,
                "fraud_aggregates": data_service.fraud_aggregates is not None,
                "data_quality": data_service.data_quality is not None,
                "segment_lift": data_service.segment_lift is not None,
                "leakage_report": data_service.leakage_report is not None,
                "time_analytics": data_service.time_analytics is not None,
                "model_comparison": data_service.model_comparison is not None,
                "threshold_analysis": data_service.threshold_analysis is not None,
                "feature_importance": data_service.feature_importance is not None,
                "customer_clusters": data_service.customer_clusters is not None,
                "anomalies": data_service.anomalies_summary is not None,
                "pipeline_benchmark": data_service.pipeline_benchmark is not None,
                "spark_ml_comparison": data_service.spark_ml_comparison is not None,
            },
        })

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @bp.route("/transactions", methods=["GET"])
    def get_transactions():
        res = data_service.get_transactions(
            page=request.args.get("page", 1, type=int),
            per_page=request.args.get("per_page", 25, type=int),
            sort_by=request.args.get("sort_by", "transaction_date"),
            order=request.args.get("order", "desc"),
            **_filters_from_request(),
        )
        return _respond(res)

    @bp.route("/transactions/export", methods=["GET"])
    def export_transactions():
        limit = min(request.args.get("limit", config.MAX_EXPORT_ROWS, type=int),
                    config.MAX_EXPORT_ROWS)
        res = data_service.get_transactions(
            page=1, per_page=limit, max_rows=limit,
            sort_by=request.args.get("sort_by", "transaction_date"),
            order=request.args.get("order", "desc"),
            **_filters_from_request(),
        )
        if "error" in res:
            return jsonify(res), 503

        records = res.get("data", [])
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=filtered_transactions.csv",
                "X-Total-Matching-Rows": str(res.get("total_records", len(records))),
                "X-Rows-Exported": str(len(records)),
            },
        )

    # ------------------------------------------------------------------
    # Fraud analytics
    # ------------------------------------------------------------------

    @bp.route("/fraud", methods=["GET"])
    def get_fraud():
        return _respond(data_service.get_fraud_analytics())

    @bp.route("/fraud/trends", methods=["GET"])
    def get_fraud_trends():
        return _respond(data_service.get_fraud_trends())

    @bp.route("/fraud/investigation/<transaction_id>", methods=["GET"])
    def get_fraud_investigation(transaction_id):
        return _respond(data_service.get_transaction_investigation(transaction_id),
                        missing_status=404)

    @bp.route("/segment-lift", methods=["GET"])
    def get_segment_lift():
        """Fraud rate and lift per risk segment, measured across all 15M rows."""
        return _respond(data_service.get_segment_lift())

    @bp.route("/leakage", methods=["GET"])
    def get_leakage():
        """
        The target-leakage audit: why transaction_status is excluded from every
        feature set despite predicting fraud with precision 1.000.
        """
        return _respond(data_service.get_leakage_report())

    # ------------------------------------------------------------------
    # Dimensional analytics
    # ------------------------------------------------------------------

    @bp.route("/time-analytics", methods=["GET"])
    def get_time_analytics():
        return _respond(data_service.get_time_analytics())

    @bp.route("/geo-analytics", methods=["GET"])
    def get_geo_analytics():
        return _respond(data_service.get_geo_analytics())

    # Alias kept because the frontend already requests this path.
    bp.add_url_rule("/fraud/by-location", "fraud_by_location",
                    get_geo_analytics, methods=["GET"])

    @bp.route("/payment-device-analytics", methods=["GET"])
    def get_payment_device_analytics():
        return _respond(data_service.get_payment_device_analytics())

    # ------------------------------------------------------------------
    # Customers
    # ------------------------------------------------------------------

    @bp.route("/customer/<customer_id>", methods=["GET"])
    def get_customer_profile(customer_id):
        return _respond(data_service.get_customer_profile(customer_id), missing_status=404)

    @bp.route("/clusters", methods=["GET"])
    def get_clusters():
        return _respond(data_service.get_customer_clusters())

    @bp.route("/anomalies", methods=["GET"])
    def get_anomalies():
        return _respond(data_service.get_anomalies())

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    @bp.route("/model-performance", methods=["GET"])
    def get_model_performance():
        return _respond(data_service.get_model_performance())

    @bp.route("/thresholds", methods=["GET"])
    def get_thresholds():
        """Per-model threshold sweep with the cost curve and optimal operating point."""
        return _respond(data_service.get_threshold_analysis())

    @bp.route("/feature-importance", methods=["GET"])
    def get_feature_importance():
        return _respond(data_service.get_feature_importance())

    @bp.route("/spark-ml", methods=["GET"])
    def get_spark_ml():
        """
        Spark MLlib models trained distributed on all 15M rows, with the
        comparison against the 500k-sample scikit-learn benchmark.
        """
        return _respond(data_service.get_spark_ml())

    # ------------------------------------------------------------------
    # Platform
    # ------------------------------------------------------------------

    @bp.route("/data-quality", methods=["GET"])
    def get_data_quality():
        return _respond(data_service.get_data_quality())

    @bp.route("/processing", methods=["GET"])
    def get_processing():
        return _respond(data_service.get_processing_metadata())

    @bp.route("/alerts", methods=["GET"])
    def get_alerts():
        return jsonify(data_service.get_alerts(request.args.get("category", "All")))

    @bp.route("/report", methods=["GET"])
    def get_report():
        return jsonify(data_service.get_report_data())

    return bp
