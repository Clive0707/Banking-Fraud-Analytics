from flask import Blueprint, jsonify, request, Response
import csv
import io

def create_analytics_blueprint(data_service):
    bp = Blueprint('analytics', __name__)

    @bp.route('/summary', methods=['GET'])
    def get_summary():
        data = data_service.get_summary()
        return jsonify(data)

    @bp.route('/transactions', methods=['GET'])
    def get_transactions():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 25, type=int)
        customer_id = request.args.get('customer_id')
        transaction_type = request.args.get('transaction_type')
        payment_method = request.args.get('payment_method')
        location = request.args.get('location')
        is_fraud = request.args.get('is_fraud')
        min_amount = request.args.get('min_amount')
        max_amount = request.args.get('max_amount')
        search = request.args.get('search')
        sort_by = request.args.get('sort_by', 'transaction_date')
        order = request.args.get('order', 'desc')

        res = data_service.get_transactions(
            page=page, per_page=per_page, customer_id=customer_id,
            transaction_type=transaction_type, payment_method=payment_method,
            location=location, is_fraud=is_fraud, min_amount=min_amount,
            max_amount=max_amount, search=search, sort_by=sort_by, order=order
        )
        return jsonify(res)

    @bp.route('/transactions/export', methods=['GET'])
    def export_transactions():
        # Export filtered transactions to CSV
        customer_id = request.args.get('customer_id')
        transaction_type = request.args.get('transaction_type')
        payment_method = request.args.get('payment_method')
        location = request.args.get('location')
        is_fraud = request.args.get('is_fraud')
        min_amount = request.args.get('min_amount')
        max_amount = request.args.get('max_amount')
        search = request.args.get('search')

        res = data_service.get_transactions(
            page=1, per_page=1000, customer_id=customer_id,
            transaction_type=transaction_type, payment_method=payment_method,
            location=location, is_fraud=is_fraud, min_amount=min_amount,
            max_amount=max_amount, search=search
        )
        
        records = res.get("data", [])
        output = io.StringIO()
        if records:
            writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=filtered_transactions.csv"}
        )

    @bp.route('/fraud', methods=['GET'])
    def get_fraud():
        return jsonify(data_service.get_fraud_analytics())

    @bp.route('/fraud/trends', methods=['GET'])
    def get_fraud_trends():
        return jsonify(data_service.get_fraud_trends())

    @bp.route('/fraud/by-location', methods=['GET'])
    def get_fraud_by_location():
        return jsonify(data_service.get_geo_analytics())

    @bp.route('/fraud/by-device', methods=['GET'])
    def get_fraud_by_device():
        return jsonify(data_service.get_payment_device_analytics())

    @bp.route('/fraud/by-payment', methods=['GET'])
    def get_fraud_by_payment():
        return jsonify(data_service.get_payment_device_analytics())

    @bp.route('/fraud/investigation/<transaction_id>', methods=['GET'])
    def get_fraud_investigation(transaction_id):
        res = data_service.get_transaction_investigation(transaction_id)
        if "error" in res:
            return jsonify(res), 404
        return jsonify(res)

    @bp.route('/customer/<customer_id>', methods=['GET'])
    def get_customer_profile(customer_id):
        res = data_service.get_customer_profile(customer_id)
        if "error" in res:
            return jsonify(res), 404
        return jsonify(res)

    @bp.route('/customers', methods=['GET'])
    def get_customers():
        return jsonify(data_service.get_customer_clusters())

    @bp.route('/clusters', methods=['GET'])
    def get_clusters():
        return jsonify(data_service.get_customer_clusters())

    @bp.route('/anomalies', methods=['GET'])
    def get_anomalies():
        return jsonify(data_service.get_anomalies())

    @bp.route('/anomalies/top', methods=['GET'])
    def get_top_anomalies():
        return jsonify(data_service.get_anomalies())

    @bp.route('/model-performance', methods=['GET'])
    def get_model_performance():
        return jsonify(data_service.get_model_performance())

    @bp.route('/data-quality', methods=['GET'])
    def get_data_quality():
        return jsonify(data_service.get_data_quality())

    @bp.route('/processing', methods=['GET'])
    def get_processing():
        return jsonify(data_service.get_processing_metadata())

    @bp.route('/alerts', methods=['GET'])
    def get_alerts():
        category = request.args.get('category', 'All')
        return jsonify(data_service.get_alerts(category))

    @bp.route('/time-analytics', methods=['GET'])
    def get_time_analytics():
        return jsonify(data_service.get_time_analytics())

    @bp.route('/geo-analytics', methods=['GET'])
    def get_geo_analytics():
        return jsonify(data_service.get_geo_analytics())

    @bp.route('/payment-device-analytics', methods=['GET'])
    def get_payment_device_analytics():
        return jsonify(data_service.get_payment_device_analytics())

    @bp.route('/report', methods=['GET'])
    def get_report():
        return jsonify(data_service.get_report_data())

    return bp
