from flask import Blueprint, jsonify, request

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

        res = data_service.get_transactions(
            page=page, per_page=per_page, customer_id=customer_id,
            transaction_type=transaction_type, payment_method=payment_method,
            location=location, is_fraud=is_fraud, min_amount=min_amount,
            max_amount=max_amount, search=search
        )
        return jsonify(res)

    @bp.route('/fraud', methods=['GET'])
    def get_fraud():
        return jsonify(data_service.get_fraud_analytics())

    @bp.route('/fraud/trends', methods=['GET'])
    def get_fraud_trends():
        return jsonify(data_service.get_fraud_trends())

    @bp.route('/customers', methods=['GET'])
    def get_customers():
        # Summary of customer segmentation & counts
        return jsonify(data_service.get_customer_clusters())

    @bp.route('/clusters', methods=['GET'])
    def get_clusters():
        return jsonify(data_service.get_customer_clusters())

    @bp.route('/anomalies', methods=['GET'])
    def get_anomalies():
        return jsonify(data_service.get_anomalies())

    @bp.route('/model-performance', methods=['GET'])
    def get_model_performance():
        return jsonify(data_service.get_model_performance())

    return bp
