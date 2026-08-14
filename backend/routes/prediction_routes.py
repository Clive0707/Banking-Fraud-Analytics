from flask import Blueprint, jsonify, request

def create_prediction_blueprint(data_service):
    bp = Blueprint('prediction', __name__)

    @bp.route('/predict', methods=['POST'])
    def predict():
        input_data = request.get_json(force=True, silent=True)
        if not input_data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        model_name = request.args.get('model')
        result = data_service.predict_fraud(input_data, model_name=model_name)
        
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    return bp
