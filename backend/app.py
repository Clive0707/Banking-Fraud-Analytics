import os
import sys
import logging
from pathlib import Path
from flask import Flask, send_from_directory, jsonify

# Include project root in path
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir))

from backend.services.data_service import DataService
from backend.routes.analytics_routes import create_analytics_blueprint
from backend.routes.prediction_routes import create_prediction_blueprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("BankingAnalyticsBackend")

def create_app():
    frontend_dir = base_dir / "frontend"
    app = Flask(__name__, static_folder=str(frontend_dir), static_url_path="")

    logger.info("Initializing Data Service...")
    data_service = DataService(base_dir=base_dir)

    # Register blueprints
    app.register_blueprint(create_analytics_blueprint(data_service), url_prefix='/api')
    app.register_blueprint(create_prediction_blueprint(data_service), url_prefix='/api')

    @app.route('/')
    def serve_frontend():
        return send_from_directory(app.static_folder, 'index.html')

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Banking Analytics Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
