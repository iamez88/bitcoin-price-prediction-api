from flask import Flask, jsonify, request
import logging
from functools import wraps
import time
from datetime import datetime
import os

from .model import load_model, train_model, get_latest_prediction

# Rate limiting
rate_limits = {}
RATE_LIMIT = 10  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()
        
        # Initialize client's rate limit data if not exists
        if client_ip not in rate_limits:
            rate_limits[client_ip] = []
        
        # Clean old timestamps
        rate_limits[client_ip] = [ts for ts in rate_limits[client_ip] if current_time - ts < RATE_LIMIT_WINDOW]
        
        # Check if rate limit exceeded
        if len(rate_limits[client_ip]) >= RATE_LIMIT:
            logging.warning(f"Rate limit exceeded for {client_ip}")
            return jsonify({
                "success": False,
                "error": "Rate limit exceeded. Please try again later."
            }), 429
        
        # Add current timestamp
        rate_limits[client_ip].append(current_time)
        
        return f(*args, **kwargs)
    return decorated_function

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Simple logging setup
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'api.log')
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file,
        filemode='a'
    )
    
    logger = app.logger
    logger.info('Application starting up')
    
    @app.route('/')
    @rate_limit
    def hello():
        """Main endpoint with API information"""
        logger.info('Main endpoint processing HTTP request')
        return jsonify({
            "success": True, 
            "message": "Bitcoin Price Direction Prediction API",
            "endpoints": {
                "/": "API information (GET)",
                "/api/health": "Health check (GET)",
                "/api/predict": "Get price direction prediction (GET)",
                "/api/model/info": "Get model information (GET)"
            }
        })
    
    @app.route('/api/health')
    def health_check():
        """Health check endpoint"""
        logger.info('Health check endpoint called')
        return jsonify({
            "success": True, 
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/predict')
    @rate_limit
    def predict():
        """Bitcoin price direction prediction endpoint"""
        logger.info('Prediction endpoint called')
        
        try:
            # Get prediction
            result = get_latest_prediction()
            
            if result is None:
                logger.error("Prediction failed")
                return jsonify({
                    "success": False,
                    "error": "Failed to make prediction"
                }), 500
            
            logger.info(f"Prediction: {result['prediction']} with confidence {result['confidence']:.4f}")
            
            return jsonify({
                "success": True,
                **result
            })
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({
                "success": False,
                "error": "An error occurred during prediction"
            }), 500
    
    @app.route('/api/model/info')
    @rate_limit
    def model_info():
        """Model information endpoint"""
        logger.info('Model info endpoint called')
        
        try:
            # Load model
            model_info = load_model()
            
            if model_info is None:
                return jsonify({
                    "success": False,
                    "error": "No model information available"
                }), 404
            
            # Return model info
            return jsonify({
                "success": True,
                "model_info": {
                    "accuracy": model_info['accuracy'],
                    "trained_at": model_info['trained_at'],
                    "features": model_info['feature_cols']
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return jsonify({
                "success": False,
                "error": "An error occurred while retrieving model information"
            }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Generic error handler"""
        logger.error(f'Unhandled exception: {str(e)}')
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
    
    return app

# Add this code to run the app when the script is executed directly
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=50505)