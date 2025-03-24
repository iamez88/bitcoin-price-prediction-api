from flask import Flask, jsonify, request
import logging
from functools import wraps
import time
from datetime import datetime
import os

from .model import load_model, train_model, get_latest_prediction

# Configure logging for all modules
def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/api.log'),
            logging.StreamHandler()  # Also log to console
        ]
    )

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate logging
    root_logger.handlers = []
    
    # Add handlers to root logger
    file_handler = logging.FileHandler('logs/api.log')
    console_handler = logging.StreamHandler()
    
    # Set format for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

# Set up logging at module level
setup_logging()
logger = logging.getLogger(__name__)

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
    
    logger.info('Application starting up')
    
    @app.route('/')
    @rate_limit
    def hello():
        """Main endpoint with API information"""
        logger.info('Main endpoint accessed')
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
        logger.info('Health check endpoint accessed')
        return jsonify({
            "success": True, 
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/predict')
    @rate_limit
    def predict():
        """Bitcoin price direction prediction endpoint"""
        logger.info('Prediction endpoint accessed')
        
        try:
            # Get prediction
            result = get_latest_prediction()
            
            if result is None:
                logger.error("Prediction failed - no result returned")
                return jsonify({
                    "success": False,
                    "error": "Failed to make prediction"
                }), 500
            
            logger.info(f"Prediction successful: {result['prediction']} with confidence {result['confidence']:.4f}")
            
            return jsonify({
                "success": True,
                **result
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({
                "success": False,
                "error": "An error occurred during prediction"
            }), 500
    
    @app.route('/api/model/info')
    @rate_limit
    def model_info():
        """Model information endpoint"""
        logger.info('Model info endpoint accessed')
        
        try:
            # Load model
            model_info = load_model()
            
            if model_info is None:
                logger.warning("No model information available")
                return jsonify({
                    "success": False,
                    "error": "No model information available"
                }), 404
            
            logger.info(f"Model info retrieved successfully")
            return jsonify({
                "success": True,
                "model_info": {
                    "accuracy": model_info['accuracy'],
                    "trained_at": model_info['trained_at'],
                    "features": model_info['feature_cols']
                }
            })
            
        except Exception as e:
            logger.error(f"Model info error: {str(e)}")
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