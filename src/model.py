import logging
import os
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data import fetch_bitcoin_data, prepare_features

logger = logging.getLogger(__name__)

# Model file path
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'btc_model.joblib')

def train_model():
    """
    Train the Bitcoin price direction prediction model
    
    Returns:
        dict or None: Model information dictionary, or None if training fails
    """
    logger.info("Starting model training")
    
    # Fetch data
    btc_data = fetch_bitcoin_data()
    if btc_data is None:
        logger.error("Failed to get data for training")
        return None
    
    # Prepare features
    data, feature_cols = prepare_features(btc_data)
    if data is None or feature_cols is None:
        logger.error("Failed to prepare features")
        return None
    
    try:
        # Define features and target
        X = data[feature_cols]
        y = data['Target']
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split - use most recent 70% for training
        train_size = int(len(data) * 0.7)
        X_train = X_scaled[-train_size:]
        X_test = X_scaled[:-train_size]
        y_train = y.iloc[-train_size:]
        y_test = y.iloc[:-train_size]        
        # Train-test split

        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=10,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model training completed with accuracy: {accuracy:.4f}")
        
        # Save model info
        model_info = {
            'model': model,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'trained_at': datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save to file
        joblib.dump(model_info, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return None

def load_model():
    """
    Load the trained model
    
    Returns:
        dict or None: Model information dictionary, or newly trained model if loading fails
    """
    if not os.path.exists(MODEL_PATH):
        logger.info("No model found. Training a new one.")
        return train_model()
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model_info = joblib.load(MODEL_PATH)
        return model_info
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.info("Training a new model instead")
        return train_model()

def get_latest_prediction():
    """
    Get prediction for the next day's Bitcoin price direction
    
    Returns:
        dict or None: Prediction results, or None if prediction fails
    """
    # Load model
    model_info = load_model()
    if model_info is None:
        logger.error("Failed to load or train model for prediction")
        return None
    
    # Fetch recent data
    btc_data = fetch_bitcoin_data(days=30)  # Just need recent data
    if btc_data is None:
        logger.error("Failed to fetch recent data for prediction")
        return None
    
    try:
        # Prepare features
        data, _ = prepare_features(btc_data)
        if data is None or len(data) == 0:
            logger.error("Failed to prepare features for prediction")
            return None
        
        # Get the most recent feature values
        latest_features = data.iloc[-1][model_info['feature_cols']]
        
        # Make prediction
        model = model_info['model']
        probabilities = model.predict_proba([latest_features])[0]
        prediction = int(probabilities[1] > 0.5)  # 1 if up, 0 if down
        
        # Get current price
        current_price = btc_data['Close'].iloc[-1]
        
        return {
            'prediction': 'up' if prediction == 1 else 'down',
            'confidence': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'probability_up': float(probabilities[1]),
            'probability_down': float(probabilities[0]),
            'current_price': float(current_price),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None