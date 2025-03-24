import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def fetch_bitcoin_data(days=365):
    """
    Fetch Bitcoin price data from Yahoo Finance
    
    Args:
        days (int): Number of days of historical data to fetch
        
    Returns:
        pandas.DataFrame or None: DataFrame with Bitcoin price data, or None if fetch fails
    """
    logger.info(f"Fetching Bitcoin data for the past {days} days")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Download data
        btc_data = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")
        
        if btc_data.empty:
            logger.error("Failed to fetch Bitcoin data")
            return None
            
        logger.info(f"Successfully fetched {len(btc_data)} days of Bitcoin data")
        
        # Validate data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in btc_data.columns:
                logger.error(f"Required column {col} missing from fetched data")
                return None
        
        # Check for missing values
        if btc_data[required_columns].isna().any().any():
            logger.warning("Fetched data contains missing values")
            # Fill missing values
            btc_data = btc_data.fillna(method='ffill')
            if btc_data[required_columns].isna().any().any():
                logger.error("Unable to handle all missing values in data")
                return None
        
        return btc_data
        
    except Exception as e:
        logger.error(f"Error fetching Bitcoin data: {str(e)}")
        return None

def prepare_features(df):
    """
    Prepare features for prediction
    
    Args:
        df (pandas.DataFrame): Raw Bitcoin price data
        
    Returns:
        tuple: (processed_dataframe, feature_columns) or (None, None) if processing fails
    """
    if df is None or df.empty:
        logger.error("Cannot prepare features from empty or None dataframe")
        return None, None
        
    try:
        # Create a copy of the dataframe
        data = df.copy()
        
        # Add a more meaningful target: significant price movements (>1%)
        price_change = data['Close'].pct_change().shift(-1)
        data['Target'] = ((price_change > 0.01) | (price_change < -0.01)).astype(int)
        
        # Keep our existing indicators
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        
        # Add RSI
        delta = data['Close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        short_ema = data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = short_ema - long_ema
        
        # Add volatility
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
        
        # Add volume features
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
        
        # Add price momentum
        data['Price_Change_1d'] = data['Close'].pct_change()
        data['Price_Change_3d'] = data['Close'].pct_change(periods=3)
        data['Price_Change_7d'] = data['Close'].pct_change(periods=7)
        
        # Add day of week (market can behave differently on weekends)
        data['Day_Of_Week'] = data.index.dayofweek
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Define features
        feature_cols = [
            'MA5', 'MA20', 'RSI', 'MACD', 'Volatility',
            'Volume_Change', 'Volume_MA5', 'Volume_MA20',
            'Price_Change_1d', 'Price_Change_3d', 'Price_Change_7d',
            'Day_Of_Week'
        ]
        
        return data, feature_cols
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None, None