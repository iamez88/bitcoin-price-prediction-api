# Bitcoin Price Direction Prediction API

A RESTful API for predicting Bitcoin price movement built with Flask and containerized with Docker.

## Tech Stack

- Python 3.11
- Flask: Web framework
- Gunicorn: WSGI HTTP Server
- pandas & numpy: Data processing
- scikit-learn & XGBoost: Machine learning
- yfinance: Yahoo Finance data retrieval
- Docker: Containerization

## Features

- Bitcoin price direction prediction (UP/DOWN)
- Data validation & sanitation
- Comprehensive logging
- Rate limiting (10 requests per minute)
- Containerized deployment

## API Endpoints

- `/` - API information
- `/docs` - API documentation (Swagger UI)
- `/api/health` - Health check
- `/api/predict` - Get price direction prediction
- `/api/model/info` - Get model information

## Getting Started

### Prerequisites

- Docker installed on your machine

### Running with Docker

1. Clone this repository:
   ```
   git clone https://github.com/your-username/bitcoin-price-prediction-api.git
   cd bitcoin-price-prediction-api
   ```

2. Build the Docker image:
   ```
   docker build -t bitcoin-prediction-api .
   ```

3. Run the container:
   ```
   docker run -p 50505:50505 bitcoin-prediction-api
   ```

4. Access the API at `http://localhost:50505`

### Making API Requests

Example using curl:
```
# Get API information
curl http://localhost:50505/

# Get a prediction
curl http://localhost:50505/api/predict

# Check model information
curl http://localhost:50505/api/model/info
``` 