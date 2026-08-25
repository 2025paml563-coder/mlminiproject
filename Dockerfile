# Build stage
FROM python:3.13.15-slim as builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY linear_regression_model_artifacts/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.13.15-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy model artifacts
COPY linear_regression_model_artifacts /app/model

# Set PATH to use locally installed packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port for MLflow model server
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# Create Flask app for model serving
RUN cat > /app/serve.py << 'EOF'
import json
from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)
model = mlflow.pyfunc.load_model("/app/model")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        predictions = model.predict(data)
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
EOF

# Start Flask server
CMD ["python", "/app/serve.py"]
