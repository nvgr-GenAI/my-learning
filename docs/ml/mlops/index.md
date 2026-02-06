# MLOps - Machine Learning Operations

## Overview

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML models in production reliably and efficiently. It encompasses the entire ML lifecycle from development to deployment, monitoring, and governance.

**Key Objectives:**
- Automate ML workflows and pipelines
- Enable continuous training and deployment
- Monitor model performance in production
- Ensure reproducibility and versioning
- Implement governance and compliance
- Scale ML systems efficiently

---

## MLOps Lifecycle

=== "Development"

    ### Experiment Tracking

    Track experiments, parameters, metrics, and artifacts to ensure reproducibility.

    **MLflow Experiment Tracking Example:**

    ```python
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    # Set tracking URI (local or remote)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("customer-churn-prediction")

    # Load data
    data = pd.read_csv("customer_data.csv")
    X = data.drop("churn", axis=1)
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run(run_name="rf-baseline"):
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="churn-predictor"
        )

        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")

        # Log artifacts
        mlflow.log_artifact("preprocessing_config.yaml")

        print(f"Model logged with run_id: {mlflow.active_run().info.run_id}")
    ```

    **Weights & Biases Integration:**

    ```python
    import wandb
    from wandb.integration.keras import WandbCallback
    import tensorflow as tf

    # Initialize W&B
    wandb.init(
        project="image-classification",
        config={
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "architecture": "ResNet50"
        }
    )

    config = wandb.config

    # Build model
    model = tf.keras.applications.ResNet50(
        weights=None,
        input_shape=(224, 224, 3),
        classes=10
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train with W&B callback
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=[
            WandbCallback(
                save_model=True,
                log_weights=True,
                log_gradients=True
            )
        ]
    )

    # Log custom metrics
    wandb.log({
        "validation_accuracy": history.history['val_accuracy'][-1],
        "validation_loss": history.history['val_loss'][-1]
    })

    wandb.finish()
    ```

    ### Data Versioning with DVC

    **DVC Setup and Workflow:**

    ```bash
    # Initialize DVC
    dvc init

    # Add remote storage (S3, GCS, Azure, etc.)
    dvc remote add -d myremote s3://my-bucket/dvc-storage

    # Track data file
    dvc add data/train.csv
    git add data/train.csv.dvc data/.gitignore
    git commit -m "Add training data"

    # Push data to remote storage
    dvc push

    # Pull data on another machine
    dvc pull
    ```

    **DVC Pipeline Definition:**

    ```yaml
    # dvc.yaml
    stages:
      preprocess:
        cmd: python src/preprocess.py
        deps:
          - data/raw/data.csv
          - src/preprocess.py
        params:
          - preprocess.test_split
          - preprocess.random_state
        outs:
          - data/processed/train.csv
          - data/processed/test.csv

      train:
        cmd: python src/train.py
        deps:
          - data/processed/train.csv
          - src/train.py
        params:
          - train.n_estimators
          - train.max_depth
        outs:
          - models/model.pkl
        metrics:
          - metrics/train_metrics.json:
              cache: false

      evaluate:
        cmd: python src/evaluate.py
        deps:
          - data/processed/test.csv
          - models/model.pkl
          - src/evaluate.py
        metrics:
          - metrics/test_metrics.json:
              cache: false
        plots:
          - plots/confusion_matrix.png
          - plots/roc_curve.png
    ```

    **params.yaml:**

    ```yaml
    preprocess:
      test_split: 0.2
      random_state: 42

    train:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 5
      learning_rate: 0.01

    evaluate:
      threshold: 0.5
    ```

    ### Feature Engineering Pipeline

    ```python
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import numpy as np

    class DateFeatureExtractor(BaseEstimator, TransformerMixin):
        """Extract date-based features."""

        def __init__(self, date_column):
            self.date_column = date_column

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            X[self.date_column] = pd.to_datetime(X[self.date_column])

            X['year'] = X[self.date_column].dt.year
            X['month'] = X[self.date_column].dt.month
            X['day'] = X[self.date_column].dt.day
            X['dayofweek'] = X[self.date_column].dt.dayofweek
            X['quarter'] = X[self.date_column].dt.quarter
            X['is_weekend'] = X['dayofweek'].isin([5, 6]).astype(int)

            X = X.drop(self.date_column, axis=1)
            return X

    class OutlierHandler(BaseEstimator, TransformerMixin):
        """Handle outliers using IQR method."""

        def __init__(self, columns, method='clip'):
            self.columns = columns
            self.method = method
            self.lower_bounds = {}
            self.upper_bounds = {}

        def fit(self, X, y=None):
            for col in self.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds[col] = Q1 - 1.5 * IQR
                self.upper_bounds[col] = Q3 + 1.5 * IQR
            return self

        def transform(self, X):
            X = X.copy()
            for col in self.columns:
                if self.method == 'clip':
                    X[col] = X[col].clip(
                        lower=self.lower_bounds[col],
                        upper=self.upper_bounds[col]
                    )
                elif self.method == 'remove':
                    mask = (X[col] >= self.lower_bounds[col]) & \
                           (X[col] <= self.upper_bounds[col])
                    X = X[mask]
            return X

    # Feature engineering pipeline
    feature_pipeline = Pipeline([
        ('date_features', DateFeatureExtractor('transaction_date')),
        ('outlier_handling', OutlierHandler(['amount', 'age'], method='clip')),
        ('scaling', StandardScaler())
    ])
    ```

=== "Deployment"

    ### Model Packaging with Docker

    **Dockerfile for ML Model:**

    ```dockerfile
    # Dockerfile
    FROM python:3.9-slim

    # Set working directory
    WORKDIR /app

    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        && rm -rf /var/lib/apt/lists/*

    # Copy requirements and install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY src/ ./src/
    COPY models/ ./models/
    COPY config/ ./config/

    # Create non-root user
    RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
    USER mluser

    # Expose port
    EXPOSE 8000

    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
        CMD curl -f http://localhost:8000/health || exit 1

    # Run application
    CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

    **requirements.txt:**

    ```text
    fastapi==0.104.1
    uvicorn[standard]==0.24.0
    pydantic==2.5.0
    scikit-learn==1.3.2
    pandas==2.1.3
    numpy==1.26.2
    joblib==1.3.2
    prometheus-client==0.19.0
    python-json-logger==2.0.7
    ```

    **Docker Compose for Development:**

    ```yaml
    # docker-compose.yml
    version: '3.8'

    services:
      ml-api:
        build: .
        container_name: ml-api
        ports:
          - "8000:8000"
        environment:
          - MODEL_PATH=/app/models/model.pkl
          - LOG_LEVEL=INFO
          - ENVIRONMENT=development
        volumes:
          - ./models:/app/models:ro
          - ./logs:/app/logs
        restart: unless-stopped

      prometheus:
        image: prom/prometheus:latest
        container_name: prometheus
        ports:
          - "9090:9090"
        volumes:
          - ./prometheus.yml:/etc/prometheus/prometheus.yml
          - prometheus_data:/prometheus
        command:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'

      grafana:
        image: grafana/grafana:latest
        container_name: grafana
        ports:
          - "3000:3000"
        environment:
          - GF_SECURITY_ADMIN_PASSWORD=admin
        volumes:
          - grafana_data:/var/lib/grafana
          - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
        depends_on:
          - prometheus

    volumes:
      prometheus_data:
      grafana_data:
    ```

    ### FastAPI Model Serving

    ```python
    # src/main.py
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    from prometheus_client import Counter, Histogram, generate_latest
    from contextlib import asynccontextmanager
    import joblib
    import numpy as np
    import pandas as pd
    import time
    import logging
    from typing import List, Dict, Any

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Prometheus metrics
    PREDICTION_COUNTER = Counter(
        'model_predictions_total',
        'Total number of predictions',
        ['model_version', 'status']
    )
    PREDICTION_LATENCY = Histogram(
        'model_prediction_latency_seconds',
        'Prediction latency in seconds',
        ['model_version']
    )

    # Global model variable
    model = None
    model_version = "1.0.0"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load model on startup, cleanup on shutdown."""
        global model
        try:
            logger.info("Loading model...")
            model = joblib.load("models/model.pkl")
            logger.info(f"Model {model_version} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        yield
        # Cleanup
        logger.info("Shutting down application")

    app = FastAPI(
        title="ML Model API",
        description="Production ML model serving API",
        version=model_version,
        lifespan=lifespan
    )

    # Request/Response models
    class PredictionRequest(BaseModel):
        """Input schema for predictions."""
        features: List[float] = Field(..., min_items=1, max_items=100)

        @validator('features')
        def validate_features(cls, v):
            if any(not isinstance(x, (int, float)) for x in v):
                raise ValueError('All features must be numeric')
            if any(np.isnan(x) or np.isinf(x) for x in v):
                raise ValueError('Features cannot contain NaN or Inf values')
            return v

    class BatchPredictionRequest(BaseModel):
        """Batch prediction request."""
        instances: List[List[float]] = Field(..., min_items=1, max_items=1000)

    class PredictionResponse(BaseModel):
        """Prediction response schema."""
        prediction: float
        probability: float
        model_version: str
        latency_ms: float

    class BatchPredictionResponse(BaseModel):
        """Batch prediction response."""
        predictions: List[Dict[str, Any]]
        total_count: int
        model_version: str
        total_latency_ms: float

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "status": "healthy",
            "model_version": model_version,
            "model_loaded": model is not None
        }

    # Readiness check
    @app.get("/ready")
    async def readiness_check():
        """Readiness check endpoint."""
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "reason": "Model not loaded"}
            )
        return {"status": "ready", "model_version": model_version}

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return generate_latest()

    # Single prediction endpoint
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest):
        """Make a single prediction."""
        start_time = time.time()

        try:
            # Prepare input
            X = np.array(request.features).reshape(1, -1)

            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0].max()

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Record metrics
            PREDICTION_COUNTER.labels(
                model_version=model_version,
                status='success'
            ).inc()
            PREDICTION_LATENCY.labels(
                model_version=model_version
            ).observe(latency_ms / 1000)

            return PredictionResponse(
                prediction=float(prediction),
                probability=float(probability),
                model_version=model_version,
                latency_ms=latency_ms
            )

        except Exception as e:
            PREDICTION_COUNTER.labels(
                model_version=model_version,
                status='error'
            ).inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Batch prediction endpoint
    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        """Make batch predictions."""
        start_time = time.time()

        try:
            X = np.array(request.instances)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X).max(axis=1)

            results = [
                {
                    "prediction": float(pred),
                    "probability": float(prob)
                }
                for pred, prob in zip(predictions, probabilities)
            ]

            latency_ms = (time.time() - start_time) * 1000

            PREDICTION_COUNTER.labels(
                model_version=model_version,
                status='success'
            ).inc(len(request.instances))

            return BatchPredictionResponse(
                predictions=results,
                total_count=len(results),
                model_version=model_version,
                total_latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Model info endpoint
    @app.get("/model/info")
    async def model_info():
        """Get model information."""
        return {
            "model_version": model_version,
            "model_type": type(model).__name__,
            "feature_count": model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
            "classes": model.classes_.tolist() if hasattr(model, 'classes_') else None
        }
    ```

    ### Serving Patterns

    **Real-time Serving (Synchronous):**

    ```python
    # Real-time prediction endpoint
    @app.post("/predict/realtime")
    async def predict_realtime(request: PredictionRequest):
        """Real-time synchronous prediction."""
        # Low latency requirement: < 100ms
        timeout = 0.1
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                make_prediction(request.features),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Prediction timeout exceeded"
            )
    ```

    **Batch Serving (Asynchronous):**

    ```python
    from celery import Celery
    import redis

    # Initialize Celery
    celery_app = Celery(
        'ml_tasks',
        broker='redis://localhost:6379/0',
        backend='redis://localhost:6379/0'
    )

    @celery_app.task(bind=True, max_retries=3)
    def batch_predict_task(self, data: List[List[float]]):
        """Async batch prediction task."""
        try:
            X = np.array(data)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'status': 'completed'
            }
        except Exception as e:
            self.retry(exc=e, countdown=60)

    @app.post("/predict/async")
    async def predict_async(request: BatchPredictionRequest):
        """Submit async batch prediction job."""
        task = batch_predict_task.delay(request.instances)
        return {
            "job_id": task.id,
            "status": "submitted",
            "status_url": f"/jobs/{task.id}"
        }

    @app.get("/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """Check job status."""
        task = celery_app.AsyncResult(job_id)

        if task.ready():
            return {
                "job_id": job_id,
                "status": "completed",
                "result": task.result
            }
        else:
            return {
                "job_id": job_id,
                "status": "processing"
            }
    ```

    **Streaming Predictions:**

    ```python
    from fastapi.responses import StreamingResponse
    import json

    @app.post("/predict/stream")
    async def predict_stream(request: Request):
        """Stream predictions for large datasets."""

        async def generate_predictions():
            async for line in request.stream():
                try:
                    data = json.loads(line)
                    features = data.get('features', [])

                    X = np.array(features).reshape(1, -1)
                    prediction = model.predict(X)[0]

                    result = {
                        'prediction': float(prediction),
                        'id': data.get('id')
                    }
                    yield json.dumps(result) + '\n'
                except Exception as e:
                    logger.error(f"Stream prediction error: {e}")

        return StreamingResponse(
            generate_predictions(),
            media_type='application/x-ndjson'
        )
    ```

    ### A/B Testing Framework

    ```python
    import random
    from enum import Enum
    from typing import Optional

    class ModelVariant(str, Enum):
        CONTROL = "control"
        VARIANT_A = "variant_a"
        VARIANT_B = "variant_b"

    class ABTestConfig:
        """A/B test configuration."""
        def __init__(self):
            self.variants = {
                ModelVariant.CONTROL: {
                    'weight': 0.5,
                    'model_path': 'models/model_v1.pkl'
                },
                ModelVariant.VARIANT_A: {
                    'weight': 0.25,
                    'model_path': 'models/model_v2.pkl'
                },
                ModelVariant.VARIANT_B: {
                    'weight': 0.25,
                    'model_path': 'models/model_v3.pkl'
                }
            }
            self.models = {}
            self._load_models()

        def _load_models(self):
            """Load all model variants."""
            for variant, config in self.variants.items():
                self.models[variant] = joblib.load(config['model_path'])

        def assign_variant(self, user_id: str) -> ModelVariant:
            """Assign user to a variant (sticky assignment)."""
            # Use hash for deterministic assignment
            hash_value = hash(user_id)
            rand_val = (hash_value % 100) / 100

            cumulative = 0
            for variant, config in self.variants.items():
                cumulative += config['weight']
                if rand_val < cumulative:
                    return variant

            return ModelVariant.CONTROL

    # Initialize A/B test
    ab_test = ABTestConfig()

    @app.post("/predict/ab")
    async def predict_ab(
        request: PredictionRequest,
        user_id: str,
        variant: Optional[ModelVariant] = None
    ):
        """Prediction with A/B testing."""

        # Assign variant if not provided
        if variant is None:
            variant = ab_test.assign_variant(user_id)

        # Get model for variant
        model = ab_test.models[variant]

        # Make prediction
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()

        # Log A/B test assignment
        logger.info(f"User {user_id} assigned to {variant}")

        return {
            'prediction': float(prediction),
            'probability': float(probability),
            'variant': variant,
            'user_id': user_id
        }
    ```

=== "Monitoring"

    ### Prometheus Metrics Collection

    **prometheus.yml Configuration:**

    ```yaml
    # prometheus.yml
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'ml-production'
        environment: 'prod'

    # Alertmanager configuration
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
                - alertmanager:9093

    # Load rules
    rule_files:
      - "alerts.yml"

    # Scrape configurations
    scrape_configs:
      # ML API metrics
      - job_name: 'ml-api'
        static_configs:
          - targets: ['ml-api:8000']
        metrics_path: '/metrics'

      # Node exporter (system metrics)
      - job_name: 'node'
        static_configs:
          - targets: ['node-exporter:9100']

      # Kubernetes pods (if using K8s)
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
    ```

    **Custom Metrics for ML Models:**

    ```python
    from prometheus_client import Counter, Histogram, Gauge, Summary
    import numpy as np
    from collections import deque

    class ModelMetrics:
        """Custom metrics for ML model monitoring."""

        def __init__(self, model_name: str, window_size: int = 1000):
            self.model_name = model_name
            self.window_size = window_size

            # Prediction metrics
            self.predictions_total = Counter(
                f'{model_name}_predictions_total',
                'Total predictions',
                ['model_version', 'status']
            )

            self.prediction_latency = Histogram(
                f'{model_name}_prediction_latency_seconds',
                'Prediction latency',
                ['model_version'],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            )

            # Model performance metrics
            self.prediction_confidence = Histogram(
                f'{model_name}_prediction_confidence',
                'Prediction confidence distribution',
                ['model_version'],
                buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
            )

            self.feature_statistics = Gauge(
                f'{model_name}_feature_statistics',
                'Feature statistics',
                ['feature_name', 'statistic']
            )

            # Data quality metrics
            self.missing_features = Counter(
                f'{model_name}_missing_features_total',
                'Missing features count',
                ['feature_name']
            )

            self.outlier_count = Counter(
                f'{model_name}_outliers_total',
                'Outlier count',
                ['feature_name']
            )

            # Drift detection metrics
            self.feature_drift_score = Gauge(
                f'{model_name}_feature_drift_score',
                'Feature drift score',
                ['feature_name']
            )

            self.prediction_drift_score = Gauge(
                f'{model_name}_prediction_drift_score',
                'Prediction drift score'
            )

            # Windows for drift detection
            self.prediction_window = deque(maxlen=window_size)
            self.feature_windows = {}

    # Initialize metrics
    model_metrics = ModelMetrics('churn_predictor')

    @app.middleware("http")
    async def monitor_requests(request: Request, call_next):
        """Middleware to monitor all requests."""
        start_time = time.time()

        response = await call_next(request)

        # Record latency
        latency = time.time() - start_time
        model_metrics.prediction_latency.labels(
            model_version=model_version
        ).observe(latency)

        return response
    ```

    **Alert Rules:**

    ```yaml
    # alerts.yml
    groups:
      - name: ml_model_alerts
        interval: 30s
        rules:
          # High error rate
          - alert: HighPredictionErrorRate
            expr: |
              rate(model_predictions_total{status="error"}[5m]) /
              rate(model_predictions_total[5m]) > 0.05
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High prediction error rate"
              description: "Error rate is {{ $value | humanizePercentage }}"

          # High latency
          - alert: HighPredictionLatency
            expr: |
              histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m])) > 0.5
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High prediction latency"
              description: "P95 latency is {{ $value }}s"

          # Low prediction confidence
          - alert: LowPredictionConfidence
            expr: |
              histogram_quantile(0.5, rate(churn_predictor_prediction_confidence_bucket[5m])) < 0.7
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "Low prediction confidence"
              description: "Median confidence is {{ $value }}"

          # Feature drift detected
          - alert: FeatureDriftDetected
            expr: churn_predictor_feature_drift_score > 0.5
            for: 15m
            labels:
              severity: warning
            annotations:
              summary: "Feature drift detected"
              description: "Drift score for {{ $labels.feature_name }}: {{ $value }}"

          # Model not responding
          - alert: ModelNotResponding
            expr: up{job="ml-api"} == 0
            for: 2m
            labels:
              severity: critical
            annotations:
              summary: "ML API is down"
              description: "ML API has been down for 2 minutes"
    ```

    ### Data Drift Detection

    ```python
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    class DriftDetector:
        """Detect data and prediction drift."""

        def __init__(self, reference_data: pd.DataFrame):
            self.reference_data = reference_data
            self.reference_stats = self._calculate_statistics(reference_data)

        def _calculate_statistics(self, data: pd.DataFrame) -> dict:
            """Calculate reference statistics."""
            stats_dict = {}
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    stats_dict[col] = {
                        'mean': data[col].mean(),
                        'std': data[col].std(),
                        'min': data[col].min(),
                        'max': data[col].max(),
                        'quantiles': data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                    }
            return stats_dict

        def calculate_psi(
            self,
            reference: np.ndarray,
            current: np.ndarray,
            bins: int = 10
        ) -> float:
            """Calculate Population Stability Index (PSI)."""
            # Create bins based on reference data
            breakpoints = np.linspace(
                reference.min(),
                reference.max(),
                bins + 1
            )

            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=breakpoints)
            cur_counts, _ = np.histogram(current, bins=breakpoints)

            # Normalize to get percentages
            ref_pct = (ref_counts + 1) / (len(reference) + bins)
            cur_pct = (cur_counts + 1) / (len(current) + bins)

            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

            return psi

        def calculate_ks_statistic(
            self,
            reference: np.ndarray,
            current: np.ndarray
        ) -> tuple:
            """Calculate Kolmogorov-Smirnov statistic."""
            statistic, p_value = stats.ks_2samp(reference, current)
            return statistic, p_value

        def detect_drift(
            self,
            current_data: pd.DataFrame,
            threshold: float = 0.2
        ) -> dict:
            """Detect drift in current data."""
            drift_report = {}

            for col in current_data.columns:
                if col in self.reference_stats:
                    ref_col = self.reference_data[col].values
                    cur_col = current_data[col].values

                    # Calculate PSI
                    psi = self.calculate_psi(ref_col, cur_col)

                    # Calculate KS statistic
                    ks_stat, p_value = self.calculate_ks_statistic(ref_col, cur_col)

                    drift_report[col] = {
                        'psi': psi,
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'drift_detected': psi > threshold,
                        'current_mean': cur_col.mean(),
                        'reference_mean': self.reference_stats[col]['mean'],
                        'mean_difference': abs(
                            cur_col.mean() - self.reference_stats[col]['mean']
                        )
                    }

                    # Update Prometheus metrics
                    model_metrics.feature_drift_score.labels(
                        feature_name=col
                    ).set(psi)

            return drift_report

    # Initialize drift detector with training data
    drift_detector = DriftDetector(training_data)

    @app.post("/monitor/drift")
    async def check_drift(data: List[Dict[str, Any]]):
        """Check for data drift."""
        current_data = pd.DataFrame(data)
        drift_report = drift_detector.detect_drift(current_data)

        # Flag drifted features
        drifted_features = [
            col for col, metrics in drift_report.items()
            if metrics['drift_detected']
        ]

        return {
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'drift_report': drift_report
        }
    ```

    ### Model Performance Monitoring

    ```python
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    class ModelPerformanceMonitor:
        """Monitor model performance over time."""

        def __init__(self, window_size: int = 1000):
            self.window_size = window_size
            self.predictions = deque(maxlen=window_size)
            self.actuals = deque(maxlen=window_size)
            self.timestamps = deque(maxlen=window_size)

        def record_prediction(
            self,
            prediction: float,
            actual: float = None,
            timestamp: float = None
        ):
            """Record a prediction and actual value."""
            self.predictions.append(prediction)
            if actual is not None:
                self.actuals.append(actual)
            self.timestamps.append(timestamp or time.time())

        def calculate_metrics(self) -> dict:
            """Calculate performance metrics."""
            if len(self.actuals) < 100:
                return {"error": "Insufficient data for metrics"}

            y_true = np.array(list(self.actuals))
            y_pred = np.array(list(self.predictions))

            # Binary classification metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'sample_count': len(y_true)
            }

            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)

            return metrics

        def detect_performance_degradation(
            self,
            baseline_metrics: dict,
            threshold: float = 0.05
        ) -> dict:
            """Detect performance degradation."""
            current_metrics = self.calculate_metrics()

            degradation_report = {}
            for metric, baseline_value in baseline_metrics.items():
                if metric in current_metrics:
                    current_value = current_metrics[metric]
                    difference = baseline_value - current_value
                    degradation_report[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'difference': difference,
                        'degraded': difference > threshold
                    }

            return degradation_report

    # Initialize performance monitor
    perf_monitor = ModelPerformanceMonitor()

    @app.post("/feedback")
    async def submit_feedback(
        prediction_id: str,
        actual_value: float,
        prediction_value: float
    ):
        """Submit ground truth feedback for monitoring."""
        perf_monitor.record_prediction(prediction_value, actual_value)

        # Calculate recent metrics
        metrics = perf_monitor.calculate_metrics()

        return {
            'status': 'recorded',
            'recent_metrics': metrics
        }
    ```

    ### Grafana Dashboard Configuration

    ```json
    {
      "dashboard": {
        "title": "ML Model Monitoring",
        "panels": [
          {
            "id": 1,
            "title": "Prediction Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(model_predictions_total[5m])",
                "legendFormat": "{{status}}"
              }
            ]
          },
          {
            "id": 2,
            "title": "Prediction Latency (P95)",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))",
                "legendFormat": "P95 Latency"
              }
            ]
          },
          {
            "id": 3,
            "title": "Error Rate",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(model_predictions_total{status='error'}[5m]) / rate(model_predictions_total[5m])"
              }
            ],
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 0.01, "color": "yellow"},
                {"value": 0.05, "color": "red"}
              ]
            }
          },
          {
            "id": 4,
            "title": "Feature Drift Scores",
            "type": "heatmap",
            "targets": [
              {
                "expr": "churn_predictor_feature_drift_score"
              }
            ]
          }
        ]
      }
    }
    ```

=== "Governance"

    ### Model Registry

    ```python
    from mlflow.tracking import MlflowClient
    from mlflow.entities.model_registry import ModelVersion
    from datetime import datetime
    from typing import Optional, List

    class ModelRegistry:
        """Centralized model registry management."""

        def __init__(self, tracking_uri: str):
            self.client = MlflowClient(tracking_uri)

        def register_model(
            self,
            model_uri: str,
            model_name: str,
            tags: dict = None,
            description: str = None
        ) -> ModelVersion:
            """Register a new model version."""
            # Register model
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                description=description,
                tags=tags or {}
            )

            # Add metadata
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="registered_at",
                value=datetime.now().isoformat()
            )

            return model_version

        def promote_model(
            self,
            model_name: str,
            version: str,
            stage: str
        ):
            """Promote model to a specific stage."""
            valid_stages = ["Staging", "Production", "Archived"]
            if stage not in valid_stages:
                raise ValueError(f"Stage must be one of {valid_stages}")

            # Transition model to new stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True
            )

            # Log transition
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=f"promoted_to_{stage.lower()}_at",
                value=datetime.now().isoformat()
            )

        def get_model_version(
            self,
            model_name: str,
            version: Optional[str] = None,
            stage: Optional[str] = None
        ) -> ModelVersion:
            """Get specific model version."""
            if version:
                return self.client.get_model_version(model_name, version)
            elif stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                return versions[0] if versions else None
            else:
                versions = self.client.get_latest_versions(model_name)
                return versions[0] if versions else None

        def list_models(self) -> List[str]:
            """List all registered models."""
            return [model.name for model in self.client.list_registered_models()]

        def get_model_metadata(self, model_name: str, version: str) -> dict:
            """Get comprehensive model metadata."""
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)

            return {
                'name': model_name,
                'version': version,
                'stage': model_version.current_stage,
                'description': model_version.description,
                'tags': model_version.tags,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'created_at': model_version.creation_timestamp,
                'run_id': model_version.run_id,
                'source': model_version.source
            }

    # Initialize registry
    registry = ModelRegistry("http://localhost:5000")

    # Example usage
    @app.post("/models/register")
    async def register_new_model(
        model_uri: str,
        model_name: str,
        description: str,
        tags: dict
    ):
        """Register a new model version."""
        model_version = registry.register_model(
            model_uri=model_uri,
            model_name=model_name,
            description=description,
            tags=tags
        )

        return {
            'model_name': model_name,
            'version': model_version.version,
            'status': 'registered'
        }

    @app.post("/models/{model_name}/promote")
    async def promote_model_version(
        model_name: str,
        version: str,
        stage: str
    ):
        """Promote model to production."""
        registry.promote_model(model_name, version, stage)

        return {
            'model_name': model_name,
            'version': version,
            'stage': stage,
            'status': 'promoted'
        }
    ```

    ### Model Approval Workflow

    ```python
    from enum import Enum
    from dataclasses import dataclass
    from typing import List, Optional

    class ApprovalStatus(str, Enum):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"

    @dataclass
    class ApprovalRequest:
        """Model approval request."""
        model_name: str
        version: str
        requester: str
        target_stage: str
        metrics: dict
        test_results: dict
        documentation_url: str
        status: ApprovalStatus = ApprovalStatus.PENDING
        approvers: List[str] = None
        comments: List[str] = None

    class ModelGovernance:
        """Model governance and approval workflow."""

        def __init__(self):
            self.approval_requests = {}
            self.required_approvers = ["ml_lead", "security_team", "compliance"]

        def submit_approval_request(
            self,
            model_name: str,
            version: str,
            requester: str,
            target_stage: str,
            metrics: dict,
            test_results: dict,
            documentation_url: str
        ) -> str:
            """Submit model for approval."""
            # Validate requirements
            self._validate_promotion_requirements(
                metrics, test_results, documentation_url
            )

            request_id = f"{model_name}_{version}_{int(time.time())}"

            approval_request = ApprovalRequest(
                model_name=model_name,
                version=version,
                requester=requester,
                target_stage=target_stage,
                metrics=metrics,
                test_results=test_results,
                documentation_url=documentation_url,
                approvers=[],
                comments=[]
            )

            self.approval_requests[request_id] = approval_request

            # Notify approvers
            self._notify_approvers(request_id, approval_request)

            return request_id

        def _validate_promotion_requirements(
            self,
            metrics: dict,
            test_results: dict,
            documentation_url: str
        ):
            """Validate promotion requirements."""
            # Check minimum metrics
            required_metrics = ['accuracy', 'precision', 'recall']
            for metric in required_metrics:
                if metric not in metrics:
                    raise ValueError(f"Missing required metric: {metric}")
                if metrics[metric] < 0.8:
                    raise ValueError(
                        f"{metric} {metrics[metric]} below threshold 0.8"
                    )

            # Check test results
            if not test_results.get('all_tests_passed'):
                raise ValueError("Not all tests passed")

            # Check documentation
            if not documentation_url:
                raise ValueError("Documentation URL required")

        def approve_request(
            self,
            request_id: str,
            approver: str,
            comment: str = None
        ):
            """Approve a model promotion request."""
            request = self.approval_requests.get(request_id)
            if not request:
                raise ValueError(f"Request {request_id} not found")

            if approver in request.approvers:
                raise ValueError(f"{approver} already approved")

            request.approvers.append(approver)
            if comment:
                request.comments.append(f"{approver}: {comment}")

            # Check if all required approvals received
            if all(
                approver in request.approvers
                for approver in self.required_approvers
            ):
                request.status = ApprovalStatus.APPROVED
                self._execute_promotion(request)

        def reject_request(
            self,
            request_id: str,
            rejector: str,
            reason: str
        ):
            """Reject a model promotion request."""
            request = self.approval_requests.get(request_id)
            if not request:
                raise ValueError(f"Request {request_id} not found")

            request.status = ApprovalStatus.REJECTED
            request.comments.append(f"{rejector} (REJECTED): {reason}")

        def _execute_promotion(self, request: ApprovalRequest):
            """Execute approved promotion."""
            registry.promote_model(
                model_name=request.model_name,
                version=request.version,
                stage=request.target_stage
            )

            # Log approval
            logger.info(
                f"Model {request.model_name} v{request.version} "
                f"promoted to {request.target_stage}"
            )

    # Initialize governance
    governance = ModelGovernance()

    @app.post("/governance/submit")
    async def submit_for_approval(
        model_name: str,
        version: str,
        requester: str,
        target_stage: str,
        metrics: dict,
        test_results: dict,
        documentation_url: str
    ):
        """Submit model for approval."""
        request_id = governance.submit_approval_request(
            model_name=model_name,
            version=version,
            requester=requester,
            target_stage=target_stage,
            metrics=metrics,
            test_results=test_results,
            documentation_url=documentation_url
        )

        return {
            'request_id': request_id,
            'status': 'submitted',
            'required_approvers': governance.required_approvers
        }

    @app.post("/governance/approve/{request_id}")
    async def approve_model(
        request_id: str,
        approver: str,
        comment: Optional[str] = None
    ):
        """Approve model promotion."""
        governance.approve_request(request_id, approver, comment)

        return {
            'request_id': request_id,
            'status': 'approved',
            'approver': approver
        }
    ```

    ### Compliance and Audit Trail

    ```python
    from typing import Any
    import json
    from datetime import datetime

    class AuditLogger:
        """Audit logging for ML operations."""

        def __init__(self, log_file: str = "audit.log"):
            self.log_file = log_file

        def log_event(
            self,
            event_type: str,
            user: str,
            resource: str,
            action: str,
            details: dict = None,
            status: str = "success"
        ):
            """Log an audit event."""
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'user': user,
                'resource': resource,
                'action': action,
                'details': details or {},
                'status': status
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')

        def log_prediction(
            self,
            user: str,
            model_name: str,
            model_version: str,
            input_data: dict,
            prediction: Any,
            confidence: float
        ):
            """Log a prediction event."""
            self.log_event(
                event_type='prediction',
                user=user,
                resource=f"{model_name}:{model_version}",
                action='predict',
                details={
                    'input_hash': hash(str(input_data)),
                    'prediction': prediction,
                    'confidence': confidence
                }
            )

        def log_model_update(
            self,
            user: str,
            model_name: str,
            old_version: str,
            new_version: str,
            reason: str
        ):
            """Log a model update event."""
            self.log_event(
                event_type='model_update',
                user=user,
                resource=model_name,
                action='update',
                details={
                    'old_version': old_version,
                    'new_version': new_version,
                    'reason': reason
                }
            )

        def log_data_access(
            self,
            user: str,
            dataset: str,
            action: str,
            record_count: int = None
        ):
            """Log data access event."""
            self.log_event(
                event_type='data_access',
                user=user,
                resource=dataset,
                action=action,
                details={
                    'record_count': record_count
                }
            )

    # Initialize audit logger
    audit_logger = AuditLogger()

    @app.middleware("http")
    async def audit_middleware(request: Request, call_next):
        """Audit middleware for tracking requests."""
        # Extract user from auth header (simplified)
        user = request.headers.get('X-User-ID', 'anonymous')

        # Log request
        audit_logger.log_event(
            event_type='api_request',
            user=user,
            resource=request.url.path,
            action=request.method,
            details={
                'client_ip': request.client.host,
                'user_agent': request.headers.get('User-Agent')
            }
        )

        response = await call_next(request)

        return response
    ```

---

## MLOps Architecture

### End-to-End MLOps Pipeline

```mermaid
graph TB
    subgraph "Data Layer"
        A[Raw Data] --> B[Data Validation]
        B --> C[Data Preprocessing]
        C --> D[Feature Store]
    end

    subgraph "Development Layer"
        D --> E[Feature Engineering]
        E --> F[Model Training]
        F --> G[Experiment Tracking]
        G --> H[Model Evaluation]
    end

    subgraph "Registry Layer"
        H --> I{Meets Criteria?}
        I -->|Yes| J[Model Registry]
        I -->|No| F
        J --> K[Model Versioning]
    end

    subgraph "Deployment Layer"
        K --> L{Approval?}
        L -->|Approved| M[Staging]
        M --> N[A/B Testing]
        N --> O[Production]
        L -->|Rejected| F
    end

    subgraph "Monitoring Layer"
        O --> P[Model Monitoring]
        P --> Q[Performance Metrics]
        P --> R[Drift Detection]
        Q --> S{Degradation?}
        R --> S
        S -->|Yes| T[Alert & Retrain]
        T --> F
    end

    style F fill:#e1f5ff
    style J fill:#fff4e1
    style O fill:#e8f5e9
    style P fill:#fce4ec
```

### CI/CD Pipeline for ML

```mermaid
graph LR
    A[Code Commit] --> B[Trigger CI]
    B --> C[Run Tests]
    C --> D{Tests Pass?}
    D -->|Yes| E[Build Container]
    D -->|No| F[Notify Developer]
    E --> G[Push to Registry]
    G --> H[Deploy to Staging]
    H --> I[Run Integration Tests]
    I --> J{Tests Pass?}
    J -->|Yes| K[Manual Approval]
    J -->|No| F
    K --> L{Approved?}
    L -->|Yes| M[Deploy to Production]
    L -->|No| N[Cancel Deployment]
    M --> O[Health Check]
    O --> P{Healthy?}
    P -->|Yes| Q[Monitor]
    P -->|No| R[Rollback]
    R --> S[Previous Version]

    style C fill:#e3f2fd
    style E fill:#fff3e0
    style M fill:#e8f5e9
    style R fill:#ffebee
```

### Model Versioning Workflow

```mermaid
graph TD
    A[Train Model] --> B[Log Experiment]
    B --> C[Generate Metrics]
    C --> D{Meets Threshold?}
    D -->|Yes| E[Register Model]
    D -->|No| A
    E --> F[Version: v1.2.3]
    F --> G[Tag: Candidate]
    G --> H[Validation]
    H --> I{Validated?}
    I -->|Yes| J[Tag: Staging]
    I -->|No| K[Tag: Failed]
    J --> L[Business Review]
    L --> M{Approved?}
    M -->|Yes| N[Tag: Production]
    M -->|No| K
    N --> O[Deploy]
    O --> P[Monitor]
    P --> Q{Performance OK?}
    Q -->|Yes| R[Keep Active]
    Q -->|No| S[Tag: Archived]
    S --> T[Rollback to Previous]

    style E fill:#e1f5ff
    style J fill:#fff4e1
    style N fill:#e8f5e9
    style S fill:#ffebee
```

### Monitoring and Alerting Architecture

```mermaid
graph TB
    subgraph "Data Collection"
        A[ML Service] --> B[Prometheus Exporter]
        A --> C[Application Logs]
        A --> D[Prediction Logs]
    end

    subgraph "Metrics Storage"
        B --> E[Prometheus]
        C --> F[Elasticsearch]
        D --> G[S3/Data Lake]
    end

    subgraph "Analysis"
        E --> H[Grafana Dashboards]
        F --> I[Kibana]
        G --> J[Drift Detection]
        J --> K[Performance Analysis]
    end

    subgraph "Alerting"
        H --> L{Threshold Breach?}
        K --> L
        L -->|Yes| M[Alertmanager]
        M --> N[PagerDuty]
        M --> O[Slack]
        M --> P[Email]
    end

    subgraph "Actions"
        N --> Q[On-Call Engineer]
        O --> R[Team Channel]
        Q --> S[Investigate]
        R --> S
        S --> T{Root Cause?}
        T -->|Data Drift| U[Retrain Model]
        T -->|Bug| V[Fix & Deploy]
        T -->|Infrastructure| W[Scale Resources]
    end

    style B fill:#e3f2fd
    style E fill:#fff3e0
    style M fill:#ffebee
    style S fill:#e8f5e9
```

### Feature Store Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[Batch Data]
        B[Streaming Data]
        C[External APIs]
    end

    subgraph "Feature Engineering"
        A --> D[Batch Processing]
        B --> E[Stream Processing]
        C --> F[API Integration]
        D --> G[Feature Transformation]
        E --> G
        F --> G
    end

    subgraph "Feature Store"
        G --> H[Feature Registry]
        H --> I[Offline Store]
        H --> J[Online Store]
        I --> K[(Data Warehouse)]
        J --> L[(Redis/DynamoDB)]
    end

    subgraph "Consumption"
        K --> M[Training Pipeline]
        L --> N[Serving Pipeline]
        M --> O[Model Training]
        N --> P[Real-time Inference]
    end

    subgraph "Monitoring"
        O --> Q[Feature Monitoring]
        P --> Q
        Q --> R[Drift Detection]
        Q --> S[Quality Metrics]
    end

    style H fill:#e1f5ff
    style K fill:#fff4e1
    style L fill:#e8f5e9
    style Q fill:#fce4ec
```

### Model Registry Flow

```mermaid
graph LR
    A[Model Training] --> B[Log to MLflow]
    B --> C[Create Model Version]
    C --> D[Register in Registry]
    D --> E{Stage}
    E --> F[None]
    E --> G[Staging]
    E --> H[Production]
    E --> I[Archived]

    F --> J[Development Testing]
    J --> K{Ready?}
    K -->|Yes| G
    K -->|No| A

    G --> L[Staging Validation]
    L --> M{Approved?}
    M -->|Yes| H
    M -->|No| F

    H --> N[Production Serving]
    N --> O[Monitor Performance]
    O --> P{Degraded?}
    P -->|Yes| Q[Trigger Retraining]
    P -->|No| N

    H --> R{New Version?}
    R -->|Yes| I

    Q --> A

    style D fill:#e1f5ff
    style G fill:#fff4e1
    style H fill:#e8f5e9
    style I fill:#ffebee
```

---

## Platform Comparisons

### MLOps Platforms Comparison

| Feature | AWS SageMaker | Google Vertex AI | Azure ML | Databricks | Kubeflow |
|---------|--------------|------------------|----------|------------|----------|
| **Deployment** | Excellent | Excellent | Excellent | Very Good | Good |
| **Experiment Tracking** | SageMaker Experiments | Vertex Experiments | Azure ML Runs | MLflow | Katib |
| **Model Registry** | Model Registry | Model Registry | Model Registry | MLflow | Custom |
| **Auto Scaling** | Yes | Yes | Yes | Yes | Kubernetes HPA |
| **Batch Inference** | Batch Transform | Batch Prediction | Batch Endpoints | Delta Live Tables | Custom Jobs |
| **Real-time Inference** | Yes, < 100ms | Yes, < 100ms | Yes, < 100ms | Model Serving | KServe |
| **Feature Store** | SageMaker FS | Vertex FS | Azure FS | Feature Store | Feast |
| **Pipeline Orchestration** | SageMaker Pipelines | Vertex Pipelines | Azure Pipelines | Workflows | KFP |
| **Data Labeling** | Ground Truth | Vertex Data Labeling | Azure Data Labeling | No | Custom |
| **AutoML** | Autopilot | Vertex AutoML | Automated ML | AutoML | Katib |
| **Monitoring** | Model Monitor | Model Monitoring | Model Monitor | Lakehouse Monitoring | Custom |
| **Cost** | $$$ | $$$ | $$$ | $$$-$$$$ | $ (infrastructure) |
| **Ease of Use** | High | High | High | Medium | Low-Medium |
| **Vendor Lock-in** | High | High | High | Medium | Low |
| **Best For** | AWS ecosystem | GCP ecosystem | Azure ecosystem | Unified analytics | Kubernetes-native |

### Model Serving Frameworks

| Framework | Language | Performance | Scalability | Features | Best For |
|-----------|----------|-------------|-------------|----------|----------|
| **TensorFlow Serving** | C++ | Excellent | Excellent | - gRPC/REST<br>- Batching<br>- Model versioning<br>- GPU support | TensorFlow models |
| **TorchServe** | Python/Java | Very Good | Very Good | - REST/gRPC<br>- Multi-model<br>- Metrics<br>- A/B testing | PyTorch models |
| **Seldon Core** | Python | Good | Excellent | - Multi-framework<br>- Advanced routing<br>- Explainability<br>- Drift detection | Kubernetes deployments |
| **BentoML** | Python | Very Good | Good | - Model packaging<br>- Adaptive batching<br>- Async inference<br>- Easy deployment | General purpose |
| **FastAPI** | Python | Good | Good | - Simple setup<br>- Auto docs<br>- Async support<br>- Type safety | Custom APIs |
| **NVIDIA Triton** | C++ | Excellent | Excellent | - Multi-framework<br>- Dynamic batching<br>- Model ensemble<br>- GPU optimization | High-performance GPU |
| **Ray Serve** | Python | Very Good | Excellent | - Distributed<br>- Python-native<br>- Model composition<br>- Autoscaling | Complex pipelines |
| **KServe** | Python | Good | Excellent | - K8s native<br>- Serverless<br>- Canary rollouts<br>- Explainability | Kubernetes |

### Monitoring Tools Comparison

| Tool | Metrics | Drift Detection | Alerting | Visualization | Integration | Cost |
|------|---------|----------------|----------|---------------|-------------|------|
| **Prometheus + Grafana** | Excellent | Manual | Excellent | Excellent | Universal | Free |
| **Evidently AI** | Good | Excellent | Basic | Good | Python libraries | Free/Paid |
| **WhyLabs** | Very Good | Excellent | Excellent | Good | API/SDK | Paid |
| **Weights & Biases** | Excellent | Good | Good | Excellent | ML frameworks | Free/Paid |
| **Arize AI** | Excellent | Excellent | Excellent | Excellent | Multiple | Paid |
| **Fiddler** | Very Good | Excellent | Excellent | Very Good | Multiple | Paid |
| **DataRobot MLOps** | Excellent | Excellent | Excellent | Excellent | Multiple | Paid |
| **AWS SageMaker Model Monitor** | Very Good | Very Good | Good | Good | AWS ecosystem | Paid (AWS) |

---

## Practical Guides

### Setting Up MLflow

**Step 1: Install MLflow**

```bash
# Install MLflow
pip install mlflow

# Install database backend (optional but recommended)
pip install psycopg2-binary  # For PostgreSQL
pip install pymysql          # For MySQL

# Install cloud storage support
pip install boto3            # For S3
pip install google-cloud-storage  # For GCS
pip install azure-storage-blob    # For Azure Blob
```

**Step 2: Configure MLflow Server**

```bash
# Create database for tracking
createdb mlflow_db

# Start MLflow server with PostgreSQL backend and S3 artifact storage
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow_db \
    --default-artifact-root s3://my-mlflow-bucket/artifacts \
    --host 0.0.0.0 \
    --port 5000
```

**Step 3: Docker Compose Setup**

```yaml
# docker-compose.mlflow.yml
version: '3.8'

services:
  postgres:
    image: postgres:14
    container_name: mlflow_postgres
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    container_name: mlflow_server
    depends_on:
      - postgres
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow_db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://my-mlflow-bucket/artifacts
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow_db
      --default-artifact-root s3://my-mlflow-bucket/artifacts
      --host 0.0.0.0
      --port 5000

volumes:
  postgres_data:
```

```bash
# Start MLflow stack
docker-compose -f docker-compose.mlflow.yml up -d
```

**Step 4: Configure Client**

```python
# mlflow_config.py
import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my-ml-project")

# Configure artifact location
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'
os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key'
```

### Containerizing ML Models

**Step 1: Create Project Structure**

```bash
ml-model-deploy/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config/
│   ├── model_config.yaml
│   └── logging_config.yaml
├── models/
│   └── model.pkl
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── utils.py
├── tests/
│   ├── test_inference.py
│   └── test_api.py
└── scripts/
    ├── train.py
    └── export_model.py
```

**Step 2: Multi-stage Dockerfile**

```dockerfile
# Dockerfile.optimized
# Stage 1: Build stage
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Step 3: Build and Push**

```bash
# Build image
docker build -t ml-model:v1.0 -f Dockerfile.optimized .

# Tag for registry
docker tag ml-model:v1.0 myregistry.com/ml-model:v1.0

# Push to registry
docker push myregistry.com/ml-model:v1.0

# Run locally
docker run -p 8000:8000 ml-model:v1.0
```

### Blue-Green Deployment

**Step 1: Kubernetes Deployment Configuration**

```yaml
# blue-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-blue
  labels:
    app: ml-model
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
      version: blue
  template:
    metadata:
      labels:
        app: ml-model
        version: blue
    spec:
      containers:
      - name: ml-model
        image: myregistry.com/ml-model:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
# green-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-green
  labels:
    app: ml-model
    version: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
      version: green
  template:
    metadata:
      labels:
        app: ml-model
        version: green
    spec:
      containers:
      - name: ml-model
        image: myregistry.com/ml-model:v2.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Step 2: Service Configuration**

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
    version: blue  # Switch between blue/green
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

**Step 3: Deployment Script**

```bash
#!/bin/bash
# deploy.sh - Blue-Green deployment script

set -e

CURRENT_VERSION=$(kubectl get service ml-model-service -o jsonpath='{.spec.selector.version}')
echo "Current version: $CURRENT_VERSION"

# Determine new version
if [ "$CURRENT_VERSION" == "blue" ]; then
    NEW_VERSION="green"
    NEW_IMAGE="myregistry.com/ml-model:v2.0"
else
    NEW_VERSION="blue"
    NEW_IMAGE="myregistry.com/ml-model:v2.0"
fi

echo "Deploying to $NEW_VERSION environment..."

# Update deployment with new image
kubectl set image deployment/ml-model-$NEW_VERSION \
    ml-model=$NEW_IMAGE

# Wait for rollout
kubectl rollout status deployment/ml-model-$NEW_VERSION

# Run smoke tests
echo "Running smoke tests..."
NEW_POD=$(kubectl get pod -l version=$NEW_VERSION -o jsonpath='{.items[0].metadata.name}')
kubectl exec $NEW_POD -- curl -f http://localhost:8000/health

# Switch traffic
echo "Switching traffic to $NEW_VERSION..."
kubectl patch service ml-model-service \
    -p "{\"spec\":{\"selector\":{\"version\":\"$NEW_VERSION\"}}}"

echo "Deployment complete. Traffic now routing to $NEW_VERSION"
echo "Monitor the deployment. To rollback, run:"
echo "kubectl patch service ml-model-service -p '{\"spec\":{\"selector\":{\"version\":\"$CURRENT_VERSION\"}}}'"
```

### Setting Up Model Monitoring

**Step 1: Install Dependencies**

```bash
pip install prometheus-client evidently
```

**Step 2: Create Monitoring Class**

```python
# monitoring.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd
import numpy as np
from collections import deque
import threading
import time

class MLModelMonitoring:
    """Comprehensive ML model monitoring."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        model_name: str = "ml_model",
        port: int = 8001
    ):
        self.reference_data = reference_data
        self.model_name = model_name
        self.port = port

        # Initialize metrics
        self._init_metrics()

        # Data collection
        self.prediction_window = deque(maxlen=1000)
        self.feature_window = deque(maxlen=1000)

        # Start metrics server
        start_http_server(self.port)

        # Start background monitoring
        self._start_background_monitoring()

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.prediction_counter = Counter(
            f'{self.model_name}_predictions_total',
            'Total predictions',
            ['status']
        )

        self.prediction_latency = Histogram(
            f'{self.model_name}_latency_seconds',
            'Prediction latency'
        )

        self.confidence_gauge = Gauge(
            f'{self.model_name}_confidence',
            'Current prediction confidence'
        )

        self.drift_score = Gauge(
            f'{self.model_name}_drift_score',
            'Data drift score',
            ['feature']
        )

    def record_prediction(
        self,
        features: dict,
        prediction: float,
        confidence: float,
        latency: float
    ):
        """Record a prediction for monitoring."""
        # Update metrics
        self.prediction_counter.labels(status='success').inc()
        self.prediction_latency.observe(latency)
        self.confidence_gauge.set(confidence)

        # Store for drift detection
        self.feature_window.append(features)
        self.prediction_window.append(prediction)

    def _start_background_monitoring(self):
        """Start background drift detection."""
        def monitor():
            while True:
                time.sleep(300)  # Check every 5 minutes
                if len(self.feature_window) >= 100:
                    self._check_drift()

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def _check_drift(self):
        """Check for data drift."""
        current_data = pd.DataFrame(list(self.feature_window))

        # Generate drift report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        # Extract drift scores
        drift_results = report.as_dict()

        # Update Prometheus metrics
        for metric in drift_results.get('metrics', []):
            if 'drift_score' in metric.get('result', {}):
                feature = metric.get('column_name', 'unknown')
                score = metric['result']['drift_score']
                self.drift_score.labels(feature=feature).set(score)
```

**Step 3: Integration with FastAPI**

```python
# Add to main.py
from monitoring import MLModelMonitoring

# Initialize monitoring
reference_data = pd.read_csv('reference_data.csv')
monitoring = MLModelMonitoring(reference_data)

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        # Make prediction
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0].max()

        latency = time.time() - start_time

        # Record for monitoring
        monitoring.record_prediction(
            features=dict(enumerate(request.features)),
            prediction=float(prediction),
            confidence=float(confidence),
            latency=latency
        )

        return {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'latency': latency
        }
    except Exception as e:
        monitoring.prediction_counter.labels(status='error').inc()
        raise
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Model CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.9'
  DOCKER_REGISTRY: myregistry.com
  MODEL_NAME: churn-predictor

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run linting
      run: |
        flake8 src/ --max-line-length=100
        black --check src/
        isort --check-only src/

    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  train:
    name: Train and Evaluate Model
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2

    - name: Download training data
      run: |
        aws s3 cp s3://my-bucket/data/train.csv data/train.csv
        aws s3 cp s3://my-bucket/data/test.csv data/test.csv

    - name: Train model
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python scripts/train.py \
          --data-path data/train.csv \
          --experiment-name ${{ env.MODEL_NAME }} \
          --run-name "ci-run-${{ github.run_number }}"

    - name: Evaluate model
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python scripts/evaluate.py \
          --test-data data/test.csv \
          --model-uri runs:/${{ env.MODEL_RUN_ID }}/model

    - name: Register model
      if: success()
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        python scripts/register_model.py \
          --model-name ${{ env.MODEL_NAME }} \
          --run-id ${{ env.MODEL_RUN_ID }}

  build:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    needs: train
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.DOCKER_REGISTRY }}
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.DOCKER_REGISTRY }}/${{ env.MODEL_NAME }}
        tags: |
          type=ref,event=branch
          type=sha,prefix={{branch}}-
          type=semver,pattern={{version}}

    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    environment: staging

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}

    - name: Deploy to staging
      run: |
        kubectl set image deployment/${{ env.MODEL_NAME }}-staging \
          ${{ env.MODEL_NAME }}=${{ env.DOCKER_REGISTRY }}/${{ env.MODEL_NAME }}:${{ github.sha }}

        kubectl rollout status deployment/${{ env.MODEL_NAME }}-staging

    - name: Run smoke tests
      run: |
        export STAGING_URL=$(kubectl get service ${{ env.MODEL_NAME }}-staging -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        python tests/smoke_tests.py --url $STAGING_URL

    - name: Run integration tests
      run: |
        export STAGING_URL=$(kubectl get service ${{ env.MODEL_NAME }}-staging -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        pytest tests/integration/ --url $STAGING_URL

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}

    - name: Blue-Green deployment
      run: |
        # Deploy to green environment
        kubectl set image deployment/${{ env.MODEL_NAME }}-green \
          ${{ env.MODEL_NAME }}=${{ env.DOCKER_REGISTRY }}/${{ env.MODEL_NAME }}:${{ github.sha }}

        kubectl rollout status deployment/${{ env.MODEL_NAME }}-green

    - name: Validation
      run: |
        export GREEN_URL=$(kubectl get service ${{ env.MODEL_NAME }}-green -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        python tests/validation.py --url $GREEN_URL

    - name: Switch traffic
      run: |
        kubectl patch service ${{ env.MODEL_NAME }} \
          -p '{"spec":{"selector":{"version":"green"}}}'

    - name: Monitor deployment
      run: |
        sleep 60
        python scripts/check_metrics.py --duration 300

  notify:
    name: Notify Team
    runs-on: ubuntu-latest
    needs: [deploy-production]
    if: always()

    steps:
    - name: Send Slack notification
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: |
          Model deployment ${{ job.status }}
          Model: ${{ env.MODEL_NAME }}
          Version: ${{ github.sha }}
          Deployer: ${{ github.actor }}
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Best Practices

### Model Versioning Best Practices

**Semantic Versioning for ML Models:**

```python
# model_versioning.py
from dataclasses import dataclass
from typing import Optional
import hashlib
import json

@dataclass
class ModelVersion:
    """Semantic versioning for ML models."""
    major: int  # Breaking changes (architecture change)
    minor: int  # New features (new features added)
    patch: int  # Bug fixes (bug fixes, minor improvements)
    metadata: dict

    def __str__(self):
        return f"v{self.major}.{self.minor}.{self.patch}"

    def increment_major(self):
        """Increment major version (breaking change)."""
        self.major += 1
        self.minor = 0
        self.patch = 0

    def increment_minor(self):
        """Increment minor version (new feature)."""
        self.minor += 1
        self.patch = 0

    def increment_patch(self):
        """Increment patch version (bug fix)."""
        self.patch += 1

class ModelVersionManager:
    """Manage model versions with metadata."""

    def __init__(self):
        self.versions = {}

    def create_version(
        self,
        model_name: str,
        model_object: Any,
        change_type: str = 'patch',
        metadata: dict = None
    ) -> str:
        """Create new model version."""
        # Get current version or start at 1.0.0
        current = self.versions.get(model_name, ModelVersion(0, 0, 0, {}))

        # Increment version based on change type
        if change_type == 'major':
            current.increment_major()
        elif change_type == 'minor':
            current.increment_minor()
        else:
            current.increment_patch()

        # Calculate model hash
        model_hash = self._calculate_hash(model_object)

        # Add metadata
        current.metadata = {
            'model_hash': model_hash,
            'created_at': datetime.now().isoformat(),
            'change_type': change_type,
            **(metadata or {})
        }

        self.versions[model_name] = current
        return str(current)

    def _calculate_hash(self, model_object: Any) -> str:
        """Calculate hash of model for change detection."""
        model_bytes = pickle.dumps(model_object)
        return hashlib.sha256(model_bytes).hexdigest()
```

**Version Control Best Practices:**

1. **Always tag models with:**
   - Training data version
   - Code version (git commit)
   - Hyperparameters
   - Dependencies versions
   - Performance metrics

2. **Use immutable model artifacts**
3. **Maintain model lineage**
4. **Automate version bumping**
5. **Document breaking changes**

### Feature Management

```python
# feature_management.py
from typing import List, Dict, Any
import yaml

class FeatureConfig:
    """Feature configuration management."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def get_training_features(self) -> List[str]:
        """Get features for training."""
        return [
            f['name'] for f in self.config['features']
            if f.get('use_for_training', True)
        ]

    def get_serving_features(self) -> List[str]:
        """Get features for serving."""
        return [
            f['name'] for f in self.config['features']
            if f.get('use_for_serving', True)
        ]

    def validate_features(self, data: Dict[str, Any]) -> bool:
        """Validate feature schema."""
        required = self.get_serving_features()
        missing = set(required) - set(data.keys())

        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Type validation
        for feature in self.config['features']:
            name = feature['name']
            if name in data:
                expected_type = feature.get('type')
                if expected_type and not isinstance(data[name], eval(expected_type)):
                    raise TypeError(
                        f"Feature {name} has wrong type. "
                        f"Expected {expected_type}, got {type(data[name])}"
                    )

        return True
```

**features.yaml:**

```yaml
features:
  - name: age
    type: int
    description: Customer age
    use_for_training: true
    use_for_serving: true
    validation:
      min: 18
      max: 100

  - name: income
    type: float
    description: Annual income
    use_for_training: true
    use_for_serving: true
    transformation: log_transform
    validation:
      min: 0

  - name: account_age_days
    type: int
    description: Days since account creation
    use_for_training: true
    use_for_serving: true
    feature_engineering:
      - bin_into_categories

  - name: transaction_count_30d
    type: int
    description: Number of transactions in last 30 days
    use_for_training: true
    use_for_serving: true
    aggregation:
      window: 30d
      function: count
```

### Testing ML Systems

```python
# tests/test_model.py
import pytest
import numpy as np
import pandas as pd
from src.model import Model

class TestModel:
    """Comprehensive model tests."""

    @pytest.fixture
    def model(self):
        """Load trained model."""
        return Model.load('models/model.pkl')

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            'age': [25, 35, 45],
            'income': [50000, 75000, 100000],
            'account_age_days': [365, 730, 1095]
        })

    def test_model_prediction_shape(self, model, sample_data):
        """Test prediction output shape."""
        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)

    def test_model_prediction_range(self, model, sample_data):
        """Test prediction values are in valid range."""
        predictions = model.predict(sample_data)
        assert all(0 <= p <= 1 for p in predictions)

    def test_model_deterministic(self, model, sample_data):
        """Test model produces consistent predictions."""
        pred1 = model.predict(sample_data)
        pred2 = model.predict(sample_data)
        np.testing.assert_array_equal(pred1, pred2)

    def test_model_handles_missing_values(self, model):
        """Test model handles missing values gracefully."""
        data_with_nan = pd.DataFrame({
            'age': [25, np.nan, 45],
            'income': [50000, 75000, np.nan],
            'account_age_days': [365, 730, 1095]
        })
        # Should not raise exception
        predictions = model.predict(data_with_nan)
        assert len(predictions) == len(data_with_nan)

    def test_model_feature_importance(self, model):
        """Test feature importance is available."""
        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert all(0 <= v <= 1 for v in importance.values())

    def test_prediction_latency(self, model, sample_data):
        """Test prediction latency is acceptable."""
        import time
        start = time.time()
        model.predict(sample_data)
        latency = time.time() - start
        assert latency < 0.1  # 100ms threshold

# tests/test_api.py
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

class TestAPI:
    """API endpoint tests."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint(self):
        """Test prediction endpoint."""
        payload = {
            "features": [25, 50000, 365]
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "confidence" in response.json()

    def test_predict_invalid_input(self):
        """Test prediction with invalid input."""
        payload = {
            "features": []  # Empty features
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_batch_predict(self):
        """Test batch prediction endpoint."""
        payload = {
            "instances": [
                [25, 50000, 365],
                [35, 75000, 730],
                [45, 100000, 1095]
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 3

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "model_predictions_total" in response.text
```

### Security Best Practices

```python
# security.py
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import secrets

security = HTTPBearer()

class SecurityManager:
    """Security management for ML APIs."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"

    def create_access_token(
        self,
        data: dict,
        expires_delta: timedelta = timedelta(hours=24)
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})

        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

    def rate_limit_key(self, user_id: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{user_id}:{datetime.now().strftime('%Y%m%d%H')}"

# Middleware for authentication
@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    """Authenticate incoming requests."""
    # Skip auth for health checks
    if request.url.path in ["/health", "/ready"]:
        return await call_next(request)

    # Extract token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header"
        )

    token = auth_header.split(" ")[1]

    # Verify token
    try:
        payload = security_manager.verify_token(token)
        request.state.user = payload
    except HTTPException:
        raise

    return await call_next(request)

# Input validation and sanitization
from pydantic import validator, Field

class SecurePredictionRequest(BaseModel):
    """Secure prediction request with validation."""
    features: List[float] = Field(..., min_items=1, max_items=100)

    @validator('features')
    def validate_features(cls, v):
        # Check for NaN and Inf
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features contain invalid values")

        # Check ranges
        if any(abs(x) > 1e6 for x in v):
            raise ValueError("Feature values out of acceptable range")

        return v
```

---

## Troubleshooting

### Common Deployment Issues

**Issue 1: Model Loading Failures**

```python
# Solution: Implement robust model loading with fallback
import joblib
import logging
from pathlib import Path

class ModelLoader:
    """Robust model loader with fallback."""

    def __init__(self, model_paths: List[str]):
        self.model_paths = model_paths
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Try loading model from multiple paths."""
        for path in self.model_paths:
            try:
                if Path(path).exists():
                    model = joblib.load(path)
                    self.logger.info(f"Successfully loaded model from {path}")
                    return model
            except Exception as e:
                self.logger.warning(f"Failed to load from {path}: {e}")
                continue

        raise RuntimeError("Failed to load model from all paths")

# Usage
loader = ModelLoader([
    "/app/models/model.pkl",
    "/backup/models/model.pkl",
    "s3://bucket/models/model.pkl"
])
model = loader.load_model()
```

**Issue 2: Memory Leaks**

```python
# Solution: Implement proper cleanup and memory monitoring
import gc
import psutil
import os

class MemoryMonitor:
    """Monitor and manage memory usage."""

    def __init__(self, threshold_percent: float = 80.0):
        self.threshold = threshold_percent
        self.process = psutil.Process(os.getpid())

    def check_memory(self):
        """Check current memory usage."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        if memory_percent > self.threshold:
            self._cleanup()

        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'percent': memory_percent
        }

    def _cleanup(self):
        """Force garbage collection."""
        gc.collect()
        logging.warning("Memory cleanup triggered")

# Middleware for memory monitoring
@app.middleware("http")
async def monitor_memory(request: Request, call_next):
    memory_monitor = MemoryMonitor()
    memory_monitor.check_memory()
    response = await call_next(request)
    return response
```

**Issue 3: Prediction Timeouts**

```python
# Solution: Implement timeout handling and async processing
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def predict_with_timeout(
    model,
    features: np.ndarray,
    timeout: float = 1.0
) -> dict:
    """Make prediction with timeout."""
    loop = asyncio.get_event_loop()

    try:
        prediction = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                model.predict,
                features
            ),
            timeout=timeout
        )
        return {'prediction': float(prediction[0]), 'status': 'success'}
    except asyncio.TimeoutError:
        return {'error': 'Prediction timeout', 'status': 'timeout'}
    except Exception as e:
        return {'error': str(e), 'status': 'error'}
```

**Issue 4: Version Mismatch**

```python
# Solution: Strict version checking
import pkg_resources

def check_dependencies():
    """Check if dependencies match requirements."""
    required = {
        'scikit-learn': '1.3.2',
        'pandas': '2.1.3',
        'numpy': '1.26.2'
    }

    issues = []
    for package, required_version in required.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if installed_version != required_version:
                issues.append(
                    f"{package}: required {required_version}, "
                    f"found {installed_version}"
                )
        except pkg_resources.DistributionNotFound:
            issues.append(f"{package}: not installed")

    if issues:
        raise RuntimeError(f"Dependency issues:\n" + "\n".join(issues))

# Run on startup
@app.on_event("startup")
async def startup_checks():
    check_dependencies()
```

---

## Cost Optimization

### Strategies for Cost Reduction

**1. Model Optimization**

```python
# Model compression techniques
from sklearn.ensemble import RandomForestClassifier
import pickle
import gzip

def compress_model(model, output_path: str):
    """Compress model for storage and faster loading."""
    # Serialize model
    model_bytes = pickle.dumps(model)

    # Compress
    with gzip.open(output_path, 'wb') as f:
        f.write(model_bytes)

    # Compare sizes
    original_size = len(model_bytes) / 1024 / 1024
    compressed_size = os.path.getsize(output_path) / 1024 / 1024

    print(f"Original: {original_size:.2f} MB")
    print(f"Compressed: {compressed_size:.2f} MB")
    print(f"Compression ratio: {original_size/compressed_size:.2f}x")

# Model quantization for deep learning
import tensorflow as tf

def quantize_model(model_path: str, output_path: str):
    """Quantize TensorFlow model to reduce size."""
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Post-training quantization
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
```

**2. Infrastructure Optimization**

```yaml
# kubernetes/model-deployment.yaml with resource optimization
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: ml-model
        image: ml-model:v1.0
        resources:
          requests:
            memory: "256Mi"  # Right-sized
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

        # Horizontal Pod Autoscaler
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

**3. Batch Processing**

```python
# Optimize with batching
class BatchPredictor:
    """Efficient batch prediction with optimal batch sizes."""

    def __init__(self, model, optimal_batch_size: int = 32):
        self.model = model
        self.batch_size = optimal_batch_size

    def predict_batch(self, features: List[np.ndarray]) -> List[float]:
        """Predict in optimized batches."""
        predictions = []

        for i in range(0, len(features), self.batch_size):
            batch = features[i:i + self.batch_size]
            batch_array = np.vstack(batch)
            batch_predictions = self.model.predict(batch_array)
            predictions.extend(batch_predictions)

        return predictions
```

**4. Caching Strategy**

```python
from functools import lru_cache
import redis
import pickle

class PredictionCache:
    """Cache predictions to reduce compute costs."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour

    def get_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key from features."""
        return hashlib.sha256(features.tobytes()).hexdigest()

    def get_cached_prediction(self, features: np.ndarray) -> Optional[dict]:
        """Get cached prediction if available."""
        key = self.get_cache_key(features)
        cached = self.redis.get(key)

        if cached:
            return pickle.loads(cached)
        return None

    def cache_prediction(self, features: np.ndarray, prediction: dict):
        """Cache prediction result."""
        key = self.get_cache_key(features)
        self.redis.setex(
            key,
            self.ttl,
            pickle.dumps(prediction)
        )

# Usage in API
@app.post("/predict")
async def predict_with_cache(request: PredictionRequest):
    features = np.array(request.features)

    # Check cache
    cached = cache.get_cached_prediction(features)
    if cached:
        return cached

    # Make prediction
    prediction = model.predict(features.reshape(1, -1))[0]
    result = {'prediction': float(prediction)}

    # Cache result
    cache.cache_prediction(features, result)

    return result
```

**Cost Optimization Checklist:**

- [ ] Right-size container resources
- [ ] Implement autoscaling
- [ ] Use spot instances where possible
- [ ] Compress models
- [ ] Implement prediction caching
- [ ] Batch requests when possible
- [ ] Use model quantization
- [ ] Optimize storage costs (lifecycle policies)
- [ ] Monitor and eliminate unused resources
- [ ] Use regional resources strategically

---

## Hands-On Tutorial: End-to-End MLOps

### Step 1: Project Setup

```bash
# Create project structure
mkdir ml-production-system
cd ml-production-system

# Create directory structure
mkdir -p {data,models,src,tests,config,notebooks,scripts}

# Initialize git
git init

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
cat > requirements.txt << EOF
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
mlflow==2.8.1
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
prometheus-client==0.19.0
pytest==7.4.3
black==23.11.0
flake8==6.1.0
dvc==3.30.1
evidently==0.4.11
EOF

pip install -r requirements.txt

# Initialize DVC
dvc init
```

### Step 2: Training Pipeline

```python
# scripts/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import argparse
import yaml

def load_config(config_path: str) -> dict:
    """Load configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_data(data_path: str) -> tuple:
    """Load and split data."""
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, params: dict):
    """Train model with given parameters."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

def main(args):
    """Main training pipeline."""
    # Load config
    config = load_config(args.config)

    # Set up MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_path)

    with mlflow.start_run(run_name=args.run_name):
        # Log parameters
        params = config['model']['params']
        mlflow.log_params(params)

        # Train model
        model = train_model(X_train, y_train, params)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric('cv_mean', cv_scores.mean())
        mlflow.log_metric('cv_std', cv_scores.std())

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=config['model']['name']
        )

        print(f"Training complete. Metrics: {metrics}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--run-name', default='training-run')

    args = parser.parse_args()
    main(args)
```

### Step 3: Deploy and Monitor

```bash
# Build Docker image
docker build -t ml-model:v1.0 .

# Deploy locally
docker-compose up -d

# Run tests
pytest tests/ -v

# Monitor metrics
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana
```

### Step 4: Set Up CI/CD

```bash
# Create GitHub Actions workflow
mkdir -p .github/workflows

# Copy the workflow file from the CI/CD section above

# Commit and push
git add .
git commit -m "Initial MLOps setup"
git push origin main
```

---

## Summary

This comprehensive guide covered:

- **Lifecycle Management**: Development, deployment, monitoring, and governance
- **Architecture Patterns**: End-to-end pipelines with detailed diagrams
- **Platform Comparisons**: In-depth analysis of MLOps platforms and tools
- **Practical Implementation**: Production-ready code examples
- **Best Practices**: Versioning, testing, security, and cost optimization
- **Troubleshooting**: Common issues and solutions
- **Hands-on Tutorial**: Complete end-to-end implementation

**Key Takeaways:**

1. MLOps is essential for production ML systems
2. Automation is critical for scalability
3. Monitoring prevents model degradation
4. Governance ensures compliance and quality
5. Cost optimization requires strategic planning

**Next Steps:**

- [ML Fundamentals](../fundamentals/index.md)
- [ML Algorithms](../algorithms/index.md)
- [Deep Learning](../deep-learning/index.md)

---

**Additional Resources:**

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DVC Documentation](https://dvc.org/doc)
