# Design an AutoML Platform

An automated machine learning platform that democratizes ML by automatically handling feature engineering, algorithm selection, neural architecture search, hyperparameter optimization, and model ensembling to produce production-ready models with minimal human intervention.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1000+ concurrent experiments, 100 datasets/day, 50K models trained/week, multi-tenant platform |
| **Key Challenges** | Neural architecture search efficiency, automated feature engineering, Bayesian hyperparameter optimization, model ensembling, distributed training coordination |
| **Core Concepts** | NAS (ENAS/DARTS), automated feature engineering, Bayesian optimization, genetic algorithms, ensemble methods, transfer learning, meta-learning |
| **Companies** | Google AutoML, H2O.ai, DataRobot, Amazon SageMaker Autopilot, Microsoft Azure AutoML, auto-sklearn |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Dataset Ingestion** | Upload CSV/Parquet/JSON datasets with automatic type inference | P0 (Must have) |
    | **Automated Feature Engineering** | Generate features via transformations, encoding, interactions | P0 (Must have) |
    | **Algorithm Selection** | Search across classical ML (XGBoost, RF) and deep learning models | P0 (Must have) |
    | **Neural Architecture Search** | Discover optimal DL architectures via ENAS, DARTS, NASNet | P0 (Must have) |
    | **Hyperparameter Optimization** | Bayesian optimization, genetic algorithms, early stopping | P0 (Must have) |
    | **Model Ensembling** | Stack/blend multiple models for improved performance | P0 (Must have) |
    | **Model Evaluation** | Cross-validation, holdout testing, metric tracking | P0 (Must have) |
    | **Model Registry** | Version and store discovered models with lineage | P0 (Must have) |
    | **Resource Management** | GPU/CPU allocation, distributed training, spot instances | P1 (Should have) |
    | **Transfer Learning** | Leverage pre-trained models (BERT, ResNet) for warm starts | P1 (Should have) |
    | **Explainability** | SHAP values, feature importance, model interpretability | P1 (Should have) |
    | **Model Export** | Export to ONNX, TensorFlow Serving, PyTorch format | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Data cleaning and imputation (users handle preprocessing)
    - Model serving infrastructure (separate system)
    - Real-time inference endpoints
    - Custom architecture design beyond search space
    - Data labeling and annotation

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Experiment Completion Time** | < 24 hours for 95% of experiments | Users expect fast iteration |
    | **Concurrent Experiments** | Support 1,000+ parallel experiments | Multi-tenant platform scale |
    | **Model Quality** | Match/exceed manual tuning 80% of cases | Justify automation overhead |
    | **Search Efficiency** | Find near-optimal model in < 100 trials | Limited compute budget |
    | **GPU Utilization** | > 80% during NAS and training | Expensive resources |
    | **Availability** | 99.9% uptime | Production ML platform SLA |
    | **Cost Efficiency** | < $50 per experiment (average) | Competitive with manual tuning |
    | **Scalability** | Support 100TB datasets via sampling/distributed | Enterprise use cases |
    | **Reproducibility** | Deterministic results with same seed | Scientific rigor |
    | **Extensibility** | Pluggable search spaces and algorithms | Adapt to new techniques |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Experiments:
    - Concurrent experiments: 1,000 experiments
    - Experiment duration: 6-48 hours average
    - New experiments per day: 500 experiments
    - Peak submissions: 200 experiments/hour

    Dataset Ingestion:
    - Datasets uploaded per day: 100 datasets
    - Average dataset size: 5 GB (10K-1M rows)
    - Large datasets (>100GB): 10% of uploads
    - Dataset preprocessing time: 10-30 minutes

    AutoML Search:
    - Feature engineering trials per experiment: 20-50 combinations
    - Model architecture trials: 100 trials (NAS/HPO combined)
    - Models trained per experiment: 100-500 models
    - Total models trained per day: 500 experiments × 200 models = 100K models/day
    - Models trained per week: 700K models

    Resource Allocation:
    - GPU experiments: 30% of experiments = 300 experiments
    - CPU experiments: 70% = 700 experiments
    - Average GPUs per NAS experiment: 4 GPUs
    - Average duration per trial: 15 minutes (with early stopping)

    Feature Engineering:
    - Features generated per dataset: 50-500 derived features
    - Feature selection: top 100 features after importance ranking
    - Feature computation time: 5-20 minutes

    Model Ensembling:
    - Top models per experiment: 5-10 models
    - Ensemble methods: stacking, blending, voting
    - Ensemble training time: 30 minutes
    ```

    ### Storage Estimates

    ```
    Datasets:
    - Active datasets: 500 datasets × 5 GB = 2.5 TB
    - Processed feature stores: 500 datasets × 10 GB = 5 TB
    - 90-day retention: 2.5 TB × 3 = 7.5 TB raw + 15 TB processed
    - With compression (60%): 13.5 TB

    Model Artifacts:
    - Models per experiment: 200 models
    - Average model size: 50 MB (classical ML) to 500 MB (DL)
    - Keep top 10 models per experiment: 500 experiments × 10 × 100 MB = 500 GB/day
    - 30-day retention: 500 GB × 30 = 15 TB
    - With deduplication (shared layers): 10 TB

    Experiment Metadata:
    - Experiment configs, hyperparameters: 10 KB per trial
    - 100K models/day × 10 KB = 1 GB/day
    - 90 days: 90 GB

    Metrics and Logs:
    - Training metrics per trial: 1 MB (loss curves, metrics)
    - 100K models/day × 1 MB = 100 GB/day
    - 90 days: 9 TB (compressed: 2 TB)

    Feature Store:
    - Engineered features per dataset: 10 GB average
    - Active feature stores: 500 datasets × 10 GB = 5 TB
    - Historical features (90 days): 15 TB

    Total: 13.5 TB (datasets) + 10 TB (models) + 2 TB (metrics) + 15 TB (features) + 90 GB (metadata) ≈ 40 TB
    ```

    ### Compute Estimates

    ```
    GPU Compute (NAS and Deep Learning):
    - GPU experiments: 300 concurrent
    - GPUs per experiment: 4 GPUs average
    - Total GPUs: 300 × 4 = 1,200 GPUs
    - GPU types: V100, A100, T4
    - Cost: 1,200 GPUs × $1.50/hour = $1,800/hour = $43,200/day

    CPU Compute (Classical ML and Feature Engineering):
    - CPU experiments: 700 concurrent
    - vCPUs per experiment: 16 vCPUs
    - Total vCPUs: 700 × 16 = 11,200 vCPUs
    - Cost: 11,200 × $0.04/hour = $448/hour = $10,752/day

    HPO Controller:
    - Bayesian optimization service: 20 instances × 8 vCPUs = 160 vCPUs
    - Cost: $128/day

    Feature Engineering Service:
    - Parallel feature generation: 100 instances × 8 vCPUs = 800 vCPUs
    - Cost: $640/day

    Control Plane:
    - API servers, schedulers, metadata DB: 200 vCPUs
    - Cost: $160/day

    Total Daily Compute Cost: $43,200 + $10,752 + $128 + $640 + $160 ≈ $55,000/day
    Cost per experiment: $55,000 / 500 = $110/experiment (includes failed trials)
    ```

    ### Network Estimates

    ```
    Dataset uploads:
    - 100 datasets/day × 5 GB = 500 GB/day
    - Average: 500 GB / 86,400 = 5.8 MB/sec ≈ 46 Mbps
    - Peak (10x): 460 Mbps

    Model downloads (for evaluation):
    - 100K models/day × 100 MB = 10 TB/day
    - Average: 10 TB / 86,400 = 116 MB/sec ≈ 930 Mbps

    Metrics and logs:
    - 100 GB/day = 1.16 MB/sec ≈ 9 Mbps

    Feature store I/O:
    - Feature reads: 500 experiments × 10 GB = 5 TB/day
    - 5 TB / 86,400 = 58 MB/sec ≈ 464 Mbps

    Total bandwidth: 46 + 930 + 9 + 464 ≈ 1.5 Gbps average
    ```

---

=== "🏗️ Step 2: High-Level Design"

    ## Architecture Diagram

    ```mermaid
    graph TB
        User["👤 User/Data Scientist"]
        WebUI["🖥️ Web UI/CLI"]
        APIGateway["🚪 API Gateway"]

        subgraph "Control Plane"
            ExperimentController["🎛️ Experiment Controller"]
            SearchOrchestrator["🔍 Search Orchestrator"]
            ResourceScheduler["⚙️ Resource Scheduler"]
            MetaLearner["🧠 Meta-Learning Service<br/>(Transfer Learning)"]
        end

        subgraph "Data Processing"
            DataIngestion["📥 Data Ingestion Service"]
            FeatureEngineering["🔧 Feature Engineering<br/>(Transformations, Encoding)"]
            FeatureStore["💾 Feature Store"]
            DataValidator["✓ Data Validator<br/>(Schema, Quality)"]
        end

        subgraph "AutoML Engine"
            NASController["🏗️ NAS Controller<br/>(ENAS/DARTS)"]
            HPOService["🎯 HPO Service<br/>(Bayesian/Genetic)"]
            ModelSelector["🤖 Model Selector<br/>(Classical ML + DL)"]
            EnsembleBuilder["🎭 Ensemble Builder<br/>(Stacking/Blending)"]
        end

        subgraph "Training Cluster"
            K8s["☸️ Kubernetes"]
            GPUPool["🎮 GPU Pool<br/>(V100/A100)"]
            CPUPool["💻 CPU Pool"]
            DistTrainer["🚀 Distributed Trainer<br/>(Ray/Dask)"]
        end

        subgraph "Model Management"
            ModelRegistry["📚 Model Registry<br/>(MLflow)"]
            ModelEvaluator["📊 Model Evaluator<br/>(Cross-validation)"]
            Explainer["🔍 Explainer<br/>(SHAP/LIME)"]
        end

        subgraph "Storage"
            ObjectStore["☁️ Object Storage<br/>(S3/GCS)"]
            MetadataDB["💾 Metadata DB<br/>(PostgreSQL)"]
            MetricsDB["📈 Metrics DB<br/>(Prometheus)"]
            CacheLayer["⚡ Cache<br/>(Redis)"]
        end

        User -->|Upload Dataset| WebUI
        WebUI -->|REST/gRPC| APIGateway
        APIGateway --> ExperimentController

        ExperimentController --> DataIngestion
        DataIngestion --> DataValidator
        DataValidator --> FeatureEngineering
        FeatureEngineering --> FeatureStore
        FeatureStore --> ObjectStore

        ExperimentController --> SearchOrchestrator
        SearchOrchestrator --> NASController
        SearchOrchestrator --> HPOService
        SearchOrchestrator --> ModelSelector

        NASController --> ResourceScheduler
        HPOService --> ResourceScheduler
        ModelSelector --> ResourceScheduler
        ResourceScheduler --> K8s

        K8s --> GPUPool
        K8s --> CPUPool
        GPUPool --> DistTrainer
        CPUPool --> DistTrainer

        DistTrainer -->|Read Features| FeatureStore
        DistTrainer -->|Save Models| ModelRegistry
        DistTrainer -->|Metrics| MetricsDB

        ModelRegistry --> ModelEvaluator
        ModelEvaluator --> EnsembleBuilder
        EnsembleBuilder --> ModelRegistry

        ModelRegistry --> Explainer
        Explainer --> MetadataDB

        MetaLearner -->|Suggest Architectures| NASController
        MetaLearner -->|Suggest Hyperparams| HPOService

        CacheLayer -.->|Cache Features| FeatureStore
        CacheLayer -.->|Cache Trial Results| HPOService

        ModelRegistry --> ObjectStore
        ExperimentController --> MetadataDB
    ```

    ---

    ## API Design

    ### Experiment Submission API

    ```protobuf
    message AutoMLExperimentRequest {
        string experiment_name = 1;
        string user_id = 2;
        DatasetConfig dataset = 3;
        TaskConfig task = 4;
        SearchConfig search_config = 5;
        ResourceConfig resources = 6;
        OptimizationConfig optimization = 7;
    }

    message DatasetConfig {
        string data_path = 1;              // s3://bucket/data.csv
        string format = 2;                 // csv, parquet, json
        string target_column = 3;          // column to predict
        repeated string feature_columns = 4;  // optional: auto-detect if empty
        float train_split = 5;             // default: 0.8
        int32 validation_folds = 6;        // default: 5 for CV
    }

    message TaskConfig {
        enum TaskType {
            CLASSIFICATION = 0;
            REGRESSION = 1;
            TIME_SERIES = 2;
            MULTICLASS = 3;
        }
        TaskType task_type = 1;
        string metric = 2;                 // accuracy, f1, rmse, mae
        bool maximize = 3;                 // true for accuracy, false for loss
        int32 num_classes = 4;             // for classification
    }

    message SearchConfig {
        int32 max_trials = 1;              // default: 100
        int32 max_parallel_trials = 2;     // default: 10
        int32 max_time_minutes = 3;        // default: 1440 (24 hours)
        bool enable_nas = 4;               // Neural architecture search
        bool enable_feature_engineering = 5;
        bool enable_ensemble = 6;
        SearchSpace search_space = 7;
    }

    message SearchSpace {
        repeated string algorithms = 1;    // xgboost, random_forest, dnn, cnn
        map<string, ParameterRange> hyperparameters = 2;
        NASConfig nas_config = 3;
    }

    message NASConfig {
        string method = 1;                 // enas, darts, nasnet
        int32 max_layers = 2;              // default: 20
        repeated string operations = 3;    // conv3x3, conv5x5, maxpool, skip
        int32 search_epochs = 4;           // default: 50
    }

    message AutoMLExperimentResponse {
        string experiment_id = 1;
        string status = 2;                 // QUEUED, RUNNING, COMPLETED, FAILED
        string dashboard_url = 3;
        EstimatedCost estimated_cost = 4;
    }

    message EstimatedCost {
        float compute_usd = 1;
        float storage_usd = 2;
        int32 estimated_hours = 3;
    }
    ```

    ### Trial Progress API

    ```protobuf
    message TrialUpdate {
        string experiment_id = 1;
        string trial_id = 2;
        int32 trial_number = 3;
        string model_type = 4;             // xgboost, resnet, ensemble
        map<string, float> hyperparameters = 5;
        float current_metric = 6;          // validation metric
        float best_metric = 7;
        int32 epoch = 8;
        bool early_stopped = 9;
    }

    message ExperimentStatus {
        string experiment_id = 1;
        int32 total_trials = 2;
        int32 completed_trials = 3;
        int32 failed_trials = 4;
        float best_metric = 5;
        string best_model_id = 6;
        int32 elapsed_minutes = 7;
        int32 estimated_remaining_minutes = 8;
    }
    ```

    ---

    ## Database Schema

    ### Experiment Metadata

    ```sql
    CREATE TABLE experiments (
        experiment_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        experiment_name VARCHAR(255),
        task_type VARCHAR(50),              -- classification, regression
        metric_name VARCHAR(50),            -- accuracy, rmse
        maximize BOOLEAN,

        -- Dataset info
        dataset_path VARCHAR(500),
        target_column VARCHAR(255),
        num_features INT,
        num_samples INT,
        train_test_split JSONB,

        -- Search configuration
        max_trials INT,
        max_parallel_trials INT,
        max_time_minutes INT,
        enable_nas BOOLEAN,
        enable_feature_engineering BOOLEAN,
        enable_ensemble BOOLEAN,

        -- Status
        status VARCHAR(50),                 -- QUEUED, RUNNING, COMPLETED, FAILED
        created_at TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,

        -- Results
        best_trial_id VARCHAR(255),
        best_metric_value FLOAT,
        total_trials_run INT,
        total_cost_usd DECIMAL(10, 2),

        INDEX idx_user_created (user_id, created_at),
        INDEX idx_status (status)
    );

    CREATE TABLE trials (
        trial_id VARCHAR(255) PRIMARY KEY,
        experiment_id VARCHAR(255) NOT NULL,
        trial_number INT,

        -- Model configuration
        model_type VARCHAR(100),            -- xgboost, resnet50, ensemble
        architecture JSONB,                 -- NAS discovered architecture
        hyperparameters JSONB,              -- learning_rate, batch_size, etc.

        -- Feature engineering
        feature_transformations JSONB,      -- transformations applied
        num_features_used INT,
        feature_importance JSONB,

        -- Training
        training_time_seconds INT,
        epochs_trained INT,
        early_stopped BOOLEAN,
        early_stop_epoch INT,

        -- Evaluation
        train_metric FLOAT,
        val_metric FLOAT,
        test_metric FLOAT,
        cv_metrics JSONB,                   -- cross-validation scores

        -- Resources
        gpus_used INT,
        cpu_cores INT,
        memory_gb INT,
        compute_cost_usd DECIMAL(10, 2),

        -- Model artifact
        model_path VARCHAR(500),            -- s3://bucket/models/...
        model_size_bytes BIGINT,

        status VARCHAR(50),                 -- RUNNING, COMPLETED, FAILED
        created_at TIMESTAMP,
        completed_at TIMESTAMP,

        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
        INDEX idx_experiment_metric (experiment_id, val_metric DESC),
        INDEX idx_model_type (model_type)
    );

    CREATE TABLE feature_stores (
        feature_store_id VARCHAR(255) PRIMARY KEY,
        experiment_id VARCHAR(255) NOT NULL,
        dataset_path VARCHAR(500),

        -- Feature metadata
        original_features JSONB,            -- list of original columns
        engineered_features JSONB,          -- generated features
        feature_statistics JSONB,           -- mean, std, min, max
        feature_correlations JSONB,

        -- Storage
        feature_path VARCHAR(500),          -- s3://bucket/features/...
        feature_size_gb DECIMAL(10, 2),

        created_at TIMESTAMP,

        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
        INDEX idx_experiment (experiment_id)
    );

    CREATE TABLE ensembles (
        ensemble_id VARCHAR(255) PRIMARY KEY,
        experiment_id VARCHAR(255) NOT NULL,

        -- Ensemble configuration
        ensemble_method VARCHAR(50),        -- stacking, blending, voting
        base_model_ids JSONB,               -- list of trial_ids
        meta_model_type VARCHAR(100),       -- logistic_regression, xgboost

        -- Performance
        ensemble_metric FLOAT,
        improvement_over_best FLOAT,        -- % improvement

        model_path VARCHAR(500),
        created_at TIMESTAMP,

        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
        INDEX idx_experiment (experiment_id)
    );
    ```

---

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Automated Feature Engineering

    ### Feature Generation Pipeline

    **Feature Transformation Categories:**

    1. **Numerical Transformations**: log, sqrt, polynomial, binning
    2. **Categorical Encoding**: one-hot, target encoding, frequency encoding
    3. **Feature Interactions**: multiplication, division, aggregations
    4. **Time-based Features**: day_of_week, month, hour, time_since
    5. **Statistical Features**: rolling mean, std, percentiles

    ```python
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.feature_selection import mutual_info_classif

    class AutoFeatureEngineer:
        def __init__(self, max_features=500, selection_threshold=0.01):
            self.max_features = max_features
            self.selection_threshold = selection_threshold
            self.generated_features = []
            self.feature_importance = {}

        def generate_features(self, df, target_col):
            """Automatically generate features from raw dataset"""
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # 1. Numerical transformations
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                # Log transform (for positive values)
                if (X[col] > 0).all():
                    X[f'{col}_log'] = np.log1p(X[col])
                    self.generated_features.append(f'{col}_log')

                # Square root transform
                if (X[col] >= 0).all():
                    X[f'{col}_sqrt'] = np.sqrt(X[col])
                    self.generated_features.append(f'{col}_sqrt')

                # Polynomial features (degree 2)
                X[f'{col}_squared'] = X[col] ** 2
                self.generated_features.append(f'{col}_squared')

                # Binning into quantiles
                X[f'{col}_binned'] = pd.qcut(X[col], q=10, labels=False, duplicates='drop')
                self.generated_features.append(f'{col}_binned')

            # 2. Feature interactions (pairwise)
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            for i, col1 in enumerate(numerical_cols[:10]):  # Limit to avoid explosion
                for col2 in numerical_cols[i+1:10]:
                    # Multiplication
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    self.generated_features.append(f'{col1}_x_{col2}')

                    # Division (avoid division by zero)
                    X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-5)
                    self.generated_features.append(f'{col1}_div_{col2}')

            # 3. Categorical encoding
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                # Frequency encoding
                freq_encoding = X[col].value_counts(normalize=True).to_dict()
                X[f'{col}_freq'] = X[col].map(freq_encoding)
                self.generated_features.append(f'{col}_freq')

                # Target encoding (mean of target per category)
                target_encoding = df.groupby(col)[target_col].mean().to_dict()
                X[f'{col}_target_enc'] = X[col].map(target_encoding)
                self.generated_features.append(f'{col}_target_enc')

            # 4. Statistical aggregations (if time series data)
            if 'timestamp' in df.columns or 'date' in df.columns:
                # Rolling window features
                for col in numerical_cols[:5]:  # Limit to important columns
                    X[f'{col}_rolling_mean_7'] = X[col].rolling(window=7).mean()
                    X[f'{col}_rolling_std_7'] = X[col].rolling(window=7).std()
                    self.generated_features.append(f'{col}_rolling_mean_7')
                    self.generated_features.append(f'{col}_rolling_std_7')

            return X, y

        def select_features(self, X, y, method='mutual_info'):
            """Select top features based on importance"""
            # Calculate feature importance
            if method == 'mutual_info':
                # Mutual information for classification
                importances = mutual_info_classif(X.fillna(0), y)
                self.feature_importance = dict(zip(X.columns, importances))
            elif method == 'correlation':
                # Correlation with target (for regression)
                correlations = X.corrwith(y).abs()
                self.feature_importance = correlations.to_dict()

            # Sort features by importance
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Select top features
            selected_features = [
                feat for feat, importance in sorted_features[:self.max_features]
                if importance > self.selection_threshold
            ]

            return X[selected_features]

        def fit_transform(self, df, target_col):
            """End-to-end feature engineering pipeline"""
            # Generate features
            X, y = self.generate_features(df, target_col)

            # Select important features
            X_selected = self.select_features(X, y)

            print(f"Generated {len(self.generated_features)} features")
            print(f"Selected {len(X_selected.columns)} features")

            return X_selected, y

    # Usage
    feature_engineer = AutoFeatureEngineer(max_features=200)
    X_transformed, y = feature_engineer.fit_transform(df, target_col='label')

    # Top 10 features by importance
    top_features = sorted(
        feature_engineer.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    print("Top 10 features:", top_features)
    ```

    ---

    ## 3.2 Neural Architecture Search (NAS)

    ### ENAS (Efficient Neural Architecture Search)

    **Concept**: Share weights across child models to reduce search cost from 1000s of GPU-days to ~1 GPU-day.

    ```python
    import torch
    import torch.nn as nn
    import numpy as np

    class ENASSearchSpace(nn.Module):
        """ENAS search space with shared weights"""
        def __init__(self, num_nodes=4, num_ops=5):
            super().__init__()
            self.num_nodes = num_nodes
            self.num_ops = num_ops

            # Define operation choices
            self.ops = nn.ModuleList([
                nn.Conv2d(32, 32, 3, padding=1),  # conv3x3
                nn.Conv2d(32, 32, 5, padding=2),  # conv5x5
                nn.MaxPool2d(3, stride=1, padding=1),  # maxpool
                nn.AvgPool2d(3, stride=1, padding=1),  # avgpool
                nn.Identity(),                     # skip connection
            ])

            # Controller LSTM for architecture sampling
            self.controller = nn.LSTM(input_size=32, hidden_size=64, num_layers=1)
            self.controller_head = nn.Linear(64, num_ops)

        def sample_architecture(self):
            """Sample architecture using controller LSTM"""
            architecture = []
            hidden = None

            for node in range(self.num_nodes):
                # Sample operation for this node
                if hidden is None:
                    hidden = (torch.zeros(1, 1, 64), torch.zeros(1, 1, 64))

                # LSTM step
                lstm_input = torch.randn(1, 1, 32)
                output, hidden = self.controller(lstm_input, hidden)

                # Sample operation via softmax
                logits = self.controller_head(output.squeeze(0))
                probs = torch.softmax(logits, dim=-1)
                op_id = torch.multinomial(probs, 1).item()

                architecture.append(op_id)

            return architecture

        def forward(self, x, architecture):
            """Forward pass with given architecture"""
            for node_id, op_id in enumerate(architecture):
                x = self.ops[op_id](x)
                x = nn.ReLU()(x)
            return x

    class ENASController:
        """Train controller to discover high-performing architectures"""
        def __init__(self, search_space, num_samples=100):
            self.search_space = search_space
            self.num_samples = num_samples
            self.architecture_history = []
            self.reward_history = []

        def train_controller(self, train_loader, val_loader, num_epochs=50):
            """Train controller using REINFORCE algorithm"""
            optimizer = torch.optim.Adam(
                self.search_space.controller.parameters(),
                lr=0.001
            )

            for epoch in range(num_epochs):
                # Sample architectures
                architectures = []
                rewards = []

                for _ in range(self.num_samples):
                    # Sample architecture
                    arch = self.search_space.sample_architecture()
                    architectures.append(arch)

                    # Train child model with sampled architecture
                    reward = self.train_child_model(
                        arch, train_loader, val_loader, epochs=5
                    )
                    rewards.append(reward)

                # Update controller to maximize reward (accuracy)
                baseline = np.mean(rewards)
                advantages = torch.tensor(rewards) - baseline

                # REINFORCE gradient update
                optimizer.zero_grad()
                loss = 0
                for arch, advantage in zip(architectures, advantages):
                    # Compute log probability of architecture
                    log_prob = self.compute_log_prob(arch)
                    loss -= log_prob * advantage  # Policy gradient

                loss.backward()
                optimizer.step()

                # Track best architecture
                best_idx = np.argmax(rewards)
                self.architecture_history.append(architectures[best_idx])
                self.reward_history.append(rewards[best_idx])

                print(f"Epoch {epoch}: Best reward = {rewards[best_idx]:.4f}")

            # Return best architecture found
            best_epoch = np.argmax(self.reward_history)
            return self.architecture_history[best_epoch]

        def train_child_model(self, architecture, train_loader, val_loader, epochs=5):
            """Train child model and return validation accuracy"""
            model = self.search_space
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            # Train for few epochs (quick evaluation)
            for epoch in range(epochs):
                model.train()
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model(data, architecture)
                    output = nn.AdaptiveAvgPool2d(1)(output).squeeze()
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data, architecture)
                    output = nn.AdaptiveAvgPool2d(1)(output).squeeze()
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)

            accuracy = correct / total
            return accuracy

        def compute_log_prob(self, architecture):
            """Compute log probability of sampling architecture"""
            # Simplified: assume uniform probability for now
            return torch.log(torch.tensor(1.0 / self.search_space.num_ops ** len(architecture)))

    # Usage
    search_space = ENASSearchSpace(num_nodes=4, num_ops=5)
    controller = ENASController(search_space, num_samples=20)

    # Run NAS
    best_architecture = controller.train_controller(
        train_loader, val_loader, num_epochs=50
    )
    print("Best architecture:", best_architecture)
    # Output: Best architecture: [2, 0, 4, 1] (ops for each node)
    ```

    ### DARTS (Differentiable Architecture Search)

    **Concept**: Make architecture search differentiable via softmax over operations, enabling gradient-based optimization.

    ```python
    class DARTSSearchSpace(nn.Module):
        """DARTS continuous relaxation of architecture"""
        def __init__(self, num_nodes=4):
            super().__init__()
            self.num_nodes = num_nodes

            # Operations
            self.ops = nn.ModuleList([
                nn.Conv2d(32, 32, 3, padding=1),
                nn.Conv2d(32, 32, 5, padding=2),
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.AvgPool2d(3, stride=1, padding=1),
                nn.Identity(),
            ])

            # Architecture parameters (alpha)
            self.alpha = nn.Parameter(torch.randn(num_nodes, len(self.ops)))

        def forward(self, x):
            """Mixed operation: weighted sum of all operations"""
            for node in range(self.num_nodes):
                # Softmax over operations
                weights = torch.softmax(self.alpha[node], dim=0)

                # Compute weighted sum of operations
                node_output = sum(
                    w * op(x) for w, op in zip(weights, self.ops)
                )
                x = node_output

            return x

        def discretize_architecture(self):
            """Convert continuous alpha to discrete architecture"""
            architecture = []
            for node in range(self.num_nodes):
                # Select operation with highest weight
                op_id = torch.argmax(self.alpha[node]).item()
                architecture.append(op_id)
            return architecture

    class DARTSOptimizer:
        """Bi-level optimization for DARTS"""
        def __init__(self, model, train_loader, val_loader):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader

            # Separate optimizers for weights and architecture
            self.weight_optimizer = torch.optim.SGD(
                [p for name, p in model.named_parameters() if name != 'alpha'],
                lr=0.025, momentum=0.9, weight_decay=3e-4
            )
            self.arch_optimizer = torch.optim.Adam(
                [model.alpha],
                lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3
            )

        def train_step(self):
            """Alternating optimization"""
            # Step 1: Update architecture alpha on validation set
            self.model.train()
            val_data, val_target = next(iter(self.val_loader))

            self.arch_optimizer.zero_grad()
            output = self.model(val_data)
            val_loss = nn.CrossEntropyLoss()(output, val_target)
            val_loss.backward()
            self.arch_optimizer.step()

            # Step 2: Update model weights on training set
            train_data, train_target = next(iter(self.train_loader))

            self.weight_optimizer.zero_grad()
            output = self.model(train_data)
            train_loss = nn.CrossEntropyLoss()(output, train_target)
            train_loss.backward()
            self.weight_optimizer.step()

            return train_loss.item(), val_loss.item()

        def search(self, num_epochs=50):
            """Run DARTS architecture search"""
            for epoch in range(num_epochs):
                train_loss, val_loss = self.train_step()

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train loss = {train_loss:.4f}, Val loss = {val_loss:.4f}")
                    print(f"Alpha:\n{self.model.alpha.data}")

            # Discretize final architecture
            final_architecture = self.model.discretize_architecture()
            return final_architecture

    # Usage
    darts_model = DARTSSearchSpace(num_nodes=4)
    darts_optimizer = DARTSOptimizer(darts_model, train_loader, val_loader)

    best_architecture = darts_optimizer.search(num_epochs=50)
    print("Discovered architecture:", best_architecture)
    ```

    ---

    ## 3.3 Hyperparameter Optimization

    ### Bayesian Optimization with Gaussian Processes

    ```python
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.stats import norm

    class BayesianOptimizer:
        """Bayesian optimization for hyperparameter tuning"""
        def __init__(self, objective_function, bounds, n_iter=100):
            self.objective_function = objective_function
            self.bounds = bounds  # [(min, max) for each hyperparameter]
            self.n_iter = n_iter

            self.X_observed = []  # Hyperparameters tried
            self.y_observed = []  # Corresponding metrics

            # Gaussian Process model
            self.gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )

        def acquisition_function(self, X, xi=0.01):
            """Expected Improvement (EI) acquisition function"""
            if len(self.y_observed) == 0:
                return np.ones(len(X))

            # Predict mean and std at X
            mu, sigma = self.gp.predict(X, return_std=True)

            # Current best observed value
            y_best = np.max(self.y_observed)

            # Expected improvement
            with np.errstate(divide='warn'):
                improvement = mu - y_best - xi
                Z = improvement / sigma
                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0

            return ei

        def propose_location(self):
            """Propose next hyperparameter configuration to try"""
            # Random search over candidate points
            n_candidates = 1000
            X_candidates = np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds],
                size=(n_candidates, len(self.bounds))
            )

            # Evaluate acquisition function
            ei_values = self.acquisition_function(X_candidates)

            # Return candidate with highest EI
            best_idx = np.argmax(ei_values)
            return X_candidates[best_idx]

        def optimize(self):
            """Run Bayesian optimization loop"""
            # Random initialization (5 trials)
            for _ in range(5):
                X_random = [
                    np.random.uniform(low=b[0], high=b[1])
                    for b in self.bounds
                ]
                y_random = self.objective_function(X_random)
                self.X_observed.append(X_random)
                self.y_observed.append(y_random)

            # Bayesian optimization loop
            for i in range(5, self.n_iter):
                # Fit GP on observed data
                self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

                # Propose next configuration
                X_next = self.propose_location()

                # Evaluate objective function
                y_next = self.objective_function(X_next)

                # Update observations
                self.X_observed.append(X_next)
                self.y_observed.append(y_next)

                # Log progress
                best_y = np.max(self.y_observed)
                print(f"Iteration {i}: Best metric = {best_y:.4f}, Current = {y_next:.4f}")

            # Return best hyperparameters found
            best_idx = np.argmax(self.y_observed)
            best_X = self.X_observed[best_idx]
            best_y = self.y_observed[best_idx]

            return best_X, best_y

    # Usage example
    def objective_function(hyperparams):
        """Train model and return validation accuracy"""
        learning_rate, batch_size, dropout = hyperparams

        # Train model with these hyperparameters
        model = create_model(dropout=dropout)
        train_model(model, lr=learning_rate, batch_size=int(batch_size), epochs=10)

        # Return validation accuracy (maximize)
        accuracy = evaluate(model, val_loader)
        return accuracy

    # Define search space
    bounds = [
        (1e-5, 1e-2),   # learning_rate
        (16, 256),       # batch_size
        (0.1, 0.5)       # dropout
    ]

    optimizer = BayesianOptimizer(objective_function, bounds, n_iter=50)
    best_hyperparams, best_metric = optimizer.optimize()

    print(f"Best hyperparameters: lr={best_hyperparams[0]:.6f}, "
          f"batch_size={int(best_hyperparams[1])}, dropout={best_hyperparams[2]:.3f}")
    print(f"Best validation accuracy: {best_metric:.4f}")
    ```

    ### Genetic Algorithm for HPO

    ```python
    import random

    class GeneticOptimizer:
        """Genetic algorithm for hyperparameter optimization"""
        def __init__(self, objective_function, bounds, population_size=20, generations=50):
            self.objective_function = objective_function
            self.bounds = bounds
            self.population_size = population_size
            self.generations = generations

        def initialize_population(self):
            """Random initialization"""
            population = []
            for _ in range(self.population_size):
                individual = [
                    random.uniform(b[0], b[1]) for b in self.bounds
                ]
                population.append(individual)
            return population

        def evaluate_population(self, population):
            """Evaluate fitness (metric) for each individual"""
            fitness = []
            for individual in population:
                metric = self.objective_function(individual)
                fitness.append(metric)
            return fitness

        def selection(self, population, fitness, k=5):
            """Tournament selection"""
            selected = []
            for _ in range(len(population)):
                # Randomly select k individuals
                tournament = random.sample(list(zip(population, fitness)), k)
                # Select best from tournament
                winner = max(tournament, key=lambda x: x[1])
                selected.append(winner[0])
            return selected

        def crossover(self, parent1, parent2):
            """Single-point crossover"""
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2

        def mutation(self, individual, mutation_rate=0.1):
            """Random mutation"""
            mutated = individual.copy()
            for i in range(len(mutated)):
                if random.random() < mutation_rate:
                    mutated[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
            return mutated

        def optimize(self):
            """Run genetic algorithm"""
            # Initialize population
            population = self.initialize_population()

            for generation in range(self.generations):
                # Evaluate fitness
                fitness = self.evaluate_population(population)

                # Track best individual
                best_idx = np.argmax(fitness)
                best_individual = population[best_idx]
                best_fitness = fitness[best_idx]

                print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

                # Selection
                selected = self.selection(population, fitness)

                # Crossover and mutation
                next_population = []
                for i in range(0, len(selected), 2):
                    parent1 = selected[i]
                    parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

                    # Crossover
                    child1, child2 = self.crossover(parent1, parent2)

                    # Mutation
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)

                    next_population.extend([child1, child2])

                population = next_population[:self.population_size]

            # Return best individual
            final_fitness = self.evaluate_population(population)
            best_idx = np.argmax(final_fitness)
            return population[best_idx], final_fitness[best_idx]

    # Usage
    genetic_opt = GeneticOptimizer(
        objective_function,
        bounds=[(1e-5, 1e-2), (16, 256), (0.1, 0.5)],
        population_size=20,
        generations=50
    )

    best_hyperparams, best_fitness = genetic_opt.optimize()
    print("Best hyperparameters (genetic):", best_hyperparams)
    ```

    ---

    ## 3.4 Model Ensembling

    ### Stacking Ensemble

    ```python
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    class StackingEnsemble:
        """Stacking ensemble with meta-learner"""
        def __init__(self, base_models, meta_model):
            self.base_models = base_models
            self.meta_model = meta_model

        def fit(self, X_train, y_train, X_val, y_val):
            """Train stacking ensemble"""
            # Step 1: Train base models and get predictions
            base_predictions_train = []
            base_predictions_val = []

            for i, model in enumerate(self.base_models):
                print(f"Training base model {i+1}/{len(self.base_models)}...")

                # Train on full training set
                model.fit(X_train, y_train)

                # Get cross-validated predictions on training set
                # (avoid overfitting)
                train_preds = cross_val_predict(
                    model, X_train, y_train, cv=5, method='predict_proba'
                )[:, 1]  # Get probability of positive class
                base_predictions_train.append(train_preds)

                # Get predictions on validation set
                val_preds = model.predict_proba(X_val)[:, 1]
                base_predictions_val.append(val_preds)

            # Stack predictions horizontally (each column = one model's predictions)
            X_train_meta = np.column_stack(base_predictions_train)
            X_val_meta = np.column_stack(base_predictions_val)

            # Step 2: Train meta-model on base model predictions
            print("Training meta-model...")
            self.meta_model.fit(X_train_meta, y_train)

            # Evaluate ensemble
            train_score = self.meta_model.score(X_train_meta, y_train)
            val_score = self.meta_model.score(X_val_meta, y_val)

            print(f"Stacking ensemble - Train: {train_score:.4f}, Val: {val_score:.4f}")

            return self

        def predict(self, X_test):
            """Make predictions with ensemble"""
            # Get predictions from base models
            base_predictions = []
            for model in self.base_models:
                preds = model.predict_proba(X_test)[:, 1]
                base_predictions.append(preds)

            # Stack predictions
            X_test_meta = np.column_stack(base_predictions)

            # Meta-model makes final prediction
            return self.meta_model.predict(X_test_meta)

    # Usage
    base_models = [
        RandomForestClassifier(n_estimators=100, max_depth=10),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
        # Add more diverse models (SVM, neural net, etc.)
    ]

    meta_model = LogisticRegression()

    stacking = StackingEnsemble(base_models, meta_model)
    stacking.fit(X_train, y_train, X_val, y_val)

    predictions = stacking.predict(X_test)
    ```

    ### Weighted Blending

    ```python
    class WeightedBlendingEnsemble:
        """Weighted average of model predictions"""
        def __init__(self, models, weights=None):
            self.models = models
            self.weights = weights or [1.0 / len(models)] * len(models)

        def optimize_weights(self, X_val, y_val):
            """Find optimal weights using validation set"""
            from scipy.optimize import minimize

            def objective(weights):
                """Minimize validation loss"""
                # Ensure weights sum to 1
                weights = weights / weights.sum()

                # Weighted predictions
                preds = np.zeros(len(X_val))
                for model, w in zip(self.models, weights):
                    preds += w * model.predict_proba(X_val)[:, 1]

                # Binary cross-entropy loss
                loss = -np.mean(
                    y_val * np.log(preds + 1e-10) + (1 - y_val) * np.log(1 - preds + 1e-10)
                )
                return loss

            # Optimize weights
            initial_weights = np.ones(len(self.models)) / len(self.models)
            bounds = [(0, 1) for _ in self.models]
            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds)

            self.weights = result.x / result.x.sum()
            print(f"Optimized weights: {self.weights}")

            return self

        def predict(self, X_test):
            """Weighted average predictions"""
            preds = np.zeros(len(X_test))
            for model, w in zip(self.models, self.weights):
                preds += w * model.predict_proba(X_test)[:, 1]

            return (preds > 0.5).astype(int)

    # Usage
    models = [model1, model2, model3]  # Trained models
    blending = WeightedBlendingEnsemble(models)
    blending.optimize_weights(X_val, y_val)

    predictions = blending.predict(X_test)
    ```

---

=== "📈 Step 4: Scalability & Performance"

    ## 4.1 GPU Resource Pooling

    ### Dynamic Resource Allocation

    ```python
    import kubernetes
    from kubernetes import client, config

    class GPUResourceManager:
        """Manage GPU allocation for AutoML experiments"""
        def __init__(self, total_gpus=100):
            self.total_gpus = total_gpus
            self.available_gpus = total_gpus
            self.experiment_allocations = {}

            # Kubernetes API client
            config.load_kube_config()
            self.v1 = client.CoreV1Api()
            self.batch_v1 = client.BatchV1Api()

        def allocate_gpus(self, experiment_id, num_gpus, priority='normal'):
            """Allocate GPUs for experiment"""
            if num_gpus > self.available_gpus:
                # Queue experiment
                return False, f"Only {self.available_gpus} GPUs available"

            # Allocate GPUs
            self.available_gpus -= num_gpus
            self.experiment_allocations[experiment_id] = {
                'gpus': num_gpus,
                'priority': priority,
                'start_time': time.time()
            }

            return True, f"Allocated {num_gpus} GPUs"

        def release_gpus(self, experiment_id):
            """Release GPUs after experiment completion"""
            if experiment_id in self.experiment_allocations:
                gpus_freed = self.experiment_allocations[experiment_id]['gpus']
                self.available_gpus += gpus_freed
                del self.experiment_allocations[experiment_id]
                return gpus_freed
            return 0

        def create_training_job(self, experiment_id, num_gpus, docker_image, command):
            """Create Kubernetes job with GPU resources"""
            job = client.V1Job(
                api_version="batch/v1",
                kind="Job",
                metadata=client.V1ObjectMeta(name=f"automl-{experiment_id}"),
                spec=client.V1JobSpec(
                    template=client.V1PodTemplateSpec(
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="trainer",
                                    image=docker_image,
                                    command=command,
                                    resources=client.V1ResourceRequirements(
                                        limits={
                                            "nvidia.com/gpu": str(num_gpus),
                                            "memory": "32Gi",
                                            "cpu": "8"
                                        }
                                    ),
                                    env=[
                                        client.V1EnvVar(name="EXPERIMENT_ID", value=experiment_id)
                                    ]
                                )
                            ],
                            restart_policy="Never",
                            # GPU node selector
                            node_selector={"accelerator": "nvidia-tesla-v100"}
                        )
                    )
                )
            )

            # Create job
            self.batch_v1.create_namespaced_job(namespace="automl", body=job)
            return job.metadata.name

        def monitor_gpu_utilization(self):
            """Query GPU utilization across cluster"""
            # Use NVIDIA DCGM or Prometheus metrics
            # This is a simplified version
            utilization = {}
            for exp_id, allocation in self.experiment_allocations.items():
                # Query GPU metrics (mock)
                utilization[exp_id] = {
                    'gpus': allocation['gpus'],
                    'utilization': 85.0,  # % utilization
                    'memory_used': 70.0   # % memory
                }
            return utilization
    ```

    ---

    ## 4.2 Early Stopping and Resource Optimization

    ### Successive Halving (Hyperband)

    ```python
    class SuccessiveHalving:
        """Hyperband-style early stopping for HPO"""
        def __init__(self, max_resource=81, eta=3):
            """
            max_resource: max epochs per trial (e.g., 81)
            eta: reduction factor (e.g., 3 means keep top 1/3)
            """
            self.max_resource = max_resource
            self.eta = eta

        def run(self, configurations, objective_function):
            """Run successive halving"""
            n = len(configurations)
            r = self.max_resource / (self.eta ** int(np.log(n) / np.log(self.eta)))

            # Successive halving rounds
            active_configs = configurations.copy()
            resource = int(r)

            while len(active_configs) > 1 and resource <= self.max_resource:
                print(f"\nRound: {len(active_configs)} configs, {resource} epochs each")

                # Train all active configs for 'resource' epochs
                results = []
                for config in active_configs:
                    metric = objective_function(config, num_epochs=resource)
                    results.append((config, metric))

                # Sort by performance
                results.sort(key=lambda x: x[1], reverse=True)

                # Keep top 1/eta configs
                n_keep = max(1, len(active_configs) // self.eta)
                active_configs = [config for config, _ in results[:n_keep]]

                # Increase resource
                resource *= self.eta

            # Return best configuration
            best_config = active_configs[0]
            best_metric = objective_function(best_config, num_epochs=self.max_resource)

            return best_config, best_metric

    # Usage
    configs = [
        {'lr': 0.001, 'batch_size': 64},
        {'lr': 0.01, 'batch_size': 128},
        {'lr': 0.0001, 'batch_size': 32},
        # ... 27 configurations total
    ]

    halving = SuccessiveHalving(max_resource=81, eta=3)
    best_config, best_metric = halving.run(configs, objective_function)
    print(f"Best config: {best_config}, Metric: {best_metric:.4f}")
    ```

    ---

    ## 4.3 Transfer Learning and Meta-Learning

    ### Warm-Starting with Pre-trained Models

    ```python
    class TransferLearningOptimizer:
        """Use transfer learning to speed up AutoML"""
        def __init__(self, pretrained_models):
            self.pretrained_models = pretrained_models

        def select_pretrained_model(self, dataset_features):
            """Select most similar pre-trained model based on dataset"""
            # Compare dataset characteristics
            best_model = None
            best_similarity = -1

            for model_name, model_info in self.pretrained_models.items():
                # Compute similarity (e.g., dataset size, num classes, domain)
                similarity = self.compute_similarity(
                    dataset_features,
                    model_info['dataset_features']
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_model = model_name

            return best_model

        def compute_similarity(self, dataset_a, dataset_b):
            """Cosine similarity between dataset features"""
            # Features: num_samples, num_features, num_classes, domain (text/image)
            vec_a = np.array([
                dataset_a['num_samples'],
                dataset_a['num_features'],
                dataset_a['num_classes']
            ])
            vec_b = np.array([
                dataset_b['num_samples'],
                dataset_b['num_features'],
                dataset_b['num_classes']
            ])

            # Cosine similarity
            similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            return similarity

        def fine_tune(self, pretrained_model, new_dataset, num_epochs=10):
            """Fine-tune pre-trained model on new dataset"""
            # Freeze early layers
            for i, layer in enumerate(pretrained_model.layers[:-3]):
                layer.trainable = False

            # Train only last layers
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, pretrained_model.parameters()),
                lr=1e-4
            )

            for epoch in range(num_epochs):
                for data, target in new_dataset:
                    optimizer.zero_grad()
                    output = pretrained_model(data)
                    loss = nn.CrossEntropyLoss()(output, target)
                    loss.backward()
                    optimizer.step()

            return pretrained_model

    # Usage
    pretrained_models = {
        'resnet50_imagenet': {
            'model': torchvision.models.resnet50(pretrained=True),
            'dataset_features': {
                'num_samples': 1e6,
                'num_features': 2048,
                'num_classes': 1000,
                'domain': 'image'
            }
        },
        'bert_base': {
            'model': transformers.BertModel.from_pretrained('bert-base-uncased'),
            'dataset_features': {
                'num_samples': 3e9,
                'num_features': 768,
                'num_classes': 2,
                'domain': 'text'
            }
        }
    }

    transfer_optimizer = TransferLearningOptimizer(pretrained_models)

    # Select best pre-trained model for new dataset
    dataset_features = {
        'num_samples': 50000,
        'num_features': 2048,
        'num_classes': 10,
        'domain': 'image'
    }

    best_model_name = transfer_optimizer.select_pretrained_model(dataset_features)
    pretrained_model = pretrained_models[best_model_name]['model']

    # Fine-tune on new dataset
    fine_tuned_model = transfer_optimizer.fine_tune(
        pretrained_model, new_dataset, num_epochs=10
    )
    ```

    ---

    ## 4.4 Feature Transformation Caching

    ### Distributed Feature Store

    ```python
    import redis
    import pickle

    class FeatureCache:
        """Cache engineered features to avoid recomputation"""
        def __init__(self, redis_host='localhost', redis_port=6379):
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=False
            )

        def cache_features(self, dataset_id, transformation_id, features):
            """Cache transformed features"""
            cache_key = f"features:{dataset_id}:{transformation_id}"

            # Serialize features
            features_bytes = pickle.dumps(features)

            # Store in Redis with 1-day TTL
            self.redis_client.setex(
                cache_key,
                86400,  # 1 day
                features_bytes
            )

        def get_features(self, dataset_id, transformation_id):
            """Retrieve cached features"""
            cache_key = f"features:{dataset_id}:{transformation_id}"

            features_bytes = self.redis_client.get(cache_key)
            if features_bytes:
                # Deserialize
                features = pickle.loads(features_bytes)
                return features
            return None

        def invalidate_cache(self, dataset_id):
            """Clear cache for dataset"""
            pattern = f"features:{dataset_id}:*"
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)

    # Usage in feature engineering
    feature_cache = FeatureCache()

    def compute_features_with_cache(dataset_id, transformation_config):
        """Compute features with caching"""
        transformation_id = hash(str(transformation_config))

        # Check cache first
        cached_features = feature_cache.get_features(dataset_id, transformation_id)
        if cached_features is not None:
            print("Cache hit! Returning cached features")
            return cached_features

        # Cache miss: compute features
        print("Cache miss. Computing features...")
        features = expensive_feature_engineering(dataset_id, transformation_config)

        # Cache for future use
        feature_cache.cache_features(dataset_id, transformation_id, features)

        return features
    ```

    ---

    ## 4.5 Distributed Training with Ray

    ```python
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    class RayDistributedAutoML:
        """Distributed AutoML using Ray Tune"""
        def __init__(self, num_gpus=4):
            ray.init(num_gpus=num_gpus)

        def train_function(self, config):
            """Training function for Ray Tune"""
            import torch
            import torch.nn as nn

            # Build model based on config
            model = self.build_model(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

            for epoch in range(config['num_epochs']):
                # Training loop
                train_loss = self.train_epoch(model, optimizer, train_loader)
                val_loss = self.validate(model, val_loader)

                # Report to Ray Tune
                tune.report(loss=val_loss, accuracy=accuracy)

        def run_automl(self, search_space, num_samples=100):
            """Run distributed AutoML search"""
            # Define search space
            config = {
                'lr': tune.loguniform(1e-5, 1e-1),
                'batch_size': tune.choice([16, 32, 64, 128]),
                'num_layers': tune.randint(1, 5),
                'hidden_size': tune.choice([64, 128, 256, 512]),
                'dropout': tune.uniform(0.1, 0.5),
                'num_epochs': 50
            }

            # ASHA scheduler for early stopping
            scheduler = ASHAScheduler(
                metric='loss',
                mode='min',
                max_t=50,
                grace_period=10,
                reduction_factor=3
            )

            # Run distributed search
            analysis = tune.run(
                self.train_function,
                config=config,
                num_samples=num_samples,
                scheduler=scheduler,
                resources_per_trial={'gpu': 1, 'cpu': 4},
                verbose=1
            )

            # Get best configuration
            best_config = analysis.best_config
            best_trial = analysis.best_trial

            print(f"Best config: {best_config}")
            print(f"Best validation loss: {best_trial.last_result['loss']:.4f}")

            return best_config

    # Usage
    ray_automl = RayDistributedAutoML(num_gpus=4)
    best_config = ray_automl.run_automl(search_space={}, num_samples=100)
    ```

---

=== "🎓 Step 5: Interview Tips & Common Questions"

    ## Question 1: How does AutoML differ from manual hyperparameter tuning?

    **Answer:**

    AutoML automates the entire ML pipeline:

    1. **Feature Engineering**: Automatically generates and selects features (polynomial, interactions, encoding)
    2. **Algorithm Selection**: Searches across multiple model types (XGBoost, Random Forest, Neural Nets)
    3. **Architecture Search**: Discovers optimal neural architectures via NAS (ENAS, DARTS)
    4. **Hyperparameter Optimization**: Bayesian optimization, genetic algorithms with early stopping
    5. **Ensembling**: Automatically combines top models via stacking/blending

    Manual tuning only optimizes hyperparameters for a fixed model/architecture.

    **Trade-off**: AutoML is compute-intensive (100+ trials) but produces near-optimal models with minimal human expertise.

    ---

    ## Question 2: Explain Neural Architecture Search (NAS) efficiency

    **Answer:**

    Traditional NAS trains each architecture from scratch (1000s of GPU-days). Efficient NAS methods:

    1. **ENAS (Efficient NAS)**: Share weights across child models. Train super-network once, sample architectures.
       - Speed: ~1 GPU-day vs 1000 GPU-days
       - Trade-off: Weight sharing may not perfectly predict final performance

    2. **DARTS (Differentiable)**: Make architecture search differentiable via softmax over operations.
       - Speed: ~4 GPU-days
       - Trade-off: Continuous relaxation approximates discrete choices

    3. **One-Shot NAS**: Train one over-parameterized network, prune to find architecture.
       - Speed: ~12 GPU-hours
       - Trade-off: Architecture quality depends on pruning strategy

    **Recommendation**: Use DARTS for research, ENAS for production (better parallelization).

    ---

    ## Question 3: How do you handle the cold start problem in Bayesian optimization?

    **Answer:**

    **Problem**: Bayesian optimization needs initial observations to build GP surrogate model.

    **Solutions**:

    1. **Random Initialization**: First 5-10 trials are random to explore space
    2. **Transfer Learning**: Use hyperparameters from similar datasets as priors
    3. **Meta-Learning**: Train meta-model on 1000s of past experiments to predict good starting hyperparameters
    4. **Warmstart**: If similar experiment exists, start from its best hyperparameters

    ```python
    # Meta-learning example
    def get_meta_learning_prior(dataset_features):
        """Predict good hyperparameters based on dataset"""
        # Train meta-model on historical experiments
        meta_model.predict({
            'num_samples': dataset_features['num_samples'],
            'num_features': dataset_features['num_features'],
            'task_type': 'classification'
        })
        # Returns: {'lr': 0.001, 'batch_size': 64}
    ```

    ---

    ## Question 4: What's the best ensembling strategy?

    **Answer:**

    | Method | When to Use | Pros | Cons |
    |--------|-------------|------|------|
    | **Voting** | Diverse models, similar performance | Simple, fast | No learning of weights |
    | **Weighted Blending** | Validation set available | Optimizes weights | Prone to overfitting |
    | **Stacking** | Plenty of data, diverse base models | Learns complex combinations | Slow, requires extra data |

    **Best Practice**: Use **stacking** for competitions/high-accuracy needs, **weighted blending** for production (faster inference).

    ```python
    # Stacking: train meta-model on base model predictions
    # Blending: optimize weights on validation set
    ```

    ---

    ## Question 5: How do you scale AutoML to 1000+ concurrent experiments?

    **Answer:**

    **Resource Management**:
    1. **GPU Pooling**: Kubernetes with GPU scheduling, dynamic allocation
    2. **Early Stopping**: Successive halving (Hyperband) kills bad trials early
    3. **Caching**: Cache feature transformations in Redis (avoid recomputation)
    4. **Distributed Training**: Use Ray/Dask for parallel trials

    **Cost Optimization**:
    1. **Spot Instances**: 70% discount for non-critical trials
    2. **Mixed CPU/GPU**: Use CPUs for classical ML, GPUs only for DL/NAS
    3. **Transfer Learning**: Warm-start from pre-trained models (fewer epochs)

    ```python
    # Example: 1000 experiments, 4 GPUs each = 4000 GPUs needed
    # With early stopping: 50% trials stop after 10 epochs → 2000 GPU equivalent
    # With spot instances: 70% discount → $1200/hour instead of $4000/hour
    ```

    ---

    ## Question 6: How do you ensure model reproducibility in AutoML?

    **Answer:**

    **Deterministic Experiment Tracking**:

    1. **Seed Management**: Set random seeds for all libraries
       ```python
       np.random.seed(42)
       torch.manual_seed(42)
       random.seed(42)
       ```

    2. **Versioning**: Track dataset version, code commit, framework versions
    3. **Config Freezing**: Store exact hyperparameters and architecture in metadata DB
    4. **Artifact Lineage**: Link model → trial → experiment → dataset → features

    **Example Metadata**:
    ```json
    {
      "experiment_id": "exp-123",
      "git_commit": "abc1234",
      "dataset_version": "v2.3",
      "feature_transformations": ["log_transform", "target_encoding"],
      "architecture": [2, 0, 4, 1],  # NAS discovered
      "hyperparameters": {"lr": 0.001, "batch_size": 64},
      "random_seed": 42,
      "model_path": "s3://bucket/models/exp-123-trial-45.pt"
    }
    ```

    ---

    ## Question 7: Estimate costs for 500 experiments per day

    **Answer:**

    ```
    Given:
    - 500 experiments/day
    - 200 trials per experiment average
    - 15 minutes per trial (with early stopping)
    - 30% GPU experiments (4 GPUs), 70% CPU (16 vCPUs)

    GPU Experiments (150 experiments):
    - Trials: 150 × 200 = 30,000 trials
    - GPU-hours: 30,000 × (15/60) × 4 GPUs = 30,000 GPU-hours/day
    - Cost: 30,000 × $1.50/hour = $45,000/day

    CPU Experiments (350 experiments):
    - Trials: 350 × 200 = 70,000 trials
    - vCPU-hours: 70,000 × (15/60) × 16 vCPUs = 280,000 vCPU-hours/day
    - Cost: 280,000 × $0.04/hour = $11,200/day

    Storage:
    - Models: 500 experiments × 10 models × 100 MB = 500 GB/day
    - S3 storage: $12/month (negligible)

    Total: $45,000 + $11,200 + $12 ≈ $56,000/day
    Cost per experiment: $56,000 / 500 = $112/experiment

    Optimizations:
    - Spot instances (70% discount) → $17,000/day
    - Better early stopping (50% reduction) → $8,500/day
    - Transfer learning (30% fewer epochs) → $6,000/day
    ```

    ---

    ## Common Pitfalls to Avoid

    1. **Not validating on holdout set**: Overfitting to validation set during HPO. Use nested CV.
    2. **Ignoring data leakage**: Feature engineering must be inside CV folds, not before.
    3. **Skipping ensemble**: Ensembles often give 2-5% improvement for free.
    4. **No early stopping**: Wastes 50%+ of compute on bad trials.
    5. **Fixed search space**: AutoML quality depends on good search space definition.
    6. **No meta-learning**: Reinventing hyperparameters for every dataset is inefficient.
    7. **Underestimating compute**: NAS can require 100+ GPU-hours per experiment.

---

=== "📝 Additional Resources"

    ## Key Concepts to Master

    1. **Neural Architecture Search**
       - ENAS (Efficient NAS with weight sharing)
       - DARTS (Differentiable architecture search)
       - NASNet, AmoebaNet, EfficientNet

    2. **Hyperparameter Optimization**
       - Bayesian optimization (Gaussian Processes)
       - Genetic algorithms
       - Hyperband / Successive halving
       - Population-based training (PBT)

    3. **Feature Engineering**
       - Automated transformations (log, polynomial, binning)
       - Categorical encoding (target, frequency, one-hot)
       - Feature interactions (multiplication, division)
       - Feature selection (mutual information, LASSO)

    4. **Ensemble Methods**
       - Stacking (meta-learner on base models)
       - Blending (weighted average)
       - Voting (majority vote)
       - Boosting vs bagging

    5. **Transfer Learning**
       - Fine-tuning pre-trained models
       - Meta-learning for warm-starts
       - Domain adaptation

    ## Real-World AutoML Systems

    **Google AutoML**:
    - Uses NAS for vision/NLP tasks
    - Transfer learning from ImageNet/BERT
    - Optimized for Google Cloud TPUs

    **H2O.ai**:
    - Classical ML focus (XGBoost, GLM, RF)
    - Automated feature engineering via recipes
    - Stacking ensembles by default

    **DataRobot**:
    - End-to-end platform (data prep → deployment)
    - Automated feature engineering
    - Model explainability (SHAP)

    **Amazon SageMaker Autopilot**:
    - Integrates with AWS ecosystem
    - Multi-framework (XGBoost, MXNet, PyTorch)
    - Automatic model tuning via Bayesian optimization

    ## Interview Strategies

    1. **Clarify requirements**: What's the compute budget? Dataset size? Acceptable latency?
    2. **Start with baseline**: Propose random search or grid search first, then Bayesian optimization
    3. **Discuss NAS trade-offs**: ENAS vs DARTS vs One-Shot NAS (speed vs quality)
    4. **Feature engineering importance**: Often 2-3x impact vs hyperparameter tuning
    5. **Ensembling strategy**: Stacking for competitions, blending for production
    6. **Scale considerations**: How to handle 1000+ concurrent experiments? (Ray, Kubernetes)

    ## Practice Problems

    - Design an **AutoML platform for time series forecasting**
    - Optimize **NAS controller** with REINFORCE algorithm
    - Build **meta-learning system** to predict good hyperparameters
    - Implement **distributed feature engineering** with Dask/Ray
    - Design **multi-tenant AutoML** with resource isolation and quota management

    ## References

    - **ENAS Paper**: "Efficient Neural Architecture Search via Parameter Sharing" (Pham et al., 2018)
    - **DARTS Paper**: "DARTS: Differentiable Architecture Search" (Liu et al., 2019)
    - **Hyperband**: "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" (Li et al., 2017)
    - **Google AutoML**: https://cloud.google.com/automl
    - **H2O.ai**: https://www.h2o.ai/products/h2o-automl/
    - **Ray Tune**: https://docs.ray.io/en/latest/tune/index.html
