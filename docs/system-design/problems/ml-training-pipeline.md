# Design ML Training Pipeline (Vertex AI/SageMaker/Kubeflow)

A distributed machine learning training infrastructure that handles petabyte-scale datasets, supports multiple ML frameworks, provides hyperparameter optimization, checkpointing/fault tolerance, and GPU/TPU resource management with multi-tenant isolation and cost tracking.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1000+ concurrent jobs, 10PB training datasets, 1000+ GPUs/TPUs, 1-7 day training runs |
| **Key Challenges** | Distributed training coordination, fault tolerance, resource allocation, hyperparameter tuning, cost optimization |
| **Core Concepts** | Distributed training (data/model parallelism), checkpointing, fault recovery, GPU pooling, Bayesian optimization, resource scheduling |
| **Companies** | Google Vertex AI, AWS SageMaker, Azure ML, Databricks ML, Kubeflow, Tesla, Meta |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Job Submission** | Submit training jobs with config (data path, model, hyperparameters, resources) | P0 (Must have) |
    | **Distributed Training** | Data parallelism, model parallelism, pipeline parallelism across GPUs/TPUs | P0 (Must have) |
    | **Hyperparameter Optimization** | Bayesian optimization, grid/random search, early stopping | P0 (Must have) |
    | **Checkpointing** | Periodic model/optimizer state saving with versioning | P0 (Must have) |
    | **Fault Tolerance** | Automatic recovery from node failures with checkpoint resume | P0 (Must have) |
    | **Progress Tracking** | Real-time metrics, logs, intermediate results, early stopping | P0 (Must have) |
    | **Resource Management** | GPU/TPU allocation, scheduling, quota enforcement, preemption | P0 (Must have) |
    | **Multi-framework Support** | PyTorch, TensorFlow, JAX, custom training scripts | P1 (Should have) |
    | **Experiment Management** | Track runs, compare metrics, versioning of models/data | P1 (Should have) |
    | **Cost Tracking** | Per-job compute cost, resource utilization reports | P1 (Should have) |
    | **Distributed Inference** | Run inference on trained checkpoints during training | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Data cleaning and preprocessing (users handle upstream)
    - Model architecture design assistance
    - Feature engineering
    - Post-training model evaluation platforms (separate service)
    - AutoML (separate service)
    - Model deployment/serving (separate system)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Job Start Latency** | < 5 minutes (container pull + setup) | Users expect quick training start |
    | **Training Throughput** | 1000+ concurrent jobs | Multi-tenant platform at scale |
    | **GPU Utilization** | > 85% average during job | Maximize expensive resource usage |
    | **Checkpoint Write Latency** | < 30 seconds for 10GB checkpoint | Minimize training interruption |
    | **Fault Recovery Time** | < 2 minutes | Resume training quickly after failure |
    | **Availability** | 99.9% uptime (8.76 hours/year) | Training infrastructure SLA |
    | **Data Throughput** | > 10 GB/sec to GPUs | Minimize data pipeline bottlenecks |
    | **Scalability** | Support 1000+ GPUs/TPUs per job | Enable large-scale training |
    | **Security** | Multi-tenant isolation, encryption, RBAC | Protect user models and data |
    | **Cost Efficiency** | < 3% overhead vs raw GPU usage | Minimize infrastructure tax |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Training Jobs:
    - Concurrent jobs: 1,000 jobs
    - Job duration: average 24 hours, range 1-168 hours
    - New jobs per day: 5,000 jobs
    - Peak submissions: 500 jobs/hour

    Resource Allocation:
    - Total GPUs: 1,000 GPUs (mix of V100, A100, H100)
    - Total TPUs: 500 TPU pods
    - Average GPUs per job: 8 GPUs (range: 1-128)
    - Average TPUs per job: 4 TPU cores

    Dataset Size:
    - Total training data: 10 PB stored
    - Average dataset size per job: 100 GB - 10 TB
    - Daily data ingestion: 500 GB
    - Data transfer to compute: > 10 GB/sec

    Hyperparameter Optimization:
    - HPO trials per job: 100 trials average
    - Max parallel trials: 10 trials
    - Trial duration: 1-6 hours each
    - Total HPO jobs: 500 HPO experiments/day

    Checkpoints:
    - Checkpoints per job: 10 checkpoints (hourly/per-epoch)
    - Average checkpoint size: 500 MB - 50 GB
    - Total checkpoint storage: 100 TB active
    - Checkpoint write frequency: every 10 minutes during training

    Monitoring:
    - Metrics per job: 50 metrics (loss, accuracy, lr, batch_time, etc.)
    - Sample frequency: every 10 seconds
    - Daily metric events: 1M metrics × 1000 jobs = 1B events/day
    - Log volume: 100 GB/day (1 MB per job average)
    ```

    ### Storage Estimates

    ```
    Checkpoints:
    - Active checkpoints: 1,000 jobs × 10 checkpoints × 5 GB = 50 TB
    - Total checkpoint history (90 days): 50 TB × 9 (3-day retention per job) = 450 TB
    - Compressed (20% ratio): 90 TB

    Training Datasets:
    - Total training data stored in object storage: 10 PB
    - Metadata (file manifests, checksums): 100 GB

    Model Artifacts:
    - Final trained models: 10,000 models × 2 GB average = 20 TB
    - Model versions and intermediate: 50 TB

    Logs and Metrics:
    - Training logs: 100 GB/day × 90 days = 9 TB (compressed: 1.8 TB)
    - Time-series metrics: 1KB per metric-point × 1B points/day × 90 days = 90 TB (compressed: 9 TB)

    Experiment Metadata:
    - Job configs, hyperparameters: 1 GB
    - Experiment tracking DB: 50 GB

    Total: 90 TB (checkpoints) + 10 TB (models) + 10 TB (logs/metrics) + 100 TB (datasets) ≈ 210 TB active
    ```

    ### Compute Estimates

    ```
    GPU Compute:
    - 1,000 GPUs total
    - 1,000 concurrent jobs × 8 GPUs/job = 8,000 GPUs needed
    - Cluster size with 85% utilization: 8,000 / 0.85 ≈ 9,400 GPUs
    - Cost: 9,400 GPUs × $1/hour = $9,400/hour = $225,600/day

    TPU Compute:
    - 500 TPU pods
    - Average utilization: 80%
    - Cost: 500 × $8/hour = $4,000/hour

    Network:
    - Data parallel bandwidth: 1,000 jobs × 10 GB/sec = 10 PB/sec
    - Realistic: 100 GB/sec aggregate network capacity needed

    Storage I/O:
    - Checkpoint writes: 500 jobs × 100 MB/min = 50 GB/min = 50 TB/hour
    - Data reads: 100 GB/sec
    - SSD cache: 10 TB for hot datasets
    ```

---

=== "🏗️ Step 2: High-Level Design"

    ## Architecture Diagram

    ```mermaid
    graph TB
        User["👤 User/Researcher"]
        CLI["📱 CLI/SDK/UI"]
        APIGateway["🚪 API Gateway"]

        JobController["📋 Job Controller"]
        ResourceScheduler["⚙️ Resource Scheduler"]
        MetricsCollector["📊 Metrics Collector"]

        Etcd["🔑 etcd<br/>(Config/State)"]
        JobDB["💾 Job DB<br/>(PostgreSQL)"]

        HPOService["🧪 HPO Service<br/>(Bayesian/Grid)"]
        CheckpointMgr["💾 Checkpoint Manager<br/>(S3/GCS)"]

        DockerRegistry["📦 Container Registry"]
        K8s["☸️ Kubernetes Cluster"]

        GPU1["🎮 GPU Node 1"]
        GPU2["🎮 GPU Node 2"]
        TPUN["🟪 TPU Node"]

        Trainer["🧠 Training Container<br/>(PyTorch/TF)"]

        MetricsStore["📈 Time-Series DB<br/>(Prometheus)"]
        LogStore["📝 Log Store<br/>(Elasticsearch)"]

        User -->|Submit Job| CLI
        CLI -->|REST/gRPC| APIGateway
        APIGateway -->|Validate & Queue| JobController
        JobController -->|Create Job CR| K8s
        JobController -->|Store Metadata| JobDB

        ResourceScheduler -->|Check Availability| Etcd
        ResourceScheduler -->|Allocate Resources| K8s
        K8s -->|Pull Image| DockerRegistry
        K8s -->|Schedule Pods| GPU1
        K8s -->|Schedule Pods| GPU2
        K8s -->|Schedule Pods| TPUN

        Trainer -->|Read Data| DataStore["📦 Object Storage<br/>(S3/GCS)"]
        Trainer -->|Save Checkpoint| CheckpointMgr
        Trainer -->|Send Metrics| MetricsCollector
        Trainer -->|Write Logs| LogStore

        MetricsCollector -->|Query HPO| HPOService
        HPOService -->|Suggest Hyperparams| JobController

        MetricsCollector -->|Store Metrics| MetricsStore
        MetricsCollector -->|Early Stopping Check| JobController

        JobController -->|Monitor Health| K8s
        JobController -->|Resume from Checkpoint| CheckpointMgr
    ```

    ---

    ## API Design

    ### Job Submission API

    ```protobuf
    message TrainingJobRequest {
        string job_name = 1;           // unique job identifier
        string user_id = 2;
        TrainingConfig config = 3;
        ResourceRequest resources = 4;
        HyperparameterTuning hpo_config = 5;
        CheckpointConfig checkpoint_config = 6;
    }

    message TrainingConfig {
        string framework = 1;          // pytorch, tensorflow, jax
        string docker_image = 2;       // user's training container
        string entry_point = 3;        // python training_script.py
        map<string, string> environment = 4;  // hyperparameters
        string data_path = 5;          // gs://bucket/data or s3://
        int32 training_steps = 6;      // total steps to train
    }

    message ResourceRequest {
        int32 num_gpus = 1;            // number of GPUs
        int32 num_tpus = 2;            // alternative: TPU cores
        string gpu_type = 3;           // v100, a100, h100
        int32 memory_gb = 4;           // RAM per node
        int32 num_nodes = 5;           // for distributed training
    }

    message HyperparameterTuning {
        string algorithm = 1;          // bayesian, grid, random
        map<string, Parameter> parameters = 2;
        int32 max_trials = 3;          // max HPO trials
        int32 max_parallel_trials = 4;
        string metric_name = 5;        // metric to optimize
        bool maximize = 6;             // true = maximize, false = minimize
    }

    message TrainingJobResponse {
        string job_id = 1;
        string status = 2;             // QUEUED, RUNNING, SUCCEEDED, FAILED
        string job_url = 3;            // link to job dashboard
    }
    ```

    ### Streaming Metrics API

    ```protobuf
    message MetricsUpdate {
        string job_id = 1;
        int64 step = 2;
        int64 timestamp = 3;
        map<string, float> metrics = 4;  // loss, accuracy, lr, batch_time
        bool should_early_stop = 5;
    }
    ```

    ---

    ## Database Schema

    ### Job Metadata Table

    ```sql
    CREATE TABLE training_jobs (
        job_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        job_name VARCHAR(255) NOT NULL,
        framework VARCHAR(50),              -- pytorch, tensorflow
        status VARCHAR(50),                 -- QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED
        created_at TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,

        -- Resource allocation
        num_gpus INT,
        gpu_type VARCHAR(50),
        num_nodes INT,

        -- Configuration
        docker_image VARCHAR(255),
        entry_point TEXT,
        data_path VARCHAR(500),

        -- Cost tracking
        estimated_cost_usd DECIMAL(10, 2),
        actual_cost_usd DECIMAL(10, 2),
        compute_hours DECIMAL(10, 2),

        -- HPO info
        is_hpo_job BOOLEAN,
        parent_hpo_job_id VARCHAR(255),
        hpo_algorithm VARCHAR(50),

        -- Results
        best_metric_value FLOAT,
        final_model_path VARCHAR(500),

        INDEX idx_user_created (user_id, created_at),
        INDEX idx_status (status),
        INDEX idx_hpo (is_hpo_job, parent_hpo_job_id)
    );

    CREATE TABLE checkpoints (
        checkpoint_id VARCHAR(255) PRIMARY KEY,
        job_id VARCHAR(255) NOT NULL,
        step INT NOT NULL,
        epoch INT,
        checkpoint_path VARCHAR(500),        -- s3://bucket/checkpoints/...
        checkpoint_size_bytes BIGINT,
        metrics JSONB,                       -- loss, accuracy at checkpoint
        created_at TIMESTAMP,
        is_best BOOLEAN,

        FOREIGN KEY (job_id) REFERENCES training_jobs(job_id),
        INDEX idx_job_step (job_id, step),
        INDEX idx_best (job_id, is_best)
    );

    CREATE TABLE hyperparameters (
        hp_id VARCHAR(255) PRIMARY KEY,
        job_id VARCHAR(255) NOT NULL,
        trial_number INT,
        hyperparams JSONB,                  -- learning_rate, batch_size, etc.
        metric_value FLOAT,
        status VARCHAR(50),
        created_at TIMESTAMP,

        FOREIGN KEY (job_id) REFERENCES training_jobs(job_id),
        INDEX idx_job_trial (job_id, trial_number)
    );
    ```

---

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Distributed Training Implementation

    ### Data Parallelism vs Model Parallelism

    **Data Parallelism** (Most Common):
    - Same model replicated across GPUs
    - Each GPU processes different batch
    - Gradients aggregated via AllReduce
    - Scales to ~100 GPUs efficiently

    ```python
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    def setup_distributed():
        # Called before training
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

    def train_with_ddp(model, dataloader, optimizer, num_epochs):
        # Wrap model for distributed training
        model = DDP(model, device_ids=[local_rank])

        for epoch in range(num_epochs):
            # DataLoader automatically shards data across GPUs
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()  # AllReduce happens automatically
                optimizer.step()

                if batch_idx % 100 == 0 and dist.get_rank() == 0:
                    print(f"Epoch {epoch} Loss: {loss.item()}")

            # Save checkpoint only on rank 0
            if dist.get_rank() == 0:
                save_checkpoint(model, optimizer, epoch)

    # Launch: torchrun --nproc_per_node=8 train.py
    if __name__ == "__main__":
        setup_distributed()
        train_with_ddp(model, train_loader, optimizer, num_epochs)
    ```

    **Model Parallelism** (For Large Models):
    - Different parts of model on different GPUs
    - Used for models > GPU memory
    - More complex, higher communication overhead

    ```python
    class ModelParallelModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(...).cuda(0)  # on GPU 0
            self.decoder = nn.Sequential(...).cuda(1)  # on GPU 1

        def forward(self, x):
            x = self.encoder(x)              # GPU 0
            x = x.to(self.decoder[0].device)  # Transfer to GPU 1
            x = self.decoder(x)              # GPU 1
            return x
    ```

    **Pipeline Parallelism** (GPipe, Megatron):
    - Split model into stages, parallelize across GPUs
    - Each GPU processes different micro-batches
    - Better GPU utilization than model parallelism

    ---

    ## 3.2 Hyperparameter Optimization

    ### Bayesian Optimization Implementation

    ```python
    from bayes_opt import BayesianOptimization

    def objective(learning_rate, batch_size, dropout):
        """Function to optimize (minimize validation loss)"""
        # Create job with these hyperparameters
        job_config = {
            'learning_rate': learning_rate,
            'batch_size': int(batch_size),
            'dropout': dropout
        }

        # Submit training job and wait for completion
        job = submit_training_job(job_config)
        val_loss = job.best_metric  # query validation loss

        return -val_loss  # maximize = minimize negative loss

    # Define search space
    pbounds = {
        'learning_rate': (1e-4, 1e-2),
        'batch_size': (32, 512),
        'dropout': (0.1, 0.5)
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        acquisition_function='ucb'
    )

    # Run optimization
    optimizer.maximize(init_points=5, n_iter=20)

    print("Best hyperparameters:", optimizer.max['params'])
    # Output: {'learning_rate': 0.001, 'batch_size': 128, 'dropout': 0.2}
    ```

    **Grid Search** (Exhaustive):
    ```python
    from itertools import product

    param_grid = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [64, 128, 256],
        'dropout': [0.1, 0.3, 0.5]
    }
    # Total: 3 × 3 × 3 = 27 trials

    for lr, bs, do in product(*param_grid.values()):
        submit_training_job({
            'learning_rate': lr,
            'batch_size': bs,
            'dropout': do
        })
    ```

    **Early Stopping Strategy**:
    ```python
    class EarlyStoppingController:
        def __init__(self, patience=5, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.counter = 0

        def check_stop(self, current_loss):
            """Returns True if should stop training"""
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
                return False

            self.counter += 1
            return self.counter >= self.patience
    ```

    ---

    ## 3.3 Checkpointing and Fault Tolerance

    ### Checkpoint Mechanism

    ```python
    import torch
    import os
    from datetime import datetime

    class CheckpointManager:
        def __init__(self, job_id, checkpoint_dir, max_checkpoints=5):
            self.job_id = job_id
            self.checkpoint_dir = checkpoint_dir
            self.max_checkpoints = max_checkpoints
            self.checkpoints = []

        def save_checkpoint(self, model, optimizer, epoch, metrics):
            """Save model state, optimizer state, and metadata"""
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

            # Save to distributed storage (S3/GCS)
            path = f"{self.checkpoint_dir}/epoch_{epoch}.pt"
            torch.save(checkpoint, path)

            # Metadata for recovery
            metadata = {
                'job_id': self.job_id,
                'epoch': epoch,
                'loss': metrics.get('loss'),
                'accuracy': metrics.get('accuracy'),
                'path': path,
                'size_bytes': os.path.getsize(path)
            }

            # Store metadata in database for quick recovery
            save_checkpoint_metadata(metadata)

            # Keep only last N checkpoints
            if len(self.checkpoints) >= self.max_checkpoints:
                old_ckpt = self.checkpoints.pop(0)
                os.remove(old_ckpt['path'])

            self.checkpoints.append(metadata)
            return path

        def load_checkpoint(self, model, optimizer, checkpoint_path):
            """Resume from checkpoint"""
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'], checkpoint['metrics']

    # Usage in training loop
    ckpt_manager = CheckpointManager(job_id="job-123", checkpoint_dir="s3://bucket/checkpoints")

    start_epoch = 0
    if resume_from_checkpoint:
        start_epoch, metrics = ckpt_manager.load_checkpoint(
            model, optimizer, last_checkpoint_path
        )
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        if epoch % 10 == 0:
            ckpt_manager.save_checkpoint(
                model, optimizer, epoch,
                {'loss': val_loss, 'accuracy': accuracy}
            )
    ```

    ### Fault Recovery System

    ```python
    class JobHealthMonitor:
        def __init__(self, job_id, check_interval=30):
            self.job_id = job_id
            self.check_interval = check_interval
            self.max_retries = 3

        def monitor_training(self):
            """Continuously monitor job health"""
            retry_count = 0

            while True:
                try:
                    # Check if pods are running
                    pods = get_job_pods(self.job_id)

                    if len(pods) < expected_num_gpus:
                        # Pod crashed, need restart
                        if retry_count < self.max_retries:
                            logging.warning(f"Pod failure detected, retrying...")
                            self.restart_job_from_checkpoint()
                            retry_count += 1
                        else:
                            self.fail_job("Max retries exceeded")
                            break

                    # Check metrics heartbeat
                    last_metric_time = get_last_metric_timestamp(self.job_id)
                    if time.time() - last_metric_time > 300:  # 5 min no metrics
                        logging.error("Training job stalled, restarting...")
                        self.restart_job_from_checkpoint()
                        retry_count += 1

                    time.sleep(self.check_interval)

                except Exception as e:
                    logging.error(f"Monitor error: {e}")

        def restart_job_from_checkpoint(self):
            """Recover from latest checkpoint"""
            latest_ckpt = get_latest_checkpoint(self.job_id)
            if latest_ckpt:
                # Kill failed pods
                kill_job_pods(self.job_id)
                # Resubmit job with checkpoint resume
                resubmit_job(self.job_id, resume_from=latest_ckpt)
            else:
                raise Exception("No checkpoint available for recovery")
    ```

---

=== "📈 Step 4: Scalability & Performance"

    ## 4.1 GPU Resource Pooling

    ### Resource Allocation Strategy

    ```python
    class GPUResourceScheduler:
        def __init__(self, total_gpus=1000):
            self.total_gpus = total_gpus
            self.available_gpus = total_gpus
            self.job_queue = []
            self.allocated_jobs = {}

        def request_resources(self, job_id, num_gpus, priority=1):
            """Request GPU resources with priority"""
            if num_gpus <= self.available_gpus:
                self.available_gpus -= num_gpus
                self.allocated_jobs[job_id] = {
                    'gpus': num_gpus,
                    'priority': priority,
                    'start_time': time.time()
                }
                return True
            else:
                # Add to queue, sorted by priority
                self.job_queue.append({
                    'job_id': job_id,
                    'gpus': num_gpus,
                    'priority': priority
                })
                return False

        def release_resources(self, job_id):
            """Release GPUs when job completes"""
            if job_id in self.allocated_jobs:
                gpus_freed = self.allocated_jobs[job_id]['gpus']
                self.available_gpus += gpus_freed
                del self.allocated_jobs[job_id]

                # Try to schedule queued jobs
                self.schedule_queued_jobs()

        def schedule_queued_jobs(self):
            """Allocate resources to waiting jobs by priority"""
            self.job_queue.sort(key=lambda x: (-x['priority'], x['gpus']))

            remaining_queue = []
            for job_req in self.job_queue:
                if self.request_resources(job_req['job_id'], job_req['gpus'], job_req['priority']):
                    continue  # Successfully scheduled
                else:
                    remaining_queue.append(job_req)

            self.job_queue = remaining_queue
    ```

    ### Distributed Coordinator (etcd/Zookeeper)

    ```python
    import etcd3

    class DistributedCoordinator:
        def __init__(self, etcd_host='localhost', etcd_port=2379):
            self.client = etcd3.client(host=etcd_host, port=etcd_port)

        def register_job(self, job_id, num_gpus, node_ids):
            """Register job allocation in distributed store"""
            job_info = {
                'job_id': job_id,
                'num_gpus': num_gpus,
                'nodes': node_ids,
                'rank_to_node': {i: node_ids[i % len(node_ids)] for i in range(num_gpus)},
                'status': 'RUNNING'
            }

            # Use lease for automatic cleanup on job end
            lease = self.client.lease(ttl=3600)
            self.client.put(
                key=f'/jobs/{job_id}',
                value=json.dumps(job_info),
                lease=lease
            )

        def get_rank_assignments(self, job_id, local_rank):
            """Get global rank and other ranks for synchronization"""
            job_info = json.loads(self.client.get(f'/jobs/{job_id}')[0])

            # Deterministic rank assignment
            node = socket.gethostname()
            global_rank = sum(1 for r, n in enumerate(job_info['rank_to_node'].items())
                            if r < len(job_info['rank_to_node']) and n == (local_rank, node))

            return {
                'global_rank': global_rank,
                'world_size': job_info['num_gpus'],
                'master_addr': job_info['nodes'][0],
                'master_port': 29500
            }
    ```

    ---

    ## 4.2 Monitoring and Metrics Collection

    ```python
    class TrainingMetricsCollector:
        def __init__(self, job_id, push_interval=10):
            self.job_id = job_id
            self.push_interval = push_interval
            self.metrics_buffer = []

        def record_metrics(self, step, metrics_dict):
            """Buffer metrics locally before pushing"""
            metric_point = {
                'job_id': self.job_id,
                'step': step,
                'timestamp': int(time.time()),
                'metrics': metrics_dict
            }

            self.metrics_buffer.append(metric_point)

            # Push to backend periodically
            if len(self.metrics_buffer) >= 100:
                self.push_metrics()

        def push_metrics(self):
            """Send metrics to time-series database"""
            if not self.metrics_buffer:
                return

            # Send to Prometheus/Datadog via HTTP or gRPC
            payload = {
                'job_id': self.job_id,
                'metrics': self.metrics_buffer
            }

            response = requests.post(
                'http://metrics-service:8080/api/v1/metrics',
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                self.metrics_buffer.clear()

        def get_current_metrics(self):
            """Real-time metrics for monitoring dashboard"""
            return {
                'loss': self.metrics_buffer[-1]['metrics']['loss'],
                'accuracy': self.metrics_buffer[-1]['metrics']['accuracy'],
                'throughput': self.metrics_buffer[-1]['metrics']['samples_per_sec'],
                'gpu_utilization': self.metrics_buffer[-1]['metrics']['gpu_util']
            }

    # Usage in training
    metrics_collector = TrainingMetricsCollector("job-456")

    for step, (data, labels) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, labels)

        metrics_collector.record_metrics(step, {
            'loss': loss.item(),
            'accuracy': (output.argmax(1) == labels).float().mean().item(),
            'samples_per_sec': batch_size / batch_time,
            'gpu_util': torch.cuda.utilization() if torch.cuda.is_available() else 0
        })
    ```

    ---

    ## 4.3 Cost Optimization

    ```python
    class CostTracker:
        def __init__(self, job_id, gpu_type='a100', hourly_rate=2.0):
            self.job_id = job_id
            self.gpu_type = gpu_type
            self.hourly_rate = hourly_rate
            self.start_time = time.time()
            self.checkpoints_saved = 0

        def estimate_job_cost(self, num_gpus, estimated_hours):
            """Estimate cost before job submission"""
            return num_gpus * self.hourly_rate * estimated_hours

        def get_actual_cost(self):
            """Calculate actual cost after job completion"""
            elapsed_hours = (time.time() - self.start_time) / 3600
            return self.num_gpus * self.hourly_rate * elapsed_hours

        def optimize_resource_usage(self):
            """Recommendations to reduce cost"""
            recommendations = []

            # Check GPU utilization
            if self.gpu_utilization < 60:
                recommendations.append("GPU utilization < 60%, consider reducing batch size or job parallelism")

            # Check checkpoint frequency
            if self.checkpoints_saved > 20:
                recommendations.append("Too many checkpoints, reduce save frequency")

            return recommendations
    ```

---

=== "🎓 Step 5: Interview Tips & Common Questions"

    ## Question 1: How do you handle distributed training across 100+ GPUs?

    **Answer:**
    - Use **Data Parallelism** (default): replicate model across GPUs, each processes different batch
    - Implement **AllReduce** for gradient synchronization (handled by DDP framework)
    - Use **NCCL** backend for high-speed GPU communication (vs Gloo for CPU)
    - Scale bottleneck: network bandwidth, not individual GPU throughput
    - For 100+ GPUs: may need **gradient accumulation** or **gradient compression** to reduce network overhead

    ```python
    # All-reduce with PyTorch DDP automatically handles this
    model = DDP(model)  # No manual synchronization needed
    loss.backward()     # Gradients are synchronized automatically
    ```

    ---

    ## Question 2: How do you recover from a failed GPU node?

    **Answer:**
    - Checkpoint **model + optimizer state** every N steps to fault-tolerant storage (S3/GCS)
    - Health monitor detects stalled training via metrics timeout
    - Automatic recovery:
      1. Kill failed pods
      2. Query latest checkpoint from database
      3. Resubmit job with `--resume-from-checkpoint` flag
      4. Load model/optimizer state and resume training from saved step
    - Recovery time < 2 minutes (mostly from checkpoint loading)
    - Trade-off: checkpointing adds ~5-10% overhead, but enables recovery

    ---

    ## Question 3: How do you optimize GPU utilization to 85%+?

    **Answer:**
    - **Batch size tuning**: Find sweet spot (too small = underutilized, too large = OOM)
    - **Data prefetching**: Load next batch while GPU processes current batch
    - **Mixed precision training**: Use FP16 for forward pass (less memory, same speed) - leverage **torch.amp**
    - **Gradient accumulation**: Increase effective batch size without OOM
    - **Overlap computation & communication**: Use async collective operations

    ```python
    # Example: Mixed precision + gradient accumulation
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 4

    for batch_idx, (data, target) in enumerate(train_loader):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target) / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    ```

    ---

    ## Question 4: Explain hyperparameter optimization trade-offs

    **Answer:**

    | Approach | Pros | Cons | Use Case |
    |----------|------|------|----------|
    | **Grid Search** | Simple, exhaustive | Exponential trials with params | Few hyperparams, small search space |
    | **Random Search** | Better for high dims, parallel | May miss good regions | Large search space |
    | **Bayesian** | Sample efficient, smart sampling | Slower per trial, complex | Limited budget, expensive trials |
    | **Population-based** | Dynamic adjustment during training | Complex to implement | Very large models |

    **Recommendation**: Start with **random search** (simple, parallelizable), then refine with **Bayesian** if budget allows.

    ---

    ## Question 5: What happens if network bandwidth is the bottleneck?

    **Answer:**
    - **Symptom**: Gradient sync time > computation time
    - **Solutions**:
      1. **Gradient compression**: Sparsify low-magnitude gradients (1-10% for vision models)
      2. **Mixed precision**: Reduce gradient size from FP32 to FP16
      3. **Delay sync**: Accumulate gradients over 2-4 steps, sync less frequently
      4. **Ring AllReduce**: Reduce communication from O(log N) to O(N) for specific topologies
      5. **Larger batches**: Amortize sync cost across more samples

    ```python
    # Gradient compression example
    def compress_gradients(gradients, compression_ratio=0.1):
        """Keep top K% of gradient values"""
        for grad in gradients:
            threshold = torch.quantile(grad.abs(), 1 - compression_ratio)
            grad[grad.abs() < threshold] = 0  # Sparse gradients
        return gradients
    ```

    ---

    ## Question 6: How do you track experiment lineage and reproduce results?

    **Answer:**
    - Store **immutable experiment metadata**:
      - Code commit hash (git)
      - Dataset version/checksum
      - Hyperparameters
      - Random seeds
      - Framework versions
    - **Checkpoint versioning**: every checkpoint maps to experiment + step
    - **Artifact tracking**: link final model → experiment metadata → data version
    - Use **MLflow, Weights & Biases, Kubeflow Pipelines** for automation

    ```python
    experiment_metadata = {
        'job_id': 'job-123',
        'git_commit': 'abc1234',
        'data_version': 'v2.3',
        'hyperparams': {'lr': 0.001, 'batch_size': 128},
        'random_seed': 42,
        'framework_versions': {
            'pytorch': '2.0.0',
            'numpy': '1.24.0'
        },
        'checkpoints': [
            {'step': 100, 'path': 's3://bucket/ckpt/epoch_1.pt'},
            {'step': 200, 'path': 's3://bucket/ckpt/epoch_2.pt'}
        ]
    }
    save_to_metadata_store(experiment_metadata)
    ```

    ---

    ## Question 7: Estimate costs for a large training job

    **Answer:**
    ```
    Given:
    - Model: 7B parameter LLM (similar to Llama 7B)
    - GPUs: 64 A100 GPUs (80GB)
    - Training duration: 2 weeks (336 hours)
    - AWS pricing: $2.68/hour per A100 on-demand

    Compute Cost:
    - 64 GPUs × $2.68/hour × 336 hours = $576,768

    Checkpoint/Storage Cost:
    - 10 checkpoints × 50GB each = 500 GB
    - S3 storage: 500 GB × $0.023/month ≈ $12 (negligible)

    Data Transfer Cost:
    - 1TB training data × $0.09/GB = $90

    Total: ~$577,000 for 2-week training run

    Optimizations to reduce cost:
    - Use spot instances (70% discount) → $173k
    - Reduce checkpoint frequency → save storage
    - Use mixed precision → reduce memory, enable larger batches
    ```

    ---

    ## Common Pitfalls to Avoid

    1. **Not checkpointing frequently**: Lose progress on failures. Checkpoint every 10-30 min.
    2. **Ignoring communication overhead**: Network can become 50% of time with many GPUs. Profile!
    3. **Fixed hyperparameters across jobs**: Use HPO framework for real projects, not manual tuning.
    4. **No experiment tracking**: Impossible to debug or reproduce results.
    5. **Overcomplicating model parallelism**: Start with data parallelism, use model parallelism only if model > GPU memory.
    6. **Not monitoring GPU utilization**: 40% utilization = wasting $$ on idle GPUs.
    7. **Insufficient error handling**: Single pod failure shouldn't kill entire job.

---

=== "📝 Additional Resources"

    ## Key Concepts to Master

    1. **Distributed Training Frameworks**
       - PyTorch Distributed (DDP, FSDP)
       - TensorFlow Distribution Strategies
       - Horovod (framework-agnostic)

    2. **Fault Tolerance Patterns**
       - Checkpoint/resume mechanism
       - Heartbeat monitoring
       - Exponential backoff for retries

    3. **Resource Scheduling**
       - Kubernetes Job scheduling
       - GPU pooling and preemption
       - Cost-aware scheduling

    4. **Monitoring Infrastructure**
       - Prometheus/Datadog for metrics
       - ELK stack for logs
       - Custom dashboards for training progress

    5. **Hyperparameter Search**
       - Bayesian optimization fundamentals
       - Hyperband algorithm
       - Population-based training (PBT)

    ## Interview Strategies

    1. **Clarify the problem**: Ask about scale (how many GPUs?), dataset size, time constraints
    2. **Start simple**: Propose single-node training first, then distributed
    3. **Discuss trade-offs**: Checkpoint frequency vs performance, cost vs speed
    4. **Code examples matter**: Show DDP/TF distributed, not just theory
    5. **Handle edge cases**: Failure recovery, skewed data distribution across GPUs
    6. **Performance analysis**: How would you profile? Where are bottlenecks?

    ## Practice Problems

    - Design a **Multi-GPU Training System** with 1000+ concurrent jobs
    - Optimize **LLM Pre-training** (100B parameters, 10K GPUs)
    - Build **Distributed Hyperparameter Search** with 1000 trials
    - Handle **GCS/S3 Data Loading** bottleneck with 100+ GPUs
    - Implement **Fault-tolerant Gradient Synchronization** for 500+ nodes

