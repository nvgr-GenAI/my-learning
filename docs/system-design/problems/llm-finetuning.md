# Design an LLM Fine-tuning Platform

A production-grade platform for fine-tuning large language models (LLMs) with parameter-efficient methods like LoRA/QLoRA, distributed training with FSDP/DeepSpeed, instruction tuning, RLHF, evaluation metrics, and automated deployment to serve customized models at scale.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1000+ fine-tuning jobs/month, models up to 70B parameters, 10TB+ training datasets, multi-tenant platform with GPU clusters |
| **Key Challenges** | Parameter-efficient fine-tuning (LoRA/QLoRA), distributed training coordination (FSDP/DeepSpeed), RLHF implementation, evaluation metrics, model versioning, GPU memory optimization |
| **Core Concepts** | LoRA/QLoRA adapters, 4-bit/8-bit quantization, FSDP sharding, instruction tuning, RLHF with PPO, gradient checkpointing, flash attention, model merging and deployment |
| **Companies** | OpenAI Fine-tuning API, Anthropic, Hugging Face, Together AI, Anyscale, Databricks, Amazon Bedrock Fine-tuning |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Dataset Management** | Upload instruction/chat datasets, validation, preprocessing, format conversion | P0 (Must have) |
    | **LoRA/QLoRA Fine-tuning** | Parameter-efficient training with rank decomposition and quantization | P0 (Must have) |
    | **Distributed Training** | FSDP/DeepSpeed for multi-GPU/multi-node training of 7B-70B models | P0 (Must have) |
    | **Instruction Tuning** | Fine-tune on instruction-response pairs with prompt templates | P0 (Must have) |
    | **Evaluation Metrics** | Perplexity, task-specific accuracy, ROUGE, BLEU, human eval integration | P0 (Must have) |
    | **Model Versioning** | Version control for base models, adapters, merged models, checkpoints | P0 (Must have) |
    | **RLHF Support** | Reward model training, PPO optimization, preference dataset management | P1 (Should have) |
    | **Deployment Integration** | Export to vLLM, TGI, or Ray Serve for inference | P1 (Should have) |
    | **Hyperparameter Optimization** | Grid search over learning rate, LoRA rank, batch size | P1 (Should have) |
    | **Multi-modal Fine-tuning** | Support for vision-language models (LLaVA, CLIP) | P2 (Nice to have) |
    | **Monitoring & Logging** | Training loss curves, GPU utilization, checkpoint progress | P0 (Must have) |
    | **Cost Tracking** | Per-job GPU-hour usage and cost estimation | P1 (Should have) |

    **Explicitly Out of Scope:**

    - Pre-training LLMs from scratch (separate workload)
    - Model serving infrastructure (handled by vLLM/TGI)
    - Data labeling and annotation tools
    - Prompt engineering optimization
    - Model compression/distillation (beyond quantization)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Job Start Latency** | < 10 minutes (model load + setup) | Fast iteration for users |
    | **Training Throughput** | 1000+ jobs/month, 50 concurrent jobs | Multi-tenant platform scale |
    | **GPU Memory Efficiency** | Fit 70B models in 8x A100 (80GB) | Maximize model size capacity |
    | **Training Speed** | > 1000 tokens/sec/GPU for 7B models | Minimize training time |
    | **Checkpoint Reliability** | < 1% data loss on failures | Protect expensive training runs |
    | **Model Quality** | Match/exceed full fine-tuning within 5% | Justify parameter-efficient methods |
    | **Availability** | 99.5% uptime (43 hours/year) | Production SLA |
    | **Scalability** | Support up to 128 GPUs per job | Enable large model fine-tuning |
    | **Security** | Multi-tenant isolation, encryption at rest/transit | Protect proprietary models/data |
    | **Cost Efficiency** | < 10% overhead vs raw GPU usage | Competitive pricing |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Fine-tuning Jobs:
    - Total jobs per month: 1,000 jobs
    - Concurrent jobs: 50 jobs
    - Average job duration: 6-48 hours (depends on model size, dataset)
    - Peak submissions: 100 jobs/day

    Model Size Distribution:
    - 7B models (Llama 3 8B, Mistral 7B): 60% of jobs
    - 13B-34B models: 30% of jobs
    - 70B models (Llama 3 70B): 10% of jobs

    Dataset Size:
    - Average training samples: 10K-100K examples
    - Average dataset size: 100 MB - 5 GB (text)
    - Total dataset storage: 10 TB
    - Daily dataset uploads: 50 GB

    GPU Allocation:
    - 7B models: 1-4 GPUs (A100 40GB or 80GB)
    - 13B-34B models: 4-8 GPUs (A100 80GB)
    - 70B models: 8-16 GPUs (A100 80GB)
    - Total GPU cluster: 400 GPUs (mix of A100, H100)

    Training Configuration:
    - LoRA/QLoRA: 90% of jobs (parameter-efficient)
    - Full fine-tuning: 10% of jobs (smaller models)
    - RLHF jobs: 20% of jobs (reward model + PPO)
    - Average training steps: 5K-50K steps
    - Checkpoint frequency: every 1000 steps or 1 hour

    Checkpoints:
    - Checkpoints per job: 10-50 checkpoints
    - LoRA adapter size: 50-500 MB (vs 13GB for full 7B model)
    - Full model checkpoint size: 13GB (7B), 26GB (13B), 140GB (70B)
    - Total checkpoint storage: 50 TB (with 30-day retention)

    Evaluation:
    - Evaluation frequency: every 5000 steps
    - Evaluation dataset size: 1K-10K samples
    - Evaluation time: 5-30 minutes per checkpoint
    ```

    ### Storage Estimates

    ```
    Training Datasets:
    - Active datasets: 500 datasets × 1 GB average = 500 GB
    - Historical datasets (1 year): 500 GB × 12 = 6 TB
    - Preprocessed datasets (tokenized): 6 TB × 1.5 = 9 TB
    - Total with compression (70%): 10.5 TB

    Model Checkpoints:
    - LoRA adapters: 1000 jobs × 20 checkpoints × 200 MB = 4 TB/month
    - Full model checkpoints: 100 jobs × 10 checkpoints × 20 GB = 20 TB/month
    - 30-day retention: (4 TB + 20 TB) × 1 = 24 TB
    - With deduplication (base models shared): 15 TB

    Base Models:
    - Llama 3 8B, 70B: 16 GB + 140 GB = 156 GB
    - Mistral 7B, Mixtral 8x7B: 13 GB + 90 GB = 103 GB
    - Total base models: 500 GB (various quantization formats)

    Model Registry:
    - Final fine-tuned models: 1000 models × 300 MB average = 300 GB/month
    - 1-year retention: 300 GB × 12 = 3.6 TB

    Metadata & Logs:
    - Training logs: 50 GB/month
    - Metrics (loss, perplexity): 100 GB/month
    - Experiment metadata: 10 GB

    Total: 10.5 TB (datasets) + 15 TB (checkpoints) + 500 GB (base models) + 3.6 TB (models) + 160 GB (logs) ≈ 30 TB
    ```

    ### Compute Estimates

    ```
    GPU Compute:
    - Concurrent jobs: 50 jobs
    - Average GPUs per job: 4 GPUs
    - Total GPUs needed: 50 × 4 = 200 GPUs
    - With 80% utilization: 200 / 0.8 = 250 GPUs in cluster
    - GPU types: A100 80GB (primary), H100 (for large models)
    - Cost: 250 GPUs × $2.50/hour = $625/hour = $15,000/day

    Cost per Job:
    - 7B model (4 GPUs × 12 hours): $120
    - 70B model (16 GPUs × 48 hours): $1,920
    - Average: $300/job
    - Monthly compute: 1000 jobs × $300 = $300,000/month

    Network Bandwidth:
    - Dataset download: 50 GB/day
    - Model download: 1000 jobs × 20 GB = 20 TB/month
    - Checkpoint writes: 50 jobs × 5 GB/hour × 24 hours = 6 TB/day
    - AllReduce (distributed training): 100 GB/sec aggregate

    Control Plane:
    - API servers: 10 instances × 4 vCPUs = 40 vCPUs
    - Job scheduler: 5 instances × 8 vCPUs = 40 vCPUs
    - Metadata DB: 3 replicas × 16 vCPUs = 48 vCPUs
    - Total: 128 vCPUs = $128/day
    ```

---

=== "🏗️ Step 2: High-Level Design"

    ## Architecture Diagram

    ```mermaid
    graph TB
        User["👤 User/ML Engineer"]
        WebUI["🖥️ Web UI/CLI/SDK"]
        APIGateway["🚪 API Gateway"]

        subgraph "Control Plane"
            JobController["📋 Job Controller"]
            ResourceScheduler["⚙️ GPU Scheduler<br/>(Kubernetes)"]
            MetricsCollector["📊 Metrics Collector"]
            CheckpointMgr["💾 Checkpoint Manager"]
        end

        subgraph "Data Pipeline"
            DatasetService["📥 Dataset Service<br/>(Upload, Validation)"]
            DataProcessor["🔧 Data Processor<br/>(Tokenization, Format)"]
            DataValidator["✓ Dataset Validator<br/>(Schema, Quality)"]
        end

        subgraph "Training Cluster"
            K8s["☸️ Kubernetes"]

            subgraph "GPU Nodes"
                TrainingPod1["🎮 Training Pod 1<br/>(LoRA Trainer)"]
                TrainingPod2["🎮 Training Pod 2<br/>(RLHF Trainer)"]
                TrainingPodN["🎮 Training Pod N"]
            end

            DistCoordinator["🔗 Distributed Coordinator<br/>(FSDP/DeepSpeed)"]
        end

        subgraph "Fine-tuning Engine"
            LoRAEngine["🧩 LoRA/QLoRA Engine<br/>(Adapter Training)"]
            FSDPEngine["⚡ FSDP Engine<br/>(Sharded Training)"]
            RLHFEngine["🎯 RLHF Engine<br/>(PPO Optimization)"]
        end

        subgraph "Evaluation Service"
            EvalOrchestrator["📊 Evaluation Orchestrator"]
            MetricEngine["📈 Metric Engine<br/>(Perplexity, ROUGE)"]
            HumanEval["👥 Human Eval Integration"]
        end

        subgraph "Model Management"
            ModelRegistry["📚 Model Registry<br/>(Hugging Face Hub)"]
            AdapterStore["🧩 Adapter Store<br/>(LoRA weights)"]
            MergeService["🔀 Model Merger<br/>(Adapter + Base)"]
            DeployService["🚀 Deploy Service<br/>(vLLM, TGI)"]
        end

        subgraph "Storage Layer"
            ObjectStore["☁️ Object Storage<br/>(S3/GCS)"]
            ModelCache["⚡ Model Cache<br/>(NVMe SSD)"]
            MetadataDB["💾 Metadata DB<br/>(PostgreSQL)"]
            MetricsDB["📈 Metrics DB<br/>(Prometheus)"]
        end

        User -->|Submit Job| WebUI
        WebUI -->|REST API| APIGateway
        APIGateway --> JobController

        JobController --> DatasetService
        DatasetService --> DataValidator
        DataValidator --> DataProcessor
        DataProcessor --> ObjectStore

        JobController --> ResourceScheduler
        ResourceScheduler --> K8s
        K8s --> TrainingPod1
        K8s --> TrainingPod2
        K8s --> TrainingPodN

        TrainingPod1 --> LoRAEngine
        TrainingPod2 --> RLHFEngine
        TrainingPodN --> FSDPEngine

        LoRAEngine --> DistCoordinator
        FSDPEngine --> DistCoordinator
        RLHFEngine --> DistCoordinator

        DistCoordinator -->|Save Checkpoints| CheckpointMgr
        CheckpointMgr --> ObjectStore

        TrainingPod1 -->|Metrics| MetricsCollector
        MetricsCollector --> MetricsDB

        JobController --> EvalOrchestrator
        EvalOrchestrator --> MetricEngine
        MetricEngine --> HumanEval

        CheckpointMgr --> ModelRegistry
        ModelRegistry --> AdapterStore
        AdapterStore --> MergeService
        MergeService --> DeployService

        ModelCache -.->|Cache Base Models| ModelRegistry

        JobController --> MetadataDB
        ResourceScheduler --> MetadataDB
    ```

    ---

    ## API Design

    ### Fine-tuning Job Submission API

    ```protobuf
    message FineTuningJobRequest {
        string job_name = 1;
        string user_id = 2;
        BaseModelConfig base_model = 3;
        DatasetConfig dataset = 4;
        TrainingConfig training_config = 5;
        ResourceRequest resources = 6;
        EvaluationConfig eval_config = 7;
        DeploymentConfig deployment = 8;
    }

    message BaseModelConfig {
        string model_id = 1;                // "meta-llama/Llama-3-8B"
        string model_source = 2;            // huggingface, s3, custom
        string quantization = 3;            // none, int8, int4, nf4
        bool use_flash_attention = 4;
    }

    message DatasetConfig {
        string dataset_path = 1;            // s3://bucket/dataset.jsonl
        string format = 2;                  // instruction, chat, completion
        int32 max_seq_length = 3;           // default: 2048
        string prompt_template = 4;         // optional custom template
        float train_test_split = 5;         // default: 0.95
    }

    message TrainingConfig {
        // Method selection
        enum TrainingMethod {
            LORA = 0;
            QLORA = 1;
            FULL_FINETUNING = 2;
            RLHF = 3;
        }
        TrainingMethod method = 1;

        // LoRA/QLoRA parameters
        int32 lora_rank = 2;                // default: 16, range: 8-128
        float lora_alpha = 3;               // default: 32
        float lora_dropout = 4;             // default: 0.05
        repeated string target_modules = 5; // ["q_proj", "v_proj", "k_proj"]

        // Training hyperparameters
        float learning_rate = 6;            // default: 2e-4
        int32 batch_size = 7;               // per device
        int32 gradient_accumulation_steps = 8;
        int32 num_epochs = 9;               // or max_steps
        int32 max_steps = 10;
        float warmup_ratio = 11;            // default: 0.03

        // Optimization
        string optimizer = 12;              // adamw, sgd, adafactor
        float weight_decay = 13;
        float gradient_clipping = 14;       // default: 1.0
        bool use_gradient_checkpointing = 15;

        // Distributed training
        string strategy = 16;               // fsdp, deepspeed, ddp
        int32 fsdp_sharding_strategy = 17;  // FULL_SHARD, SHARD_GRAD_OP
    }

    message ResourceRequest {
        int32 num_gpus = 1;
        string gpu_type = 2;                // a100_40gb, a100_80gb, h100
        int32 num_nodes = 3;                // for multi-node training
        int32 memory_gb = 4;                // host memory
    }

    message EvaluationConfig {
        repeated string metrics = 1;        // perplexity, rouge, bleu, accuracy
        string eval_dataset_path = 2;
        int32 eval_steps = 3;               // evaluate every N steps
        bool enable_human_eval = 4;
    }

    message DeploymentConfig {
        bool auto_deploy = 1;
        string deployment_target = 2;       // vllm, tgi, sagemaker
        int32 inference_gpus = 3;
        int32 max_concurrent_requests = 4;
    }

    message FineTuningJobResponse {
        string job_id = 1;
        string status = 2;                  // QUEUED, RUNNING, COMPLETED, FAILED
        string dashboard_url = 3;
        EstimatedCost estimated_cost = 4;
    }

    message EstimatedCost {
        float compute_usd = 1;
        int32 estimated_hours = 2;
        int32 gpu_hours = 3;
    }
    ```

    ### Checkpoint and Model Management API

    ```protobuf
    message Checkpoint {
        string checkpoint_id = 1;
        string job_id = 2;
        int32 step = 3;
        int64 timestamp = 4;

        // Metrics at checkpoint
        float train_loss = 5;
        float eval_loss = 6;
        float perplexity = 7;
        map<string, float> custom_metrics = 8;

        // Storage
        string checkpoint_path = 9;         // s3://bucket/checkpoints/...
        int64 checkpoint_size_bytes = 10;
        bool is_best = 11;
    }

    message ModelArtifact {
        string model_id = 1;
        string base_model_id = 2;
        string adapter_path = 3;            // LoRA adapter weights
        string merged_model_path = 4;       // optional: full merged model

        // Training info
        string training_job_id = 5;
        map<string, string> training_config = 6;
        map<string, float> final_metrics = 7;

        // Deployment
        bool is_deployed = 8;
        string deployment_endpoint = 9;
        int64 created_at = 10;
    }

    message MergeAdapterRequest {
        string base_model_id = 1;
        string adapter_path = 2;
        string output_path = 3;
        string quantization = 4;            // optional: quantize merged model
    }
    ```

    ---

    ## Database Schema

    ### Fine-tuning Jobs Table

    ```sql
    CREATE TABLE finetuning_jobs (
        job_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        job_name VARCHAR(255),

        -- Model configuration
        base_model_id VARCHAR(255),         -- meta-llama/Llama-3-8B
        base_model_size VARCHAR(50),        -- 7B, 13B, 70B
        quantization VARCHAR(50),           -- none, int8, int4, nf4

        -- Training method
        training_method VARCHAR(50),        -- lora, qlora, full, rlhf
        lora_rank INT,
        lora_alpha FLOAT,
        target_modules JSONB,

        -- Dataset
        dataset_path VARCHAR(500),
        dataset_format VARCHAR(50),         -- instruction, chat, completion
        num_training_samples INT,
        max_seq_length INT,

        -- Hyperparameters
        learning_rate FLOAT,
        batch_size INT,
        num_epochs INT,
        max_steps INT,

        -- Resources
        num_gpus INT,
        gpu_type VARCHAR(50),
        num_nodes INT,
        distributed_strategy VARCHAR(50),   -- fsdp, deepspeed, ddp

        -- Status
        status VARCHAR(50),                 -- QUEUED, RUNNING, EVALUATING, COMPLETED, FAILED
        created_at TIMESTAMP,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,

        -- Results
        best_checkpoint_id VARCHAR(255),
        final_train_loss FLOAT,
        final_eval_loss FLOAT,
        final_perplexity FLOAT,

        -- Cost
        gpu_hours DECIMAL(10, 2),
        total_cost_usd DECIMAL(10, 2),

        INDEX idx_user_created (user_id, created_at),
        INDEX idx_status (status),
        INDEX idx_base_model (base_model_id)
    );

    CREATE TABLE checkpoints (
        checkpoint_id VARCHAR(255) PRIMARY KEY,
        job_id VARCHAR(255) NOT NULL,
        step INT NOT NULL,
        epoch INT,

        -- Metrics
        train_loss FLOAT,
        eval_loss FLOAT,
        perplexity FLOAT,
        learning_rate FLOAT,
        custom_metrics JSONB,

        -- Storage
        checkpoint_path VARCHAR(500),       -- s3://bucket/checkpoints/...
        checkpoint_size_bytes BIGINT,
        checkpoint_type VARCHAR(50),        -- full_model, lora_adapter, optimizer_state
        is_best BOOLEAN,

        created_at TIMESTAMP,

        FOREIGN KEY (job_id) REFERENCES finetuning_jobs(job_id),
        INDEX idx_job_step (job_id, step),
        INDEX idx_best (job_id, is_best)
    );

    CREATE TABLE model_registry (
        model_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        model_name VARCHAR(255),

        -- Base model
        base_model_id VARCHAR(255),
        base_model_size VARCHAR(50),

        -- Adapter info (for LoRA/QLoRA)
        adapter_path VARCHAR(500),
        adapter_size_bytes BIGINT,
        lora_rank INT,

        -- Merged model (optional)
        merged_model_path VARCHAR(500),
        merged_model_size_bytes BIGINT,

        -- Training provenance
        training_job_id VARCHAR(255),
        training_method VARCHAR(50),
        training_config JSONB,

        -- Evaluation
        eval_metrics JSONB,

        -- Deployment
        is_deployed BOOLEAN,
        deployment_endpoint VARCHAR(500),
        deployment_target VARCHAR(50),      -- vllm, tgi, sagemaker

        created_at TIMESTAMP,

        FOREIGN KEY (training_job_id) REFERENCES finetuning_jobs(job_id),
        INDEX idx_user (user_id),
        INDEX idx_base_model (base_model_id)
    );

    CREATE TABLE datasets (
        dataset_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        dataset_name VARCHAR(255),

        -- Storage
        dataset_path VARCHAR(500),
        dataset_format VARCHAR(50),         -- instruction, chat, completion

        -- Statistics
        num_samples INT,
        avg_sample_length INT,
        max_sample_length INT,
        dataset_size_bytes BIGINT,

        -- Validation
        validation_status VARCHAR(50),      -- VALID, INVALID, PENDING
        validation_errors JSONB,

        -- Processing
        tokenized_path VARCHAR(500),
        tokenizer_id VARCHAR(255),

        created_at TIMESTAMP,

        INDEX idx_user (user_id)
    );
    ```

---

=== "🔍 Step 3: Deep Dive"

    ## 3.1 LoRA/QLoRA Implementation

    ### LoRA (Low-Rank Adaptation)

    **Concept**: Instead of fine-tuning all parameters, inject trainable low-rank matrices into each transformer layer. Original weights frozen, only adapters trained.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALayer(nn.Module):
        """
        LoRA adapter layer: W = W_0 + BA
        where B is (d × r), A is (r × k), r << min(d, k)
        """
        def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.05):
            super().__init__()
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            # Original linear layer (frozen)
            self.linear = nn.Linear(in_features, out_features, bias=False)
            self.linear.weight.requires_grad = False

            # LoRA low-rank matrices (trainable)
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.dropout = nn.Dropout(dropout)

            # Initialize
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, x):
            """Forward pass: original + low-rank adaptation"""
            # Original transformation
            result = self.linear(x)

            # LoRA adaptation: x @ A^T @ B^T
            lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T

            # Scaled addition
            return result + lora_output * self.scaling

    def inject_lora_into_model(model, target_modules=["q_proj", "v_proj"], rank=16):
        """
        Replace target modules with LoRA layers in transformer model
        """
        for name, module in model.named_modules():
            # Check if module name matches target
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA layer
                    parent_module = get_parent_module(model, name)
                    module_name = name.split('.')[-1]

                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=rank
                    )
                    # Copy original weights
                    lora_layer.linear.weight.data = module.weight.data.clone()

                    setattr(parent_module, module_name, lora_layer)

        return model

    # Usage with Hugging Face model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

    # Inject LoRA adapters
    model = inject_lora_into_model(
        model,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        rank=16
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    # Output: Trainable params: 4,194,304 (0.05% of 8B parameters)
    ```

    ### QLoRA (Quantized LoRA)

    **Concept**: Combine LoRA with 4-bit quantization (NF4) of base model to fit 70B models in limited GPU memory.

    ```python
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import bitsandbytes as bnb

    class QLoRAConfig:
        """Configuration for QLoRA training"""
        def __init__(
            self,
            rank=16,
            alpha=32,
            dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            quantization="nf4",  # nf4 (4-bit NormalFloat) or int4
            double_quant=True,   # nested quantization
            compute_dtype=torch.bfloat16
        ):
            self.rank = rank
            self.alpha = alpha
            self.dropout = dropout
            self.target_modules = target_modules
            self.quantization = quantization
            self.double_quant = double_quant
            self.compute_dtype = compute_dtype

    def load_model_with_qlora(model_id, qlora_config):
        """
        Load model in 4-bit quantized format for QLoRA training
        """
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora_config.quantization,  # nf4 or fp4
            bnb_4bit_use_double_quant=qlora_config.double_quant,
            bnb_4bit_compute_dtype=qlora_config.compute_dtype
        )

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",  # automatic device placement
            torch_dtype=torch.bfloat16
        )

        # Prepare for training (convert some layers to trainable)
        model = prepare_model_for_kbit_training(model)

        return model

    def prepare_model_for_kbit_training(model):
        """
        Prepare quantized model for training:
        - Enable gradient checkpointing
        - Cast LayerNorm to fp32 for stability
        """
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Cast normalization layers to fp32
        for name, module in model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                module = module.to(torch.float32)

        # Enable input gradients
        model.enable_input_require_grads()

        return model

    # Usage: Load 70B model in 4-bit on 8x A100 (80GB)
    qlora_config = QLoRAConfig(rank=64, alpha=128)

    model = load_model_with_qlora(
        "meta-llama/Llama-3-70B",
        qlora_config
    )

    # Add LoRA adapters using PEFT library
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=qlora_config.rank,
        lora_alpha=qlora_config.alpha,
        lora_dropout=qlora_config.dropout,
        target_modules=qlora_config.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Memory usage: ~40GB per GPU for 70B model (vs 280GB without quantization)
    print(f"Model loaded on {torch.cuda.device_count()} GPUs")
    ```

    ---

    ## 3.2 Distributed Training with FSDP

    ### FSDP (Fully Sharded Data Parallel)

    **Concept**: Shard model parameters, gradients, and optimizer states across GPUs. Each GPU holds only a fraction of the model.

    ```python
    import torch
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    def setup_fsdp_training(model, rank, world_size):
        """
        Configure FSDP for distributed training
        """
        # Initialize distributed environment
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(rank)

        # Mixed precision configuration
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Auto-wrap policy: wrap each transformer layer
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={LlamaDecoderLayer}
        )

        # Wrap model with FSDP
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # Shard params + grads + optimizer
            mixed_precision=mixed_precision_policy,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Overlap communication
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,  # Memory optimization
        )

        return model

    def train_with_fsdp(model, train_loader, optimizer, num_epochs, rank):
        """
        Training loop with FSDP
        """
        model.train()

        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()

                # Forward pass
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                # Backward pass (FSDP handles gradient synchronization)
                loss.backward()

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                if rank == 0 and batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")

        return model

    # Launch distributed training
    def main():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Load model
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

        # Setup FSDP
        model = setup_fsdp_training(model, rank, world_size)

        # Training
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        train_with_fsdp(model, train_loader, optimizer, num_epochs=3, rank=rank)

        # Save model (only rank 0)
        if rank == 0:
            torch.save(model.state_dict(), "model.pt")

    # Launch with torchrun:
    # torchrun --nproc_per_node=8 train_fsdp.py
    ```

    ### DeepSpeed Integration

    ```python
    import deepspeed
    from deepspeed.ops.adam import FusedAdam

    def train_with_deepspeed(model, train_loader, deepspeed_config):
        """
        Training with DeepSpeed ZeRO-3 for extreme memory efficiency
        """
        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=deepspeed_config,
            model_parameters=model.parameters()
        )

        for epoch in range(num_epochs):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(model_engine.device)
                labels = batch["labels"].to(model_engine.device)

                # Forward pass
                outputs = model_engine(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                # Backward pass
                model_engine.backward(loss)

                # Optimizer step
                model_engine.step()

        return model_engine

    # DeepSpeed configuration (ds_config.json)
    deepspeed_config = {
        "train_batch_size": 128,
        "gradient_accumulation_steps": 4,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
        },
        "zero_optimization": {
            "stage": 3,  # ZeRO-3: shard params + grads + optimizer
            "offload_optimizer": {
                "device": "cpu",  # Offload optimizer to CPU RAM
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 2e-5,
                "warmup_num_steps": 1000,
                "total_num_steps": 10000
            }
        }
    }
    ```

    ---

    ## 3.3 Instruction Tuning Implementation

    ### Dataset Preparation

    ```python
    import json
    from typing import List, Dict
    from transformers import AutoTokenizer

    class InstructionDataset:
        """
        Format datasets for instruction tuning
        """
        def __init__(self, tokenizer, max_length=2048):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def format_alpaca_style(self, instruction, input_text, output):
            """
            Format: Alpaca/Dolly style instruction-response
            """
            if input_text:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

            return prompt

        def format_chat_style(self, messages: List[Dict[str, str]]):
            """
            Format: ChatML/OpenAI chat format
            """
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            return formatted

        def tokenize_and_prepare(self, text):
            """
            Tokenize with proper masking for loss computation
            """
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Create labels (mask prompt, only compute loss on response)
            labels = input_ids.clone()

            # Find response start (after "### Response:" or last <|im_start|>assistant)
            response_token = self.tokenizer.encode("### Response:", add_special_tokens=False)[0]
            response_start = (input_ids == response_token).nonzero(as_tuple=True)[1]

            if len(response_start) > 0:
                # Mask everything before response
                labels[:, :response_start[0]] = -100  # -100 is ignored in loss

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        def load_and_prepare_dataset(self, dataset_path):
            """
            Load dataset and prepare for training
            """
            with open(dataset_path, 'r') as f:
                raw_data = [json.loads(line) for line in f]

            prepared_data = []
            for example in raw_data:
                # Format based on dataset structure
                if "instruction" in example:
                    # Alpaca style
                    text = self.format_alpaca_style(
                        example["instruction"],
                        example.get("input", ""),
                        example["output"]
                    )
                elif "messages" in example:
                    # Chat style
                    text = self.format_chat_style(example["messages"])

                # Tokenize
                prepared = self.tokenize_and_prepare(text)
                prepared_data.append(prepared)

            return prepared_data

    # Usage
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
    dataset = InstructionDataset(tokenizer, max_length=2048)

    # Load and prepare dataset
    train_data = dataset.load_and_prepare_dataset("train.jsonl")
    print(f"Prepared {len(train_data)} training examples")
    ```

    ### Training Loop

    ```python
    from torch.utils.data import DataLoader, Dataset

    class InstructionDataLoader(Dataset):
        def __init__(self, prepared_data):
            self.data = prepared_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {
                "input_ids": self.data[idx]["input_ids"].squeeze(0),
                "attention_mask": self.data[idx]["attention_mask"].squeeze(0),
                "labels": self.data[idx]["labels"].squeeze(0)
            }

    def train_instruction_tuning(model, train_data, eval_data, config):
        """
        Fine-tune model on instruction dataset
        """
        # Prepare dataloaders
        train_dataset = InstructionDataLoader(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        num_training_steps = len(train_loader) * config.num_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps
        )

        # Training loop
        model.train()
        global_step = 0

        for epoch in range(config.num_epochs):
            epoch_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                # Move to GPU
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % 100 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}")

                # Checkpoint saving
                if global_step % 1000 == 0:
                    save_checkpoint(model, optimizer, global_step)

                # Evaluation
                if global_step % 5000 == 0:
                    eval_metrics = evaluate(model, eval_data)
                    print(f"Eval metrics: {eval_metrics}")

        return model
    ```

    ---

    ## 3.4 RLHF (Reinforcement Learning from Human Feedback)

    ### Reward Model Training

    ```python
    class RewardModel(nn.Module):
        """
        Reward model: learns to predict human preferences
        Input: (prompt, response) pair
        Output: scalar reward
        """
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            # Get last hidden state
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]

            # Pool last token (response end)
            sequence_lengths = attention_mask.sum(dim=1) - 1
            pooled = last_hidden[torch.arange(len(last_hidden)), sequence_lengths]

            # Predict reward
            reward = self.reward_head(pooled)
            return reward

    def train_reward_model(model, preference_dataset):
        """
        Train reward model on human preference comparisons
        Dataset format: (prompt, chosen_response, rejected_response)
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        for batch in preference_dataset:
            # Tokenize chosen and rejected responses
            chosen_ids = batch["chosen_input_ids"].cuda()
            chosen_mask = batch["chosen_attention_mask"].cuda()

            rejected_ids = batch["rejected_input_ids"].cuda()
            rejected_mask = batch["rejected_attention_mask"].cuda()

            # Forward pass
            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)

            # Loss: maximize margin between chosen and rejected
            # Loss = -log(sigmoid(chosen_reward - rejected_reward))
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return model
    ```

    ### PPO (Proximal Policy Optimization)

    ```python
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

    class RLHFTrainer:
        """
        RLHF training with PPO
        """
        def __init__(self, policy_model, reward_model, config):
            self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                policy_model
            )
            self.reward_model = reward_model
            self.config = config

            # PPO configuration
            ppo_config = PPOConfig(
                batch_size=config.batch_size,
                learning_rate=1e-5,
                ppo_epochs=4,
                mini_batch_size=4,
                gradient_accumulation_steps=4,
                optimize_cuda_cache=True
            )

            self.ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=self.policy_model,
                ref_model=None,  # optional reference model
                tokenizer=tokenizer
            )

        def generate_and_score(self, prompts):
            """
            Generate responses and score with reward model
            """
            # Generate responses
            responses = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

                output = self.policy_model.generate(
                    input_ids,
                    max_length=512,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7
                )

                response = tokenizer.decode(output[0], skip_special_tokens=True)
                responses.append(response)

            # Score with reward model
            rewards = []
            for prompt, response in zip(prompts, responses):
                full_text = prompt + response
                input_ids = tokenizer.encode(full_text, return_tensors="pt").cuda()
                attention_mask = torch.ones_like(input_ids)

                reward = self.reward_model(input_ids, attention_mask)
                rewards.append(reward.item())

            return responses, rewards

        def train_step(self, prompts):
            """
            PPO training step
            """
            # Generate responses and get rewards
            responses, rewards = self.generate_and_score(prompts)

            # Convert to tensors
            query_tensors = [tokenizer.encode(p, return_tensors="pt")[0] for p in prompts]
            response_tensors = [tokenizer.encode(r, return_tensors="pt")[0] for r in responses]
            reward_tensors = [torch.tensor(r) for r in rewards]

            # PPO step
            stats = self.ppo_trainer.step(
                query_tensors,
                response_tensors,
                reward_tensors
            )

            return stats

        def train(self, prompt_dataset, num_epochs=3):
            """
            Full RLHF training loop
            """
            for epoch in range(num_epochs):
                for batch_prompts in prompt_dataset:
                    stats = self.train_step(batch_prompts)

                    print(f"Epoch {epoch}, "
                          f"Mean reward: {stats['ppo/mean_scores']:.4f}, "
                          f"Policy loss: {stats['ppo/policy_loss']:.4f}")

            return self.policy_model

    # Usage
    policy_model = "meta-llama/Llama-3-8B-Instruct"
    reward_model = load_reward_model("reward_model_checkpoint")

    rlhf_trainer = RLHFTrainer(policy_model, reward_model, config)
    trained_model = rlhf_trainer.train(prompt_dataset, num_epochs=3)
    ```

    ---

    ## 3.5 Evaluation Metrics

    ```python
    import numpy as np
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu

    class EvaluationMetrics:
        """
        Comprehensive evaluation metrics for fine-tuned LLMs
        """
        def __init__(self, model, tokenizer, eval_dataset):
            self.model = model
            self.tokenizer = tokenizer
            self.eval_dataset = eval_dataset

        def compute_perplexity(self):
            """
            Compute perplexity on evaluation dataset
            """
            self.model.eval()
            total_loss = 0
            total_tokens = 0

            with torch.no_grad():
                for batch in self.eval_dataset:
                    input_ids = batch["input_ids"].cuda()
                    attention_mask = batch["attention_mask"].cuda()
                    labels = batch["labels"].cuda()

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    num_tokens = (labels != -100).sum()

                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)

            return perplexity

        def compute_rouge(self, predictions, references):
            """
            Compute ROUGE scores (for summarization/generation)
            """
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

            scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                scores['rouge1'].append(score['rouge1'].fmeasure)
                scores['rouge2'].append(score['rouge2'].fmeasure)
                scores['rougeL'].append(score['rougeL'].fmeasure)

            return {
                'rouge1': np.mean(scores['rouge1']),
                'rouge2': np.mean(scores['rouge2']),
                'rougeL': np.mean(scores['rougeL'])
            }

        def compute_task_accuracy(self, task_type):
            """
            Task-specific accuracy (classification, QA, etc.)
            """
            correct = 0
            total = 0

            self.model.eval()
            with torch.no_grad():
                for example in self.eval_dataset:
                    # Generate response
                    prompt = example["prompt"]
                    ground_truth = example["answer"]

                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
                    output = self.model.generate(
                        input_ids,
                        max_length=512,
                        do_sample=False  # greedy for consistency
                    )

                    prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)

                    # Extract answer (task-specific parsing)
                    if task_type == "multiple_choice":
                        pred_answer = self.extract_choice(prediction)
                        if pred_answer == ground_truth:
                            correct += 1
                    elif task_type == "binary":
                        pred_binary = self.extract_binary(prediction)
                        if pred_binary == ground_truth:
                            correct += 1

                    total += 1

            accuracy = correct / total if total > 0 else 0
            return accuracy

        def comprehensive_evaluation(self):
            """
            Run all evaluation metrics
            """
            metrics = {}

            # Perplexity
            print("Computing perplexity...")
            metrics['perplexity'] = self.compute_perplexity()

            # Generate predictions for ROUGE
            print("Computing ROUGE scores...")
            predictions, references = self.generate_predictions()
            metrics.update(self.compute_rouge(predictions, references))

            # Task accuracy
            print("Computing task accuracy...")
            metrics['accuracy'] = self.compute_task_accuracy(task_type="multiple_choice")

            return metrics

    # Usage
    evaluator = EvaluationMetrics(model, tokenizer, eval_dataset)
    results = evaluator.comprehensive_evaluation()

    print(f"""
    Evaluation Results:
    - Perplexity: {results['perplexity']:.2f}
    - ROUGE-1: {results['rouge1']:.4f}
    - ROUGE-2: {results['rouge2']:.4f}
    - ROUGE-L: {results['rougeL']:.4f}
    - Accuracy: {results['accuracy']:.4f}
    """)
    ```

---

=== "📈 Step 4: Scale & Optimize"

    ## 4.1 Memory Optimization

    ### Gradient Checkpointing

    ```python
    def enable_gradient_checkpointing(model):
        """
        Trade compute for memory: recompute activations during backward pass
        Memory savings: ~50%, Speed penalty: ~20%
        """
        model.gradient_checkpointing_enable()

        # For custom models
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

        return model

    # Memory usage example (7B model):
    # Without gradient checkpointing: ~40 GB
    # With gradient checkpointing: ~20 GB
    ```

    ### Flash Attention

    ```python
    from flash_attn import flash_attn_qkvpacked_func

    class FlashAttentionLayer(nn.Module):
        """
        Flash Attention: memory-efficient attention computation
        Memory: O(N) instead of O(N^2)
        Speed: 2-4x faster than standard attention
        """
        def __init__(self, hidden_size, num_heads):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads

            self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)

        def forward(self, hidden_states):
            batch_size, seq_len, _ = hidden_states.shape

            # Project to Q, K, V
            qkv = self.qkv_proj(hidden_states)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

            # Flash attention
            output = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True  # for causal LM
            )

            output = output.reshape(batch_size, seq_len, self.hidden_size)
            return output

    # Enable flash attention in Hugging Face models
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8B",
        attn_implementation="flash_attention_2",  # requires flash-attn package
        torch_dtype=torch.bfloat16
    )
    ```

    ### Mixed Precision Training

    ```python
    from torch.cuda.amp import autocast, GradScaler

    def train_with_mixed_precision(model, train_loader, optimizer):
        """
        BF16/FP16 mixed precision: 2x faster, 50% less memory
        """
        scaler = GradScaler()  # for FP16 only (not needed for BF16)

        for batch in train_loader:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            # Automatic mixed precision
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # BF16 vs FP16:
    # - BF16 (bfloat16): Better numerical stability, no gradient scaling needed
    # - FP16 (float16): Requires gradient scaling, slightly faster on older GPUs
    # Recommendation: Use BF16 on A100/H100
    ```

    ---

    ## 4.2 Multi-Node Training

    ### Distributed Setup

    ```python
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    def setup_multi_node_training(rank, world_size, master_addr, master_port):
        """
        Setup for multi-node distributed training
        """
        # Set environment variables
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        # Set device
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        return local_rank

    def train_multi_node(model, train_loader, config):
        """
        Multi-node training loop
        """
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = setup_multi_node_training(
            rank, world_size,
            config.master_addr,
            config.master_port
        )

        # Move model to GPU and wrap with DDP
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])

        # Training loop
        for epoch in range(config.num_epochs):
            train_loader.sampler.set_epoch(epoch)  # for distributed sampler

            for batch in train_loader:
                input_ids = batch["input_ids"].to(local_rank)
                labels = batch["labels"].to(local_rank)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Cleanup
        dist.destroy_process_group()

    # Launch with SLURM:
    # srun --nodes=4 --gpus-per-node=8 python train_multi_node.py

    # Or with torchrun:
    # torchrun --nnodes=4 --nproc_per_node=8 \
    #          --master_addr=node0 --master_port=29500 \
    #          train_multi_node.py
    ```

    ---

    ## 4.3 Cost Optimization

    ### Spot Instance Management

    ```python
    class SpotInstanceHandler:
        """
        Handle spot instance interruptions gracefully
        """
        def __init__(self, checkpoint_manager):
            self.checkpoint_manager = checkpoint_manager
            self.last_checkpoint_time = time.time()
            self.checkpoint_interval = 600  # 10 minutes

        def check_spot_interruption(self):
            """
            Check for spot instance termination notice
            AWS: 2-minute warning via metadata API
            """
            try:
                response = requests.get(
                    "http://169.254.169.254/latest/meta-data/spot/instance-action",
                    timeout=1
                )
                if response.status_code == 200:
                    return True  # Interruption imminent
            except:
                pass
            return False

        def train_with_spot_protection(self, model, train_loader, optimizer, global_step):
            """
            Training loop with spot interruption handling
            """
            for batch in train_loader:
                # Check for interruption
                if self.check_spot_interruption():
                    print("Spot interruption detected! Saving checkpoint...")
                    self.checkpoint_manager.save_checkpoint(
                        model, optimizer, global_step, is_emergency=True
                    )
                    sys.exit(0)  # Graceful exit

                # Regular checkpoint (every 10 minutes)
                if time.time() - self.last_checkpoint_time > self.checkpoint_interval:
                    self.checkpoint_manager.save_checkpoint(
                        model, optimizer, global_step
                    )
                    self.last_checkpoint_time = time.time()

                # Training step
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

            return global_step

    # Cost savings:
    # On-demand A100: $3.00/hour
    # Spot A100: $0.90/hour (70% savings)
    # With checkpointing overhead: ~5% time penalty, ~65% cost savings
    ```

    ### Dynamic Batch Size Scaling

    ```python
    def find_optimal_batch_size(model, sample_batch, max_memory_gb=80):
        """
        Automatically find largest batch size that fits in GPU memory
        """
        batch_size = 1
        max_batch_size = 128

        while batch_size <= max_batch_size:
            try:
                # Try current batch size
                test_batch = {
                    k: v[:batch_size].cuda() for k, v in sample_batch.items()
                }

                outputs = model(**test_batch)
                loss = outputs.loss
                loss.backward()

                # Check memory usage
                memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB
                if memory_used > max_memory_gb * 0.9:  # 90% threshold
                    break

                # Clear
                del outputs, loss
                torch.cuda.empty_cache()

                # Try larger batch size
                batch_size *= 2

            except torch.cuda.OutOfMemoryError:
                # OOM, use previous batch size
                batch_size //= 2
                break

        torch.cuda.empty_cache()
        return max(1, batch_size)

    # Usage
    optimal_batch_size = find_optimal_batch_size(model, sample_batch)
    print(f"Optimal batch size: {optimal_batch_size}")
    ```

    ---

    ## 4.4 Model Merging and Deployment

    ### LoRA Adapter Merging

    ```python
    def merge_lora_adapter(base_model_path, adapter_path, output_path):
        """
        Merge LoRA adapter weights back into base model
        W_merged = W_base + (B @ A) * scaling
        """
        from peft import PeftModel

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Load model with adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # Merge adapter into base weights
        merged_model = model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        print(f"Merged model saved to {output_path}")

        # Memory: merged model = base model size (no adapter overhead)
        return merged_model

    # Deployment size:
    # - LoRA adapter only: 50-500 MB
    # - Merged model: 13 GB (7B model)
    # Trade-off: Adapter is smaller but requires base model at inference
    ```

    ### vLLM Deployment

    ```python
    from vllm import LLM, SamplingParams

    def deploy_with_vllm(model_path, tensor_parallel_size=4):
        """
        Deploy fine-tuned model with vLLM for fast inference
        vLLM: 10-20x faster than vanilla Hugging Face
        """
        # Initialize vLLM engine
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,  # multi-GPU
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.9,
            # Performance optimizations
            enable_prefix_caching=True,  # KV cache reuse
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
            max_num_seqs=256
        )

        return llm

    def serve_requests(llm):
        """
        Serve inference requests
        """
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )

        # Batch inference (efficient)
        prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            # ... more prompts
        ]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Response: {output.outputs[0].text}")
            print("---")

    # Throughput comparison (7B model on 4x A100):
    # - Hugging Face: ~50 tokens/sec
    # - vLLM: ~1000 tokens/sec (20x faster)
    ```

    ---

    ## 4.5 Monitoring and Observability

    ```python
    import wandb
    from prometheus_client import Counter, Histogram

    class TrainingMonitor:
        """
        Comprehensive monitoring for fine-tuning jobs
        """
        def __init__(self, job_id, project_name):
            self.job_id = job_id

            # Initialize Weights & Biases
            wandb.init(
                project=project_name,
                name=job_id,
                config={
                    "model": "llama-3-8b",
                    "method": "qlora",
                    "lora_rank": 16
                }
            )

            # Prometheus metrics
            self.loss_metric = Histogram(
                'training_loss',
                'Training loss per step',
                ['job_id']
            )
            self.gpu_util_metric = Histogram(
                'gpu_utilization',
                'GPU utilization percentage',
                ['job_id', 'gpu_id']
            )

        def log_training_step(self, step, metrics):
            """
            Log metrics for each training step
            """
            # WandB logging
            wandb.log({
                "step": step,
                "loss": metrics["loss"],
                "learning_rate": metrics["lr"],
                "perplexity": np.exp(metrics["loss"]),
                "gpu_memory_allocated": metrics.get("gpu_memory", 0),
                "tokens_per_second": metrics.get("throughput", 0)
            }, step=step)

            # Prometheus metrics
            self.loss_metric.labels(job_id=self.job_id).observe(metrics["loss"])

        def log_gpu_metrics(self):
            """
            Monitor GPU utilization
            """
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                self.gpu_util_metric.labels(
                    job_id=self.job_id,
                    gpu_id=i
                ).observe(util.gpu)

                wandb.log({
                    f"gpu_{i}_utilization": util.gpu,
                    f"gpu_{i}_memory": util.memory
                })

        def log_checkpoint(self, step, checkpoint_path, metrics):
            """
            Log checkpoint creation
            """
            wandb.log({
                "checkpoint_step": step,
                "checkpoint_path": checkpoint_path,
                "checkpoint_metrics": metrics
            })

            # Save model artifact to WandB
            artifact = wandb.Artifact(
                f"checkpoint-{step}",
                type="model",
                metadata=metrics
            )
            artifact.add_reference(checkpoint_path)
            wandb.log_artifact(artifact)

    # Usage
    monitor = TrainingMonitor(job_id="job-123", project_name="llm-finetuning")

    for step, batch in enumerate(train_loader):
        # Training step
        outputs = model(**batch)
        loss = outputs.loss

        # Log metrics
        monitor.log_training_step(step, {
            "loss": loss.item(),
            "lr": optimizer.param_groups[0]['lr'],
            "gpu_memory": torch.cuda.max_memory_allocated() / 1e9
        })

        # Log GPU metrics (every 100 steps)
        if step % 100 == 0:
            monitor.log_gpu_metrics()
    ```

---

=== "🎓 Step 5: Interview Tips & Common Questions"

    ## Question 1: How does LoRA reduce memory and compute requirements?

    **Answer:**

    LoRA injects trainable low-rank matrices (rank r) into frozen transformer layers:
    - **Parameters**: Full fine-tuning = 7B params, LoRA = 4M params (0.06% for r=16)
    - **Memory**: Only store gradients for 4M params vs 7B params → 99% memory savings
    - **Training speed**: Fewer parameters to update → 1.5-2x faster
    - **Quality**: Matches full fine-tuning performance within 2-3% on most tasks

    **Trade-off**: LoRA works best for adaptation tasks (instruction tuning). Less effective for teaching completely new knowledge (use larger ranks or full fine-tuning).

    ```
    Memory comparison (7B model, single GPU):
    - Full fine-tuning: 28 GB (model) + 28 GB (gradients) + 56 GB (optimizer) = 112 GB
    - LoRA: 14 GB (frozen model) + 0.1 GB (adapter params+grads+optimizer) = 14 GB
    → Fits on 1x A100 (40GB) instead of requiring 4x A100
    ```

    ---

    ## Question 2: Explain FSDP vs DeepSpeed ZeRO for 70B model training

    **Answer:**

    Both shard model states across GPUs, but differ in implementation:

    | Feature | FSDP | DeepSpeed ZeRO-3 |
    |---------|------|------------------|
    | **Sharding** | Params + grads + optimizer | Params + grads + optimizer |
    | **Memory savings** | 8x reduction (8 GPUs) | 8x reduction + CPU offload |
    | **Ease of use** | Native PyTorch, simpler | Requires config file |
    | **Performance** | Better for PyTorch models | Better for large models (>70B) |
    | **CPU offload** | Limited | Full support |

    **Recommendation**:
    - 7B-13B models: Use FSDP (simpler, native PyTorch)
    - 70B models: Use DeepSpeed ZeRO-3 (better CPU offload, mature tooling)

    **Memory breakdown (70B model on 8x A100 80GB)**:
    ```
    Without sharding: 140 GB (model) → requires 2x A100 per GPU → impossible
    With FSDP/ZeRO-3: 140 GB / 8 GPUs = 17.5 GB per GPU → fits easily
    ```

    ---

    ## Question 3: How do you evaluate fine-tuned LLMs?

    **Answer:**

    Multi-faceted evaluation is critical:

    **1. Perplexity** (automatic, efficient)
    - Measures how "surprised" model is by test data
    - Lower is better
    - Fast, but doesn't capture generation quality

    **2. Task-specific metrics** (automatic)
    - Classification: Accuracy, F1
    - Summarization: ROUGE, BLEU
    - QA: Exact match, F1

    **3. Human evaluation** (gold standard, expensive)
    - Helpfulness, harmlessness, honesty
    - Win rate vs baseline
    - Cost: $1-5 per comparison

    **4. LLM-as-judge** (scalable approximation)
    - Use GPT-4 to rate responses
    - 80-90% agreement with human raters
    - Cost: $0.01 per evaluation

    **Best practice**: Use perplexity for quick iteration, then run human eval on top 3 checkpoints.

    ---

    ## Question 4: What's the difference between instruction tuning and RLHF?

    **Answer:**

    **Instruction Tuning** (Supervised Fine-tuning, SFT):
    - Train on (instruction, response) pairs
    - Simple cross-entropy loss
    - Teaches model to follow instructions
    - Data: 10K-100K examples
    - Cost: $100-500 per job

    **RLHF** (Reinforcement Learning from Human Feedback):
    - Step 1: Train reward model on human preferences
    - Step 2: Optimize policy with PPO to maximize reward
    - Aligns model with human preferences (helpful, harmless)
    - Data: 10K-100K preference pairs
    - Cost: $1,000-5,000 per job (more compute-intensive)

    **Typical pipeline**: Base model → SFT → RLHF
    - SFT: Teach format and basic behavior
    - RLHF: Refine quality and alignment

    ---

    ## Question 5: How do you handle training failures and spot interruptions?

    **Answer:**

    **Checkpoint Strategy**:
    1. **Frequent checkpoints**: Every 1000 steps or 10 minutes
    2. **Spot monitoring**: Check AWS metadata API every 30 seconds
    3. **Emergency checkpoint**: On interruption notice (2-minute warning)
    4. **Distributed checkpointing**: Only rank 0 saves to avoid corruption

    **Recovery**:
    1. Detect last valid checkpoint from metadata DB
    2. Resume training from checkpoint (model + optimizer state)
    3. Adjust learning rate scheduler (account for skipped steps)

    ```python
    # Resume from checkpoint
    checkpoint = load_checkpoint(last_checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step']

    # Continue training
    for step in range(start_step, max_steps):
        # Training continues seamlessly
    ```

    **Cost impact**: 5% time overhead from checkpointing, but enables 70% cost savings with spot instances.

    ---

    ## Question 6: How do you scale to 1000 fine-tuning jobs per month?

    **Answer:**

    **Resource Management**:
    1. **GPU pooling**: Kubernetes with GPU scheduling
    2. **Job queuing**: Priority queue (paid > free tier)
    3. **Auto-scaling**: Add nodes during peak times
    4. **Multi-tenancy**: Isolate jobs with namespaces

    **Cost Optimization**:
    1. **Spot instances**: 70% savings for non-critical jobs
    2. **LoRA preference**: 90% of jobs use LoRA (14 GB vs 112 GB)
    3. **Batch evaluation**: Share eval service across jobs
    4. **Model caching**: Cache base models on NVMe (avoid repeated downloads)

    **Capacity planning**:
    ```
    1000 jobs/month = ~35 jobs/day
    Average duration: 12 hours
    Concurrent jobs: 35 × (12 / 24) = 18 jobs
    Average GPUs per job: 4 GPUs
    Total GPUs: 18 × 4 = 72 GPUs
    With 80% utilization: 90 GPUs needed

    Cost: 90 GPUs × $2.50/hour × 720 hours/month = $162,000/month
    With spot: $162,000 × 0.3 = $48,600/month
    ```

    ---

    ## Question 7: Estimate costs for fine-tuning a 70B model

    **Answer:**

    ```
    Given:
    - Model: Llama 3 70B
    - GPUs: 16x A100 (80GB) with FSDP or DeepSpeed ZeRO-3
    - Training: 10K samples, 3 epochs
    - Duration: 48 hours

    Compute Cost:
    - On-demand: 16 GPUs × $3.00/hour × 48 hours = $2,304
    - Spot: $2,304 × 0.3 = $691 (70% savings)

    Storage Cost:
    - Base model: 140 GB × $0.023/GB/month = $3/month (negligible)
    - Checkpoints: 20 checkpoints × 140 GB = 2.8 TB
    - S3 storage: 2.8 TB × $0.023/GB = $64/month
    - LoRA adapters (instead of full model): 20 × 500 MB = 10 GB → $0.23/month

    Network Cost:
    - Model download: 140 GB × $0.09/GB = $12.60
    - Checkpoint uploads: 2.8 TB × $0.09/GB = $252

    Total:
    - With full fine-tuning (on-demand): $2,304 + $64 + $265 = $2,633
    - With QLoRA (spot): $691 + $0.23 + $265 = $956 (64% savings)

    Recommendation: Use QLoRA + spot instances → $956 per 70B fine-tuning job
    ```

    ---

    ## Common Pitfalls to Avoid

    1. **Not using LoRA/QLoRA**: Full fine-tuning wastes 10x memory and money for most tasks
    2. **Forgetting gradient checkpointing**: 50% memory savings for ~20% speed penalty
    3. **No flash attention**: 2-4x slower training, especially for long sequences
    4. **Skipping evaluation**: Perplexity can decrease while generation quality degrades
    5. **Improper prompt formatting**: Instruction format must match training data
    6. **Over-training**: More epochs ≠ better. Watch eval loss for overfitting
    7. **Not merging adapters**: Deploying with adapters requires base model at inference
    8. **Ignoring warmup**: Learning rate warmup critical for stable training

---

=== "📝 Additional Resources"

    ## Key Concepts to Master

    1. **Parameter-Efficient Fine-tuning (PEFT)**
       - LoRA (Low-Rank Adaptation)
       - QLoRA (Quantized LoRA with NF4)
       - Prefix tuning, P-tuning
       - Adapter layers

    2. **Distributed Training**
       - FSDP (Fully Sharded Data Parallel)
       - DeepSpeed ZeRO (stages 1, 2, 3)
       - Data parallelism vs model parallelism
       - Pipeline parallelism

    3. **Quantization**
       - 4-bit (NF4, FP4) for QLoRA
       - 8-bit (INT8) quantization
       - GPTQ, AWQ for inference

    4. **RLHF Pipeline**
       - Supervised fine-tuning (SFT)
       - Reward model training
       - PPO (Proximal Policy Optimization)
       - Direct Preference Optimization (DPO)

    5. **Memory Optimization**
       - Gradient checkpointing
       - Flash Attention 2
       - Mixed precision (BF16/FP16)
       - CPU offloading

    ## Real-World LLM Fine-tuning Platforms

    **OpenAI Fine-tuning API**:
    - Supports GPT-3.5, GPT-4 fine-tuning
    - Managed infrastructure (no GPU management)
    - Cost: $0.008/1K tokens training, $0.012/1K tokens inference
    - Use case: Quick prototyping, no ML expertise needed

    **Hugging Face AutoTrain**:
    - Open-source, supports any Hugging Face model
    - LoRA/QLoRA support
    - Integration with Spaces for deployment
    - Cost: Pay for GPU hours on Hugging Face infrastructure

    **Together AI**:
    - Fine-tune Llama, Mistral, Mixtral
    - Optimized for production (vLLM deployment)
    - Pricing: $0.50-2.00 per 1M tokens training
    - Use case: Enterprise fine-tuning at scale

    **Databricks Mosaic AI**:
    - End-to-end platform (data prep → deployment)
    - FSDP/DeepSpeed integration
    - MLflow tracking
    - Use case: Enterprise with existing Databricks infrastructure

    **Amazon Bedrock Fine-tuning**:
    - Managed fine-tuning for foundation models
    - Integration with AWS services (S3, SageMaker)
    - Cost: Model-dependent, ~$1000-5000 per job
    - Use case: AWS-native deployments

    ## Interview Strategies

    1. **Start with use case**: Clarify instruction tuning vs RLHF vs domain adaptation
    2. **Model size matters**: 7B → LoRA on single GPU, 70B → QLoRA + FSDP on 8-16 GPUs
    3. **Discuss trade-offs**: Full fine-tuning vs LoRA (quality vs efficiency)
    4. **Memory calculations**: Show GPU memory breakdown
    5. **Cost optimization**: Spot instances, LoRA adapters, gradient checkpointing
    6. **Evaluation strategy**: Perplexity + human eval + task-specific metrics

    ## Practice Problems

    - Design **multi-modal fine-tuning** for vision-language models (LLaVA)
    - Build **automatic prompt optimization** during fine-tuning
    - Implement **continual learning** (fine-tune without catastrophic forgetting)
    - Design **federated fine-tuning** (privacy-preserving, distributed data)
    - Optimize **longest sequence length** for 70B model on 8x A100

    ## References

    - **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
    - **QLoRA Paper**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
    - **FSDP Tutorial**: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
    - **DeepSpeed**: https://www.deepspeed.ai/tutorials/
    - **Hugging Face PEFT**: https://huggingface.co/docs/peft
    - **vLLM**: https://docs.vllm.ai/
    - **OpenAI Fine-tuning Guide**: https://platform.openai.com/docs/guides/fine-tuning
