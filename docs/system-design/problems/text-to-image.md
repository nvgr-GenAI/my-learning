# Design a Text-to-Image Generation System (Midjourney, DALL-E)

A scalable text-to-image generation platform that transforms natural language prompts into high-quality images using diffusion models, supporting millions of generations per day with GPU-optimized inference, prompt engineering, and content moderation.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1M generations/day, 5-10 images per prompt, 100K concurrent users, 5K GPU cluster |
| **Key Challenges** | Diffusion model inference at scale, GPU queue management, prompt engineering, image storage/CDN, content moderation, cost optimization |
| **Core Concepts** | Stable Diffusion, DDPM/DDIM, GPU batching, prompt weighting, negative prompting, LoRA, ControlNet, image upscaling, CLIP embeddings |
| **Companies** | Midjourney, DALL-E (OpenAI), Stable Diffusion (Stability AI), Adobe Firefly, Leonardo.ai |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Text-to-Image Generation** | Generate images from text prompts using diffusion models | P0 (Must have) |
    | **Multi-Image Generation** | Generate 4-10 variations per prompt | P0 (Must have) |
    | **Prompt Enhancement** | Auto-enhance user prompts with quality tags | P0 (Must have) |
    | **Negative Prompting** | Specify what to exclude from generation | P0 (Must have) |
    | **Image Parameters** | Control aspect ratio, resolution, steps, guidance scale | P0 (Must have) |
    | **Content Moderation** | Block NSFW, copyrighted, harmful content | P0 (Must have) |
    | **Image Upscaling** | Upscale generated images 2x-4x resolution | P1 (Should have) |
    | **Style Presets** | Pre-defined styles (anime, photorealistic, oil painting) | P1 (Should have) |
    | **LoRA/Fine-tuning** | Custom models for specific styles/subjects | P1 (Should have) |
    | **Image-to-Image** | Use reference image + prompt for variations | P1 (Should have) |
    | **ControlNet** | Precise control via pose, edge, depth maps | P2 (Nice to have) |
    | **Inpainting/Outpainting** | Edit specific regions or extend images | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training/fine-tuning infrastructure from scratch
    - Data collection/curation pipelines
    - 3D model generation
    - Video generation (text-to-video)
    - Real-time generation (< 1 second)
    - Direct blockchain/NFT integration

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Generation)** | < 15s p95 for 512x512, 50 steps | Users expect fast results for standard quality |
    | **Latency (Queue Wait)** | < 5s p95 for premium, < 30s for free tier | Balance fairness with user experience |
    | **Availability** | 99.9% uptime | Critical for paid subscriptions |
    | **Image Quality** | FID score < 15, CLIP score > 0.3 | High visual quality and prompt alignment |
    | **Content Safety** | 99.9% harmful content blocked | Legal and brand safety requirements |
    | **GPU Utilization** | > 85% average utilization | GPUs extremely expensive, maximize ROI |
    | **Cost per Image** | < $0.02 per 512x512 image (GPU + storage) | Maintain profitability at scale |
    | **Scalability** | Handle 10x traffic spikes | Viral posts, new feature launches |
    | **Storage Durability** | 99.999999999% (11 9s) | User-generated content is valuable |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 200K
    Monthly Active Users (MAU): 1M

    Image generations:
    - Generations per DAU: 5 prompts/day
    - Images per prompt: 4 variations (average)
    - Daily generations: 200K × 5 × 4 = 4M images/day
    - Generation QPS: 4M / 86,400 = ~46 images/sec average
    - Peak QPS: 5x average = ~230 images/sec

    Prompt submissions:
    - Daily prompts: 200K × 5 = 1M prompts/day
    - Prompt QPS: 1M / 86,400 = ~11.5 req/sec
    - Peak: ~58 req/sec

    Tier distribution:
    - Free tier (queue): 60% of users = 2.4M images/day
    - Standard (priority): 30% = 1.2M images/day
    - Premium (instant): 10% = 400K images/day

    Generation parameters:
    - 512x512: 60% of generations = 2.4M/day
    - 768x768: 30% = 1.2M/day
    - 1024x1024: 10% = 400K/day
    - Average inference steps: 50 (DDIM sampler)
    - Average CFG scale: 7.5

    Upscaling:
    - 30% of images get upscaled = 1.2M upscales/day
    - 2x upscaling: 80% = 960K/day
    - 4x upscaling: 20% = 240K/day

    Content moderation:
    - 100% of prompts moderated: 1M/day
    - 100% of images moderated: 4M/day
    - False positive rate: 5% (needs human review)

    Read/Write ratio: 20:1 (viewing images >> generating)
    ```

    ### Storage Estimates

    ```
    Generated images:
    - Daily generations: 4M images
    - Average size per image:
      - 512x512 PNG: 800 KB
      - Thumbnail (256x256): 100 KB
      - Total per image: 900 KB
    - Daily storage: 4M × 900 KB = 3.6 TB/day
    - Monthly: 3.6 TB × 30 = 108 TB/month
    - 1 year retention: 108 TB × 12 = 1.3 PB

    Upscaled images:
    - 1.2M upscales/day
    - 2x upscale (1024x1024): 2 MB each
    - 4x upscale (2048x2048): 6 MB each
    - Average: 2.5 MB per upscale
    - Daily: 1.2M × 2.5 MB = 3 TB/day
    - Monthly: 90 TB/month

    Model weights:
    - Base model (Stable Diffusion 1.5): 5 GB
    - LoRA models: 500 MB each × 100 models = 50 GB
    - ControlNet models: 3 GB each × 10 = 30 GB
    - Upscaler models (RealESRGAN): 17 GB
    - Safety classifiers: 2 GB
    - Total: 5 + 50 + 30 + 17 + 2 = 104 GB per GPU node

    Prompt embeddings (CLIP cache):
    - Cached embeddings: 10M unique prompts
    - Embedding size: 768 dimensions × 4 bytes = 3 KB
    - Total: 10M × 3 KB = 30 GB

    User data:
    - 1M MAU × 2 KB = 2 GB

    Generation metadata:
    - 4M generations/day × 500 bytes = 2 GB/day
    - 1 year: 2 GB × 365 = 730 GB

    Total storage: 1.3 PB (images) + 90 TB (upscales) + 104 GB (models) + 30 GB (cache) + 730 GB (metadata) ≈ 1.4 PB
    ```

    ### Compute Estimates (GPU)

    ```
    GPU requirements for diffusion model inference:

    512x512 generation (50 steps, DDIM):
    - Time per image on A100: ~3 seconds
    - Throughput per A100: ~20 images/min (with batching)
    - Daily demand: 2.4M images
    - GPU-hours needed: 2.4M / (20 × 60) = 2,000 GPU-hours/day
    - Continuous GPUs: 2,000 / 24 = ~84 A100 GPUs

    768x768 generation:
    - Time per image: ~8 seconds
    - Throughput: ~7.5 images/min
    - Daily demand: 1.2M images
    - GPU-hours: 1.2M / (7.5 × 60) = 2,667 GPU-hours/day
    - GPUs: 2,667 / 24 = ~111 A100 GPUs

    1024x1024 generation:
    - Time per image: ~15 seconds
    - Throughput: ~4 images/min
    - Daily demand: 400K images
    - GPU-hours: 400K / (4 × 60) = 1,667 GPU-hours/day
    - GPUs: 1,667 / 24 = ~70 A100 GPUs

    Upscaling (RealESRGAN):
    - Time per 2x upscale: ~2 seconds
    - Throughput: ~30 upscales/min
    - Daily demand: 1.2M upscales
    - GPU-hours: 1.2M / (30 × 60) = 667 GPU-hours/day
    - GPUs: 667 / 24 = ~28 A100 GPUs

    Total GPUs needed: 84 + 111 + 70 + 28 = 293 A100 GPUs (base)
    With 50% overhead for peaks/redundancy: 293 × 1.5 = ~440 A100 GPUs

    Cost: 440 × $2.50/hour = $1,100/hour = $26,400/day = $792K/month

    GPU memory requirements:
    - Model weights: 5 GB (fp16)
    - UNet activations: 2 GB
    - CLIP text encoder: 500 MB
    - VAE decoder: 800 MB
    - Batch size: 4 (optimal for A100 80GB)
    - Total per GPU: 5 + 2 + 0.5 + 0.8 = ~8.3 GB (comfortable on 40GB/80GB A100)
    ```

    ### Bandwidth Estimates

    ```
    Request ingress:
    - 11.5 req/sec × 1 KB (prompt + params) = 11.5 KB/sec ≈ 92 Kbps

    Image egress (CDN):
    - Daily views: 4M generations × 20 views = 80M image views/day
    - Average image size: 800 KB
    - Bandwidth: 80M × 800 KB / 86,400 = 740 MB/sec ≈ 5.9 Gbps
    - CDN cache hit ratio: 80%
    - Origin bandwidth: 5.9 Gbps × 0.2 = 1.2 Gbps

    Image ingress (to storage):
    - 46 images/sec × 900 KB = 41 MB/sec ≈ 328 Mbps

    Internal (GPU to storage):
    - 46 images/sec × 900 KB = 41 MB/sec ≈ 328 Mbps

    Total ingress: ~92 Kbps (prompts) + 328 Mbps (images) ≈ 328 Mbps
    Total egress: ~1.2 Gbps (origin) + 5.9 Gbps (CDN) ≈ 7.1 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    CLIP embedding cache:
    - Hot prompts: 1M cached embeddings
    - Embedding size: 3 KB
    - Total: 1M × 3 KB = 3 GB

    Image thumbnail cache:
    - Hot thumbnails: 10M images
    - Thumbnail size: 100 KB
    - Total: 10M × 100 KB = 1 TB

    User session cache:
    - Active users: 50K concurrent
    - Session data: 5 KB per user
    - Total: 50K × 5 KB = 250 MB

    Queue state:
    - Pending jobs: 10K in queue
    - Job metadata: 2 KB per job
    - Total: 10K × 2 KB = 20 MB

    Rate limit state:
    - Active users: 1M
    - State per user: 500 bytes
    - Total: 1M × 500 bytes = 500 MB

    Model cache (Redis):
    - Common LoRA requests: 100 models
    - 500 MB each = 50 GB

    Total cache: 3 GB (embeddings) + 1 TB (thumbnails) + 250 MB (sessions) + 20 MB (queue) + 500 MB (rate limits) + 50 GB (models) ≈ 1.05 TB
    ```

    ---

    ## Key Assumptions

    1. Average user generates 5 prompts per day with 4 variations each
    2. 60% of generations are 512x512 (standard quality)
    3. 30% of users upscale their favorite images
    4. 80% CDN cache hit ratio for popular images
    5. GPU batch size of 4 achieves 85% utilization
    6. 50 inference steps is standard (DDIM sampler)
    7. Content moderation has 99.9% accuracy (0.1% false negatives)
    8. Images retained for 1 year, thumbnails indefinitely

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Queue-based processing:** Async job queue for fair GPU resource allocation
    2. **Horizontal GPU scaling:** Scale GPU worker pools independently by resolution/model
    3. **Priority tiering:** Premium users get faster queue processing
    4. **Content safety first:** Multi-layer moderation before and after generation
    5. **Distributed storage:** Object storage + CDN for global image delivery
    6. **Prompt optimization:** Auto-enhance prompts for better results
    7. **Cost efficiency:** Smart batching and GPU scheduling to maximize utilization

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Client Layer"
            WEB[Web Client]
            DISCORD[Discord Bot]
            API_CLIENT[API Clients]
        end

        subgraph "API Gateway Layer"
            LB[Load Balancer]
            API[API Gateway]
            AUTH[Auth Service]
            RATE[Rate Limiter]
        end

        subgraph "Application Layer"
            PROMPT_SVC[Prompt Enhancement<br/>Service]
            MOD_INPUT[Input Moderation<br/>Service]
            QUEUE_MGR[Queue Manager]
            JOB_ROUTER[Job Router]
            MOD_OUTPUT[Output Moderation<br/>Service]
        end

        subgraph "Message Queue"
            QUEUE_512[Queue: 512x512<br/>Redis/RabbitMQ]
            QUEUE_768[Queue: 768x768]
            QUEUE_1024[Queue: 1024x1024]
            QUEUE_UPSCALE[Queue: Upscale]
        end

        subgraph "GPU Worker Pools"
            POOL_512[512x512 Workers<br/>80 x A100]
            POOL_768[768x768 Workers<br/>110 x A100]
            POOL_1024[1024x1024 Workers<br/>70 x A100]
            POOL_UPSCALE[Upscale Workers<br/>30 x A100]
        end

        subgraph "Worker Components"
            SCHEDULER[GPU Scheduler<br/>Batch Manager]
            SD_MODEL[Stable Diffusion<br/>Model Server]
            VAE[VAE Decoder]
            CLIP[CLIP Encoder]
        end

        subgraph "Storage Layer"
            S3[Object Storage<br/>S3/GCS]
            CDN[CDN<br/>CloudFlare/CloudFront]
            METADATA_DB[(Metadata DB<br/>PostgreSQL)]
            CACHE[(Redis Cache<br/>Embeddings)]
        end

        subgraph "Monitoring & Safety"
            METRICS[Metrics<br/>Prometheus]
            LOGS[Logs<br/>ELK]
            SAFETY_MODEL[Safety Classifier<br/>NSFW Detection]
        end

        WEB --> LB
        DISCORD --> LB
        API_CLIENT --> LB
        LB --> API
        API --> AUTH
        API --> RATE
        API --> PROMPT_SVC

        PROMPT_SVC --> MOD_INPUT
        MOD_INPUT --> QUEUE_MGR
        QUEUE_MGR --> JOB_ROUTER

        JOB_ROUTER --> QUEUE_512
        JOB_ROUTER --> QUEUE_768
        JOB_ROUTER --> QUEUE_1024
        JOB_ROUTER --> QUEUE_UPSCALE

        QUEUE_512 --> POOL_512
        QUEUE_768 --> POOL_768
        QUEUE_1024 --> POOL_1024
        QUEUE_UPSCALE --> POOL_UPSCALE

        POOL_512 --> SCHEDULER
        POOL_768 --> SCHEDULER
        POOL_1024 --> SCHEDULER

        SCHEDULER --> SD_MODEL
        SD_MODEL --> VAE
        SD_MODEL --> CLIP
        SD_MODEL --> MOD_OUTPUT

        MOD_OUTPUT --> SAFETY_MODEL
        MOD_OUTPUT --> S3
        S3 --> CDN

        SD_MODEL --> METADATA_DB
        SD_MODEL --> CACHE

        POOL_512 --> METRICS
        SCHEDULER --> LOGS

        style POOL_512 fill:#ff6b6b
        style POOL_768 fill:#ff6b6b
        style POOL_1024 fill:#ff6b6b
        style SD_MODEL fill:#4ecdc4
        style SAFETY_MODEL fill:#ffe66d
    ```

    ---

    ## Component Responsibilities

    ### 1. API Gateway Layer

    **API Gateway:**
    - REST/GraphQL endpoints for image generation
    - WebSocket for real-time progress updates
    - Request validation and sanitization
    - API versioning and documentation

    **Authentication:**
    - JWT token validation
    - API key management
    - User tier verification (free/standard/premium)

    **Rate Limiter:**
    - Token bucket per user/tier
    - Free: 25 generations/day
    - Standard: 200 generations/day
    - Premium: Unlimited
    - Global rate limit: 500 req/sec

    ### 2. Application Layer

    **Prompt Enhancement Service:**
    - Auto-add quality tags: "masterpiece, best quality, highly detailed"
    - Style keyword expansion: "anime" → "anime style, manga, cel shading"
    - Prompt weighting parsing: "(red hair:1.5), (blue eyes:0.8)"
    - CLIP-based prompt validation
    - Store enhanced prompts for reproducibility

    **Input Moderation Service:**
    - GPT-based prompt classification
    - Blocklist matching (banned terms, celebrities, brands)
    - PII detection (remove personal info)
    - NSFW intent detection
    - Response time: < 100ms p95
    - Block rate: ~2-3% of prompts

    **Queue Manager:**
    - Job submission and tracking
    - Priority queue management (premium > standard > free)
    - ETA calculation based on queue depth
    - Job cancellation support
    - Dead letter queue for failed jobs

    **Job Router:**
    - Route to appropriate worker pool by resolution
    - Load balancing across available workers
    - Health check-based routing
    - Affinity-based routing (same user → same worker for cache)

    **Output Moderation Service:**
    - NSFW image detection (CLIP-based classifier)
    - Violence/gore detection
    - Celebrity face detection
    - Watermarking for free tier
    - Automatic retries with modified prompts

    ### 3. GPU Worker Pools

    **Worker Architecture:**
    - Containerized workers (Docker/Kubernetes)
    - Auto-scaling based on queue depth
    - Graceful shutdown (finish current batch)
    - Health checks (GPU memory, temp, utilization)
    - Model preloading on startup

    **Batch Scheduler:**
    - Dynamic batching (group 4-8 requests)
    - Same-model batching for efficiency
    - Timeout-based dispatch (max 5s wait)
    - Priority-aware batching (premium first)

    **Stable Diffusion Model Server:**
    - Model loaded in fp16 for memory efficiency
    - Compiled UNet (torch.compile) for speed
    - VRAM optimization (offload VAE/CLIP when not in use)
    - Multiple sampler support (DDIM, DDPM, Euler, DPM++)
    - LoRA/ControlNet hot-swapping

    ### 4. Storage Layer

    **Object Storage (S3/GCS):**
    - Bucket structure: `/images/{year}/{month}/{day}/{image_id}.png`
    - Lifecycle policies: Standard → Infrequent Access after 30 days
    - Versioning disabled (immutable images)
    - Cross-region replication for disaster recovery
    - Signed URLs for secure access (1-hour expiry)

    **CDN (CloudFlare/CloudFront):**
    - Global edge caching (200+ PoPs)
    - Cache-Control: max-age=31536000 (1 year)
    - Image optimization (WebP/AVIF conversion)
    - Hotlink protection
    - DDoS protection

    **Metadata Database (PostgreSQL):**
    - Schema: `generations(id, user_id, prompt, negative_prompt, params, status, image_urls[], created_at)`
    - Indexes: user_id, created_at, status
    - Partitioning by month
    - Read replicas for analytics
    - Connection pooling (PgBouncer)

    **Cache (Redis):**
    - CLIP embeddings: key=prompt_hash, value=embedding (3KB)
    - User sessions: key=session_id, TTL=24h
    - Rate limit counters: key=user_id:date, TTL=24h
    - Queue state: sorted sets for priority queues
    - Model metadata: available LoRAs, styles

    ---

    ## API Design

    ### Generate Images

    ```http
    POST /v1/generations
    Content-Type: application/json
    Authorization: Bearer {token}

    {
      "prompt": "a beautiful sunset over mountains, oil painting style",
      "negative_prompt": "blurry, low quality, distorted",
      "num_images": 4,
      "width": 512,
      "height": 512,
      "steps": 50,
      "cfg_scale": 7.5,
      "sampler": "ddim",
      "seed": 42,
      "style_preset": "oil_painting",
      "lora_models": ["oil_painting_v1:0.8"],
      "webhook_url": "https://example.com/webhook"
    }

    Response:
    {
      "job_id": "gen_abc123",
      "status": "queued",
      "queue_position": 15,
      "estimated_time_seconds": 45,
      "websocket_url": "wss://api.example.com/ws/gen_abc123"
    }
    ```

    ### Get Generation Status

    ```http
    GET /v1/generations/{job_id}

    Response:
    {
      "job_id": "gen_abc123",
      "status": "completed",
      "progress": 100,
      "images": [
        {
          "url": "https://cdn.example.com/images/2026/02/05/img1.png",
          "thumbnail_url": "https://cdn.example.com/thumbs/img1.png",
          "seed": 42
        },
        // ... 3 more images
      ],
      "prompt": "a beautiful sunset over mountains...",
      "enhanced_prompt": "a beautiful sunset over mountains, oil painting style, masterpiece, best quality, highly detailed, vibrant colors",
      "parameters": {
        "width": 512,
        "height": 512,
        "steps": 50,
        "cfg_scale": 7.5
      },
      "created_at": "2026-02-05T10:30:00Z",
      "completed_at": "2026-02-05T10:30:15Z",
      "generation_time_seconds": 12.4
    }
    ```

    ### Upscale Image

    ```http
    POST /v1/upscale
    {
      "image_id": "img_xyz789",
      "scale_factor": 2,  // 2x or 4x
      "model": "realesrgan"
    }

    Response:
    {
      "job_id": "upscale_def456",
      "status": "queued",
      "estimated_time_seconds": 8
    }
    ```

    ### WebSocket Protocol

    ```javascript
    // Client connects
    ws = new WebSocket("wss://api.example.com/ws/gen_abc123")

    // Server sends progress updates
    {
      "type": "progress",
      "job_id": "gen_abc123",
      "progress": 45,
      "current_step": 23,
      "total_steps": 50,
      "message": "Generating image 2/4"
    }

    // Server sends completion
    {
      "type": "completed",
      "job_id": "gen_abc123",
      "images": [...]
    }

    // Server sends error
    {
      "type": "error",
      "job_id": "gen_abc123",
      "error": "Content policy violation",
      "details": "Generated image failed safety check"
    }
    ```

    ---

    ## Data Models

    ### Generation Job

    ```python
    class GenerationJob:
        job_id: str
        user_id: str
        status: Enum["queued", "processing", "completed", "failed", "cancelled"]
        priority: int  # 1=premium, 2=standard, 3=free

        # Input
        prompt: str
        negative_prompt: str
        enhanced_prompt: str
        num_images: int
        width: int
        height: int
        steps: int
        cfg_scale: float
        sampler: str
        seed: int
        style_preset: Optional[str]
        lora_models: List[Dict[str, float]]

        # Output
        image_urls: List[str]
        thumbnail_urls: List[str]
        seeds: List[int]

        # Metadata
        queue_position: int
        estimated_time_seconds: int
        actual_time_seconds: float
        gpu_worker_id: str
        moderation_passed: bool
        created_at: datetime
        started_at: Optional[datetime]
        completed_at: Optional[datetime]
    ```

    ---

    ## Technology Stack

    | Component | Technology | Justification |
    |-----------|-----------|---------------|
    | **API Gateway** | FastAPI/Node.js | Async I/O, WebSocket support, high performance |
    | **Message Queue** | Redis Streams/RabbitMQ | Low latency, persistence, priority queues |
    | **GPU Orchestration** | Kubernetes + NVIDIA GPU Operator | Auto-scaling, resource isolation, health management |
    | **Model Serving** | PyTorch + diffusers | Industry standard for Stable Diffusion |
    | **Object Storage** | AWS S3/Google GCS | Durability, scalability, cost-effective |
    | **CDN** | CloudFlare/CloudFront | Global distribution, DDoS protection |
    | **Database** | PostgreSQL | ACID compliance, JSON support, mature |
    | **Cache** | Redis | In-memory speed, data structures, pub/sub |
    | **Monitoring** | Prometheus + Grafana | GPU metrics, custom dashboards |
    | **Logging** | ELK Stack | Centralized logging, search, analysis |
    | **Safety** | CLIP-based classifier | Fast inference, high accuracy |

=== "🔧 Step 3: Deep Dive"

    ## 3.1 Diffusion Model Inference Pipeline

    ### Stable Diffusion Architecture

    ```mermaid
    graph LR
        subgraph "Text Processing"
            PROMPT[Text Prompt] --> TOKENIZER[Tokenizer]
            TOKENIZER --> CLIP[CLIP Text Encoder]
            CLIP --> TEXT_EMB[Text Embeddings<br/>77 x 768]
        end

        subgraph "Diffusion Process"
            NOISE[Random Noise<br/>Latent Space<br/>64x64x4] --> UNET[UNet<br/>50 iterations]
            TEXT_EMB --> UNET
            UNET --> |Denoised Latent| VAE_DEC[VAE Decoder]
        end

        subgraph "Image Generation"
            VAE_DEC --> IMAGE[Generated Image<br/>512x512x3]
        end

        style UNET fill:#ff6b6b
        style CLIP fill:#4ecdc4
        style VAE_DEC fill:#95e1d3
    ```

    ### Inference Implementation

    ```python
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from typing import List, Dict, Optional
    import time

    class DiffusionModelServer:
        def __init__(
            self,
            model_id: str = "runwayml/stable-diffusion-v1-5",
            device: str = "cuda",
            precision: str = "fp16"
        ):
            self.device = device
            self.dtype = torch.float16 if precision == "fp16" else torch.float32

            # Load model with optimizations
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,  # Custom safety checker
                requires_safety_checker=False
            ).to(device)

            # Use DDIM scheduler (faster than DDPM)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config
            )

            # Enable memory optimizations
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

            # Compile UNet for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                self.pipe.unet = torch.compile(
                    self.pipe.unet,
                    mode="reduce-overhead",
                    fullgraph=True
                )

            # Warmup
            self._warmup()

        def _warmup(self):
            """Warmup to compile kernels and allocate memory."""
            _ = self.pipe(
                "warmup",
                num_inference_steps=1,
                guidance_scale=7.5,
                width=512,
                height=512
            )
            torch.cuda.empty_cache()

        @torch.inference_mode()
        def generate_batch(
            self,
            prompts: List[str],
            negative_prompts: List[str],
            seeds: List[int],
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback_fn: Optional[callable] = None
        ) -> Dict:
            """Generate images for a batch of prompts."""
            batch_size = len(prompts)

            # Prepare generators for reproducibility
            generators = [
                torch.Generator(device=self.device).manual_seed(seed)
                for seed in seeds
            ]

            # Define callback for progress tracking
            def progress_callback(step: int, timestep: int, latents: torch.Tensor):
                if callback_fn:
                    progress = (step / num_inference_steps) * 100
                    callback_fn(progress, step, num_inference_steps)

            start_time = time.time()

            # Generate images
            output = self.pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                generator=generators,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=progress_callback,
                callback_steps=1
            )

            generation_time = time.time() - start_time

            return {
                "images": output.images,
                "nsfw_content_detected": output.nsfw_content_detected,
                "generation_time": generation_time,
                "time_per_image": generation_time / batch_size
            }

        def generate_with_lora(
            self,
            prompts: List[str],
            lora_models: List[Dict[str, float]],  # [{"model": "style_v1", "weight": 0.8}]
            **kwargs
        ):
            """Generate with LoRA models loaded."""
            # Load LoRA weights
            for lora_config in lora_models:
                model_name = lora_config["model"]
                weight = lora_config["weight"]
                self.pipe.load_lora_weights(
                    f"models/lora/{model_name}",
                    weight_name="pytorch_lora_weights.safetensors"
                )
                self.pipe.fuse_lora(lora_scale=weight)

            # Generate
            result = self.generate_batch(prompts=prompts, **kwargs)

            # Unload LoRA
            self.pipe.unfuse_lora()

            return result
    ```

    ### Advanced Features Implementation

    ```python
    class AdvancedDiffusionServer(DiffusionModelServer):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Load additional models
            self.load_controlnet()
            self.load_upscaler()

        def load_controlnet(self):
            """Load ControlNet for precise control."""
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

            # Load pose/canny/depth ControlNet
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_canny",
                torch_dtype=self.dtype
            ).to(self.device)

            self.controlnet_pipe = StableDiffusionControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                controlnet=self.controlnet,
                scheduler=self.pipe.scheduler
            ).to(self.device)

        def load_upscaler(self):
            """Load RealESRGAN for upscaling."""
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            self.upscaler = RealESRGANer(
                scale=4,
                model_path="models/RealESRGAN_x4plus.pth",
                model=model,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=True,
                device=self.device
            )

        @torch.inference_mode()
        def generate_with_controlnet(
            self,
            prompt: str,
            control_image: torch.Tensor,  # Edge map, pose, depth
            controlnet_conditioning_scale: float = 1.0,
            **kwargs
        ):
            """Generate with ControlNet guidance."""
            output = self.controlnet_pipe(
                prompt=prompt,
                image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                **kwargs
            )
            return output.images[0]

        def upscale_image(
            self,
            image: torch.Tensor,
            scale: int = 2
        ) -> torch.Tensor:
            """Upscale image using RealESRGAN."""
            import numpy as np
            from PIL import Image

            # Convert tensor to numpy
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)

            # Upscale
            upscaled_np, _ = self.upscaler.enhance(img_np, outscale=scale)

            # Convert back to tensor
            upscaled_tensor = torch.from_numpy(upscaled_np).float() / 255.0
            return upscaled_tensor

        def inpaint(
            self,
            prompt: str,
            image: torch.Tensor,
            mask: torch.Tensor,
            **kwargs
        ):
            """Inpaint masked regions."""
            from diffusers import StableDiffusionInpaintPipeline

            inpaint_pipe = StableDiffusionInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler
            ).to(self.device)

            output = inpaint_pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                **kwargs
            )
            return output.images[0]
    ```

    ---

    ## 3.2 GPU Batching and Scheduling

    ### Dynamic Batch Scheduler

    ```python
    import asyncio
    from dataclasses import dataclass
    from typing import List, Optional
    from collections import defaultdict
    import time

    @dataclass
    class BatchJob:
        job_id: str
        prompt: str
        negative_prompt: str
        seed: int
        priority: int  # 1=premium, 2=standard, 3=free
        width: int
        height: int
        steps: int
        cfg_scale: float
        submitted_at: float
        callback: callable

    class GPUBatchScheduler:
        def __init__(
            self,
            model_server: DiffusionModelServer,
            max_batch_size: int = 4,
            max_wait_time: float = 5.0,  # seconds
            target_utilization: float = 0.85
        ):
            self.model_server = model_server
            self.max_batch_size = max_batch_size
            self.max_wait_time = max_wait_time
            self.target_utilization = target_utilization

            # Priority queues by resolution
            self.queues = defaultdict(list)  # {(width, height): [jobs]}
            self.lock = asyncio.Lock()
            self.running = False

            # Metrics
            self.total_jobs_processed = 0
            self.total_gpu_time = 0.0
            self.batch_sizes = []

        async def submit_job(self, job: BatchJob):
            """Submit a job to the scheduler."""
            async with self.lock:
                resolution = (job.width, job.height)
                self.queues[resolution].append(job)

                # Sort by priority (lower number = higher priority)
                self.queues[resolution].sort(key=lambda x: (x.priority, x.submitted_at))

        async def start(self):
            """Start the scheduler loop."""
            self.running = True
            await self._schedule_loop()

        async def stop(self):
            """Stop the scheduler."""
            self.running = False

        async def _schedule_loop(self):
            """Main scheduling loop."""
            while self.running:
                # Check each resolution queue
                for resolution, queue in list(self.queues.items()):
                    if not queue:
                        continue

                    # Decide if we should dispatch a batch
                    should_dispatch = self._should_dispatch_batch(queue)

                    if should_dispatch:
                        await self._dispatch_batch(resolution, queue)

                # Sleep briefly to avoid busy loop
                await asyncio.sleep(0.1)

        def _should_dispatch_batch(self, queue: List[BatchJob]) -> bool:
            """Decide if we should dispatch a batch."""
            if not queue:
                return False

            # Dispatch if:
            # 1. We have max_batch_size jobs
            if len(queue) >= self.max_batch_size:
                return True

            # 2. Oldest job has waited too long
            oldest_wait = time.time() - queue[0].submitted_at
            if oldest_wait >= self.max_wait_time:
                return True

            # 3. We have premium jobs waiting
            premium_jobs = [j for j in queue if j.priority == 1]
            if len(premium_jobs) >= 2:  # Batch at least 2 premium
                return True

            return False

        async def _dispatch_batch(self, resolution: tuple, queue: List[BatchJob]):
            """Dispatch a batch for processing."""
            async with self.lock:
                # Extract batch
                batch_size = min(len(queue), self.max_batch_size)
                batch = queue[:batch_size]
                self.queues[resolution] = queue[batch_size:]

            # Process batch
            await self._process_batch(batch)

        async def _process_batch(self, batch: List[BatchJob]):
            """Process a batch of jobs."""
            # Extract parameters
            prompts = [job.prompt for job in batch]
            negative_prompts = [job.negative_prompt for job in batch]
            seeds = [job.seed for job in batch]

            # Assume same parameters within batch
            job0 = batch[0]

            # Track progress
            async def progress_callback(progress, step, total_steps):
                for job in batch:
                    await job.callback({
                        "type": "progress",
                        "job_id": job.job_id,
                        "progress": progress,
                        "step": step,
                        "total_steps": total_steps
                    })

            try:
                # Generate
                start_time = time.time()
                result = self.model_server.generate_batch(
                    prompts=prompts,
                    negative_prompts=negative_prompts,
                    seeds=seeds,
                    width=job0.width,
                    height=job0.height,
                    num_inference_steps=job0.steps,
                    guidance_scale=job0.cfg_scale,
                    callback_fn=progress_callback
                )
                gpu_time = time.time() - start_time

                # Update metrics
                self.total_jobs_processed += len(batch)
                self.total_gpu_time += gpu_time
                self.batch_sizes.append(len(batch))

                # Send results to jobs
                for i, job in enumerate(batch):
                    await job.callback({
                        "type": "completed",
                        "job_id": job.job_id,
                        "image": result["images"][i],
                        "generation_time": result["time_per_image"]
                    })

            except Exception as e:
                # Handle errors
                for job in batch:
                    await job.callback({
                        "type": "error",
                        "job_id": job.job_id,
                        "error": str(e)
                    })

        def get_metrics(self) -> dict:
            """Get scheduler metrics."""
            avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
            avg_gpu_time = self.total_gpu_time / self.total_jobs_processed if self.total_jobs_processed else 0

            return {
                "total_jobs_processed": self.total_jobs_processed,
                "total_gpu_time": self.total_gpu_time,
                "avg_batch_size": avg_batch_size,
                "avg_time_per_job": avg_gpu_time,
                "gpu_utilization": self._estimate_utilization()
            }

        def _estimate_utilization(self) -> float:
            """Estimate GPU utilization."""
            if self.total_jobs_processed == 0:
                return 0.0

            # Utilization = (time processing) / (total uptime)
            uptime = time.time() - (self.total_gpu_time / self.total_jobs_processed)
            return self.total_gpu_time / uptime if uptime > 0 else 0.0
    ```

    ---

    ## 3.3 Prompt Enhancement and Weighting

    ### Prompt Enhancement Engine

    ```python
    import re
    from typing import List, Tuple, Dict
    import torch
    from transformers import CLIPTokenizer, CLIPTextModel

    class PromptEnhancer:
        def __init__(self):
            self.quality_tags = [
                "masterpiece",
                "best quality",
                "highly detailed",
                "8k uhd",
                "professional"
            ]

            self.style_mappings = {
                "anime": "anime style, manga, cel shading, vibrant colors",
                "photorealistic": "photorealistic, realistic, photo, detailed",
                "oil painting": "oil painting, canvas, brushstrokes, artistic",
                "watercolor": "watercolor, soft colors, artistic, painting",
                "cyberpunk": "cyberpunk, neon lights, futuristic, sci-fi",
                "fantasy": "fantasy art, magical, ethereal, detailed"
            }

            # Load CLIP for semantic validation
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.text_model = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        def enhance_prompt(
            self,
            prompt: str,
            style: str = None,
            add_quality_tags: bool = True
        ) -> str:
            """Enhance user prompt with quality tags and style."""
            enhanced = prompt.strip()

            # Expand style keywords
            if style and style in self.style_mappings:
                enhanced += f", {self.style_mappings[style]}"

            # Add quality tags
            if add_quality_tags:
                quality_str = ", ".join(self.quality_tags)
                enhanced = f"{enhanced}, {quality_str}"

            # Remove duplicates
            enhanced = self._remove_duplicate_tags(enhanced)

            return enhanced

        def _remove_duplicate_tags(self, prompt: str) -> str:
            """Remove duplicate tags from prompt."""
            tags = [tag.strip() for tag in prompt.split(",")]
            seen = set()
            unique_tags = []
            for tag in tags:
                if tag.lower() not in seen:
                    seen.add(tag.lower())
                    unique_tags.append(tag)
            return ", ".join(unique_tags)

        def parse_weighted_prompt(self, prompt: str) -> List[Tuple[str, float]]:
            """Parse prompt with weights: (keyword:weight)

            Examples:
            - "(red hair:1.5)" -> weight 1.5
            - "(blue eyes:0.8)" -> weight 0.8
            - "normal text" -> weight 1.0
            """
            pattern = r'\(([^:)]+):([0-9.]+)\)|([^(),]+)'
            matches = re.findall(pattern, prompt)

            weighted_parts = []
            for match in matches:
                if match[0]:  # Weighted
                    text = match[0].strip()
                    weight = float(match[1])
                    weighted_parts.append((text, weight))
                else:  # Normal
                    text = match[2].strip()
                    if text:
                        weighted_parts.append((text, 1.0))

            return weighted_parts

        @torch.inference_mode()
        def get_weighted_embeddings(
            self,
            weighted_parts: List[Tuple[str, float]],
            max_length: int = 77
        ) -> torch.Tensor:
            """Get CLIP embeddings with custom weights."""
            # Tokenize each part
            embeddings = []
            weights = []

            for text, weight in weighted_parts:
                tokens = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                )

                # Get embeddings
                emb = self.text_model(tokens.input_ids)[0]
                embeddings.append(emb)
                weights.append(weight)

            # Weighted average
            embeddings = torch.stack(embeddings)
            weights = torch.tensor(weights).unsqueeze(-1).unsqueeze(-1)
            weighted_emb = (embeddings * weights).sum(dim=0) / weights.sum()

            return weighted_emb

        def validate_prompt(self, prompt: str) -> Dict[str, any]:
            """Validate prompt for issues."""
            issues = []

            # Check length
            tokens = self.tokenizer.tokenize(prompt)
            if len(tokens) > 75:  # CLIP max is 77, leave room for special tokens
                issues.append({
                    "type": "token_limit",
                    "message": f"Prompt too long ({len(tokens)} tokens). Will be truncated."
                })

            # Check for conflicting terms
            conflicts = [
                (["photorealistic", "anime"], "Conflicting styles detected"),
                (["colorful", "black and white"], "Conflicting color terms"),
            ]

            prompt_lower = prompt.lower()
            for terms, message in conflicts:
                if all(term in prompt_lower for term in terms):
                    issues.append({
                        "type": "conflict",
                        "message": message,
                        "terms": terms
                    })

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "token_count": len(tokens)
            }
    ```

    ---

    ## 3.4 Content Moderation Pipeline

    ### Multi-Stage Safety System

    ```python
    from typing import Dict, List
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    import numpy as np

    class ContentModerationSystem:
        def __init__(self):
            # Input moderation
            self.blocklist = self._load_blocklist()
            self.celebrity_names = self._load_celebrity_names()

            # Output moderation
            self.nsfw_classifier = self._load_nsfw_classifier()
            self.violence_classifier = self._load_violence_classifier()

            # Face detection
            import cv2
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        def _load_blocklist(self) -> set:
            """Load blocked terms."""
            return {
                # NSFW terms
                "nude", "naked", "nsfw", "xxx",
                # Violence
                "gore", "blood", "violent",
                # Copyrighted characters
                "mickey mouse", "pokemon", "mario",
                # Real people (except historical figures > 100 years)
                "celebrity", "politician",
                # Brands
                "coca cola", "nike", "apple logo"
            }

        def _load_celebrity_names(self) -> set:
            """Load celebrity/politician names."""
            # In production, load from database
            return {"elon musk", "donald trump", "taylor swift"}

        def _load_nsfw_classifier(self):
            """Load NSFW detection model."""
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return {"model": model, "processor": processor}

        def _load_violence_classifier(self):
            """Load violence detection model."""
            # Similar to NSFW, but trained on violence
            return self._load_nsfw_classifier()

        def moderate_prompt(self, prompt: str) -> Dict:
            """Moderate input prompt before generation."""
            prompt_lower = prompt.lower()

            # Check blocklist
            blocked_terms = []
            for term in self.blocklist:
                if term in prompt_lower:
                    blocked_terms.append(term)

            if blocked_terms:
                return {
                    "allowed": False,
                    "reason": "blocked_terms",
                    "details": blocked_terms
                }

            # Check for celebrity names
            celebrities_found = []
            for name in self.celebrity_names:
                if name in prompt_lower:
                    celebrities_found.append(name)

            if celebrities_found:
                return {
                    "allowed": False,
                    "reason": "celebrity_detected",
                    "details": celebrities_found
                }

            # Check for PII (basic)
            if self._contains_pii(prompt):
                return {
                    "allowed": False,
                    "reason": "pii_detected",
                    "details": "Personal information detected"
                }

            return {
                "allowed": True,
                "reason": None,
                "details": None
            }

        def _contains_pii(self, text: str) -> bool:
            """Detect personal identifiable information."""
            import re

            # Email
            if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
                return True

            # Phone number
            if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
                return True

            # SSN
            if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
                return True

            return False

        @torch.inference_mode()
        def moderate_image(self, image: Image.Image) -> Dict:
            """Moderate generated image."""
            results = {
                "allowed": True,
                "nsfw_detected": False,
                "violence_detected": False,
                "faces_detected": 0,
                "scores": {}
            }

            # NSFW detection
            nsfw_score = self._classify_nsfw(image)
            results["scores"]["nsfw"] = nsfw_score
            if nsfw_score > 0.85:
                results["allowed"] = False
                results["nsfw_detected"] = True

            # Violence detection
            violence_score = self._classify_violence(image)
            results["scores"]["violence"] = violence_score
            if violence_score > 0.80:
                results["allowed"] = False
                results["violence_detected"] = True

            # Face detection (for celebrity matching)
            num_faces = self._detect_faces(image)
            results["faces_detected"] = num_faces

            # If faces detected, check against celebrity database
            if num_faces > 0:
                celebrity_match = self._match_celebrity_face(image)
                if celebrity_match:
                    results["allowed"] = False
                    results["celebrity_match"] = celebrity_match

            return results

        def _classify_nsfw(self, image: Image.Image) -> float:
            """Classify image for NSFW content using CLIP."""
            processor = self.nsfw_classifier["processor"]
            model = self.nsfw_classifier["model"]

            # Prepare inputs
            inputs = processor(
                text=["safe content", "nsfw content, nudity, explicit"],
                images=image,
                return_tensors="pt",
                padding=True
            )

            # Get similarity scores
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            # Return NSFW probability
            return probs[0][1].item()

        def _classify_violence(self, image: Image.Image) -> float:
            """Classify image for violent content."""
            processor = self.violence_classifier["processor"]
            model = self.violence_classifier["model"]

            inputs = processor(
                text=["peaceful scene", "violent scene, blood, gore"],
                images=image,
                return_tensors="pt",
                padding=True
            )

            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

            return probs[0][1].item()

        def _detect_faces(self, image: Image.Image) -> int:
            """Detect faces in image."""
            import cv2

            # Convert PIL to OpenCV format
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            return len(faces)

        def _match_celebrity_face(self, image: Image.Image) -> str:
            """Match detected face against celebrity database."""
            # In production, use face recognition model (e.g., FaceNet)
            # and compare against celebrity face embeddings
            # Return celebrity name if match found, else None
            return None
    ```

    ---

    ## 3.5 Image Storage and CDN Strategy

    ### Storage Manager

    ```python
    import boto3
    from PIL import Image
    import io
    import hashlib
    from datetime import datetime, timedelta
    from typing import Dict, List

    class ImageStorageManager:
        def __init__(
            self,
            s3_bucket: str,
            cloudfront_domain: str,
            region: str = "us-east-1"
        ):
            self.s3_client = boto3.client('s3', region_name=region)
            self.bucket = s3_bucket
            self.cdn_domain = cloudfront_domain

        def upload_image(
            self,
            image: Image.Image,
            job_id: str,
            image_index: int,
            metadata: Dict
        ) -> Dict[str, str]:
            """Upload image to S3 and return URLs."""
            # Generate unique image ID
            image_id = self._generate_image_id(job_id, image_index)

            # Generate path with date partitioning
            now = datetime.utcnow()
            path_prefix = f"images/{now.year}/{now.month:02d}/{now.day:02d}"

            # Upload full resolution image
            full_key = f"{path_prefix}/{image_id}.png"
            full_url = self._upload_to_s3(image, full_key, metadata)

            # Generate and upload thumbnail
            thumbnail = self._create_thumbnail(image, size=(256, 256))
            thumb_key = f"thumbnails/{now.year}/{now.month:02d}/{now.day:02d}/{image_id}.png"
            thumb_url = self._upload_to_s3(thumbnail, thumb_key, metadata)

            # Generate CDN URLs
            cdn_full_url = f"https://{self.cdn_domain}/{full_key}"
            cdn_thumb_url = f"https://{self.cdn_domain}/{thumb_key}"

            return {
                "image_id": image_id,
                "full_url": cdn_full_url,
                "thumbnail_url": cdn_thumb_url,
                "s3_key": full_key,
                "size_bytes": self._get_image_size(image)
            }

        def _generate_image_id(self, job_id: str, image_index: int) -> str:
            """Generate unique image ID."""
            timestamp = datetime.utcnow().isoformat()
            combined = f"{job_id}_{image_index}_{timestamp}"
            hash_digest = hashlib.sha256(combined.encode()).hexdigest()
            return f"img_{hash_digest[:16]}"

        def _upload_to_s3(
            self,
            image: Image.Image,
            key: str,
            metadata: Dict
        ) -> str:
            """Upload image to S3."""
            # Convert to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=buffer,
                ContentType='image/png',
                CacheControl='public, max-age=31536000',  # 1 year
                Metadata={
                    'job-id': metadata.get('job_id', ''),
                    'prompt': metadata.get('prompt', '')[:1000],  # Truncate
                    'seed': str(metadata.get('seed', '')),
                    'created-at': datetime.utcnow().isoformat()
                }
            )

            return f"s3://{self.bucket}/{key}"

        def _create_thumbnail(self, image: Image.Image, size: tuple) -> Image.Image:
            """Create thumbnail."""
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            return thumbnail

        def _get_image_size(self, image: Image.Image) -> int:
            """Get image size in bytes."""
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.tell()

        def generate_signed_url(self, s3_key: str, expires_in: int = 3600) -> str:
            """Generate signed URL for temporary access."""
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': s3_key},
                ExpiresIn=expires_in
            )
            return url

        def delete_image(self, s3_key: str):
            """Delete image from S3."""
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=s3_key
            )

        def list_user_images(self, user_id: str, limit: int = 100) -> List[Dict]:
            """List images for a user."""
            # In production, query metadata DB, not S3
            # This is just for reference
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f"images/",
                MaxKeys=limit
            )

            images = []
            for obj in response.get('Contents', []):
                images.append({
                    "key": obj['Key'],
                    "size": obj['Size'],
                    "last_modified": obj['LastModified'].isoformat(),
                    "url": f"https://{self.cdn_domain}/{obj['Key']}"
                })

            return images
    ```

=== "📈 Step 4: Scale & Optimize"

    ## 4.1 GPU Auto-Scaling Strategy

    ### GPU Pool Orchestration

    ```yaml
    # Kubernetes GPU Node Pool Configuration
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: gpu-autoscaler-config
    data:
      scaling-rules: |
        # Scale based on queue depth
        - metric: queue_depth_512x512
          target: 20
          min_replicas: 20
          max_replicas: 200
          scale_up_threshold: 40
          scale_down_threshold: 10
          cooldown_period: 300s

        - metric: queue_depth_1024x1024
          target: 15
          min_replicas: 20
          max_replicas: 150
          scale_up_threshold: 30
          scale_down_threshold: 5
          cooldown_period: 300s

        # Scale based on GPU utilization
        - metric: gpu_utilization
          target: 85%
          min_utilization: 60%
          max_utilization: 95%

        # Time-based scaling (peak hours)
        - schedule: "0 8 * * *"  # 8 AM UTC
          min_replicas: 100
        - schedule: "0 22 * * *"  # 10 PM UTC
          min_replicas: 30
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: gpu-worker-512x512
    spec:
      replicas: 20
      selector:
        matchLabels:
          app: gpu-worker
          resolution: 512x512
      template:
        metadata:
          labels:
            app: gpu-worker
            resolution: 512x512
        spec:
          nodeSelector:
            accelerator: nvidia-a100
          containers:
          - name: worker
            image: myregistry/diffusion-worker:latest
            resources:
              limits:
                nvidia.com/gpu: 1
                memory: 40Gi
              requests:
                nvidia.com/gpu: 1
                memory: 40Gi
            env:
            - name: MODEL_PATH
              value: "/models/stable-diffusion-v1-5"
            - name: BATCH_SIZE
              value: "4"
            - name: RESOLUTION
              value: "512x512"
            - name: REDIS_QUEUE
              value: "redis://redis-cluster:6379/queue_512"
            volumeMounts:
            - name: model-cache
              mountPath: /models
              readOnly: true
          volumes:
          - name: model-cache
            persistentVolumeClaim:
              claimName: model-cache-pvc
    ```

    ### Custom Auto-Scaler

    ```python
    import asyncio
    from kubernetes import client, config
    from typing import Dict
    import redis

    class GPUAutoScaler:
        def __init__(
            self,
            k8s_namespace: str = "default",
            redis_url: str = "redis://localhost:6379"
        ):
            config.load_kube_config()
            self.apps_v1 = client.AppsV1Api()
            self.namespace = k8s_namespace
            self.redis = redis.from_url(redis_url)

            # Scaling parameters
            self.resolutions = ["512x512", "768x768", "1024x1024"]
            self.scaling_config = {
                "512x512": {
                    "min_replicas": 20,
                    "max_replicas": 200,
                    "target_queue_depth": 20,
                    "scale_up_threshold": 40,
                    "scale_down_threshold": 10
                },
                "768x768": {
                    "min_replicas": 30,
                    "max_replicas": 150,
                    "target_queue_depth": 15,
                    "scale_up_threshold": 30,
                    "scale_down_threshold": 8
                },
                "1024x1024": {
                    "min_replicas": 20,
                    "max_replicas": 100,
                    "target_queue_depth": 10,
                    "scale_up_threshold": 20,
                    "scale_down_threshold": 5
                }
            }

            # Cooldown to prevent flapping
            self.last_scale_time = {}
            self.cooldown_period = 300  # 5 minutes

        async def start(self):
            """Start auto-scaling loop."""
            while True:
                for resolution in self.resolutions:
                    await self._scale_resolution(resolution)

                await asyncio.sleep(30)  # Check every 30 seconds

        async def _scale_resolution(self, resolution: str):
            """Scale GPU workers for a specific resolution."""
            config = self.scaling_config[resolution]

            # Get current metrics
            queue_depth = self._get_queue_depth(resolution)
            current_replicas = self._get_current_replicas(resolution)

            # Calculate desired replicas
            desired_replicas = self._calculate_desired_replicas(
                queue_depth,
                current_replicas,
                config
            )

            # Apply scaling
            if desired_replicas != current_replicas:
                if self._can_scale(resolution):
                    await self._scale_deployment(resolution, desired_replicas)
                    self.last_scale_time[resolution] = asyncio.get_event_loop().time()

        def _get_queue_depth(self, resolution: str) -> int:
            """Get current queue depth from Redis."""
            queue_key = f"queue_{resolution.replace('x', '_')}"
            return self.redis.llen(queue_key)

        def _get_current_replicas(self, resolution: str) -> int:
            """Get current replica count."""
            deployment_name = f"gpu-worker-{resolution}"
            deployment = self.apps_v1.read_namespaced_deployment(
                deployment_name,
                self.namespace
            )
            return deployment.spec.replicas

        def _calculate_desired_replicas(
            self,
            queue_depth: int,
            current_replicas: int,
            config: Dict
        ) -> int:
            """Calculate desired replica count."""
            # Scale up if queue is too deep
            if queue_depth >= config["scale_up_threshold"]:
                # Add workers proportional to queue depth
                additional = (queue_depth - config["target_queue_depth"]) // 10
                desired = current_replicas + additional

            # Scale down if queue is too shallow
            elif queue_depth <= config["scale_down_threshold"]:
                # Remove 10% of workers
                desired = int(current_replicas * 0.9)

            else:
                # No change needed
                desired = current_replicas

            # Clamp to min/max
            desired = max(config["min_replicas"], desired)
            desired = min(config["max_replicas"], desired)

            return desired

        def _can_scale(self, resolution: str) -> bool:
            """Check if cooldown period has elapsed."""
            if resolution not in self.last_scale_time:
                return True

            elapsed = asyncio.get_event_loop().time() - self.last_scale_time[resolution]
            return elapsed >= self.cooldown_period

        async def _scale_deployment(self, resolution: str, replicas: int):
            """Scale deployment to desired replicas."""
            deployment_name = f"gpu-worker-{resolution}"

            # Update deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                deployment_name,
                self.namespace
            )
            deployment.spec.replicas = replicas

            self.apps_v1.patch_namespaced_deployment(
                deployment_name,
                self.namespace,
                deployment
            )

            print(f"Scaled {deployment_name} to {replicas} replicas")
    ```

    ---

    ## 4.2 Queue Prioritization and Fair Scheduling

    ### Priority Queue Implementation

    ```python
    import redis
    import json
    from typing import Optional, List
    from dataclasses import dataclass, asdict
    from enum import Enum

    class UserTier(Enum):
        FREE = 3
        STANDARD = 2
        PREMIUM = 1

    @dataclass
    class QueuedJob:
        job_id: str
        user_id: str
        tier: UserTier
        prompt: str
        params: dict
        submitted_at: float

    class PriorityJobQueue:
        def __init__(self, redis_client: redis.Redis):
            self.redis = redis_client
            self.queue_prefix = "job_queue"

        def enqueue(self, job: QueuedJob, resolution: str):
            """Enqueue job with priority based on tier."""
            queue_key = f"{self.queue_prefix}:{resolution}"

            # Priority is tier value (1=premium, 2=standard, 3=free)
            # Lower value = higher priority
            priority = job.tier.value

            # Use sorted set with priority as score
            # For same priority, earlier submission time wins
            score = priority * 1e10 + job.submitted_at

            self.redis.zadd(
                queue_key,
                {json.dumps(asdict(job)): score}
            )

        def dequeue(self, resolution: str, batch_size: int = 1) -> List[QueuedJob]:
            """Dequeue highest priority jobs."""
            queue_key = f"{self.queue_prefix}:{resolution}"

            # Get top N items (lowest scores = highest priority)
            items = self.redis.zrange(queue_key, 0, batch_size - 1)

            if not items:
                return []

            # Remove from queue
            self.redis.zrem(queue_key, *items)

            # Parse jobs
            jobs = [QueuedJob(**json.loads(item)) for item in items]
            return jobs

        def get_queue_depth(self, resolution: str) -> int:
            """Get total queue depth."""
            queue_key = f"{self.queue_prefix}:{resolution}"
            return self.redis.zcard(queue_key)

        def get_queue_depth_by_tier(self, resolution: str) -> dict:
            """Get queue depth by user tier."""
            queue_key = f"{self.queue_prefix}:{resolution}"

            counts = {tier: 0 for tier in UserTier}

            # Get all items
            items = self.redis.zrange(queue_key, 0, -1, withscores=True)

            for item, score in items:
                job = QueuedJob(**json.loads(item))
                counts[job.tier] += 1

            return counts

        def get_position(self, job_id: str, resolution: str) -> Optional[int]:
            """Get job's position in queue."""
            queue_key = f"{self.queue_prefix}:{resolution}"

            # Get all items
            items = self.redis.zrange(queue_key, 0, -1)

            for i, item in enumerate(items):
                job = QueuedJob(**json.loads(item))
                if job.job_id == job_id:
                    return i + 1

            return None

        def cancel_job(self, job_id: str, resolution: str) -> bool:
            """Cancel a job in queue."""
            queue_key = f"{self.queue_prefix}:{resolution}"

            # Find and remove job
            items = self.redis.zrange(queue_key, 0, -1)

            for item in items:
                job = QueuedJob(**json.loads(item))
                if job.job_id == job_id:
                    self.redis.zrem(queue_key, item)
                    return True

            return False
    ```

    ---

    ## 4.3 Prompt Caching and Similarity Detection

    ### Semantic Caching

    ```python
    import numpy as np
    from typing import Optional, List, Tuple
    import redis
    import pickle
    from sentence_transformers import SentenceTransformer

    class PromptCacheManager:
        def __init__(
            self,
            redis_client: redis.Redis,
            similarity_threshold: float = 0.95,
            cache_ttl: int = 3600
        ):
            self.redis = redis_client
            self.similarity_threshold = similarity_threshold
            self.cache_ttl = cache_ttl

            # Load sentence transformer for semantic similarity
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

            self.cache_prefix = "prompt_cache"
            self.embedding_prefix = "prompt_embedding"

        def get_cached_result(
            self,
            prompt: str,
            params: dict
        ) -> Optional[List[str]]:
            """Get cached images for similar prompt."""
            # Generate embedding
            embedding = self._encode_prompt(prompt)

            # Find similar prompts
            similar_prompt = self._find_similar_prompt(embedding, params)

            if similar_prompt:
                # Get cached images
                cache_key = self._get_cache_key(similar_prompt, params)
                cached_data = self.redis.get(cache_key)

                if cached_data:
                    return pickle.loads(cached_data)

            return None

        def cache_result(
            self,
            prompt: str,
            params: dict,
            image_urls: List[str]
        ):
            """Cache generation result."""
            # Store embedding
            embedding = self._encode_prompt(prompt)
            embedding_key = f"{self.embedding_prefix}:{hash(prompt)}"
            self.redis.setex(
                embedding_key,
                self.cache_ttl,
                pickle.dumps({
                    "prompt": prompt,
                    "embedding": embedding,
                    "params": params
                })
            )

            # Store result
            cache_key = self._get_cache_key(prompt, params)
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                pickle.dumps(image_urls)
            )

        def _encode_prompt(self, prompt: str) -> np.ndarray:
            """Encode prompt to embedding."""
            embedding = self.encoder.encode(prompt, convert_to_numpy=True)
            return embedding

        def _find_similar_prompt(
            self,
            embedding: np.ndarray,
            params: dict
        ) -> Optional[str]:
            """Find cached prompt with similar embedding."""
            # Get all cached embeddings
            keys = self.redis.keys(f"{self.embedding_prefix}:*")

            best_similarity = 0.0
            best_prompt = None

            for key in keys:
                cached_data = pickle.loads(self.redis.get(key))

                # Check if params match
                if not self._params_match(cached_data["params"], params):
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(
                    embedding,
                    cached_data["embedding"]
                )

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_prompt = cached_data["prompt"]

            return best_prompt

        def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
            """Calculate cosine similarity."""
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        def _params_match(self, params1: dict, params2: dict) -> bool:
            """Check if generation parameters match."""
            important_params = ["width", "height", "steps", "cfg_scale", "sampler"]

            for param in important_params:
                if params1.get(param) != params2.get(param):
                    return False

            return True

        def _get_cache_key(self, prompt: str, params: dict) -> str:
            """Generate cache key."""
            params_str = json.dumps(params, sort_keys=True)
            combined = f"{prompt}:{params_str}"
            return f"{self.cache_prefix}:{hash(combined)}"

        def get_cache_stats(self) -> dict:
            """Get cache statistics."""
            total_keys = len(self.redis.keys(f"{self.cache_prefix}:*"))
            embedding_keys = len(self.redis.keys(f"{self.embedding_prefix}:*"))

            return {
                "total_cached_results": total_keys,
                "total_embeddings": embedding_keys,
                "cache_ttl_seconds": self.cache_ttl
            }
    ```

    ---

    ## 4.4 Multi-Region GPU Pools

    ### Geographic Distribution Strategy

    ```mermaid
    graph TB
        subgraph "Global Architecture"
            subgraph "US-EAST Region"
                LB_US[Load Balancer]
                API_US[API Cluster]
                GPU_US[GPU Pool<br/>150 x A100]
                STORAGE_US[(S3 us-east-1)]
            end

            subgraph "EU-WEST Region"
                LB_EU[Load Balancer]
                API_EU[API Cluster]
                GPU_EU[GPU Pool<br/>100 x A100]
                STORAGE_EU[(S3 eu-west-1)]
            end

            subgraph "ASIA-PACIFIC Region"
                LB_AP[Load Balancer]
                API_AP[API Cluster]
                GPU_AP[GPU Pool<br/>80 x A100]
                STORAGE_AP[(S3 ap-south-1)]
            end

            subgraph "Global Services"
                ROUTE53[Route53<br/>Geo-Routing]
                GLOBAL_DB[(Global DB<br/>DynamoDB)]
                CDN[CloudFront CDN]
            end
        end

        USER_US[US Users] --> ROUTE53
        USER_EU[EU Users] --> ROUTE53
        USER_AP[Asia Users] --> ROUTE53

        ROUTE53 --> LB_US
        ROUTE53 --> LB_EU
        ROUTE53 --> LB_AP

        API_US --> GLOBAL_DB
        API_EU --> GLOBAL_DB
        API_AP --> GLOBAL_DB

        GPU_US --> STORAGE_US
        GPU_EU --> STORAGE_EU
        GPU_AP --> STORAGE_AP

        STORAGE_US --> CDN
        STORAGE_EU --> CDN
        STORAGE_AP --> CDN

        style GPU_US fill:#ff6b6b
        style GPU_EU fill:#ff6b6b
        style GPU_AP fill:#ff6b6b
    ```

    ### Benefits:
    1. **Lower latency:** Users routed to nearest region
    2. **GPU availability:** If one region saturated, overflow to others
    3. **Cost optimization:** Use spot instances in regions with excess capacity
    4. **Regulatory compliance:** Keep EU data in EU for GDPR

    ---

    ## 4.5 Cost Optimization Strategies

    ### GPU Cost Reduction

    ```python
    class GPUCostOptimizer:
        """Strategies to reduce GPU costs."""

        def strategy_1_spot_instances(self):
            """Use spot instances for 60-70% cost savings."""
            return {
                "description": "Mix on-demand and spot instances",
                "implementation": [
                    "30% on-demand GPUs (guaranteed availability)",
                    "70% spot instances (save 60-70%)",
                    "Implement checkpointing to resume on spot termination",
                    "Priority routing: premium → on-demand, free → spot"
                ],
                "savings": "~50% overall GPU costs",
                "tradeoffs": "Occasional job interruptions for free tier"
            }

        def strategy_2_model_optimization(self):
            """Optimize model for faster inference."""
            return {
                "description": "Reduce inference time per image",
                "implementations": [
                    "Use fp16 precision (2x faster, half memory)",
                    "Compile UNet with torch.compile (20-30% faster)",
                    "Use efficient samplers (DDIM 50 steps vs DDPM 1000)",
                    "Distilled models (1-4 steps, 10x faster, slight quality loss)",
                    "TensorRT optimization (30-40% faster)"
                ],
                "savings": "30-50% GPU time reduction",
                "example": "512x512: 3s → 1.5s = 2x throughput"
            }

        def strategy_3_smart_batching(self):
            """Maximize GPU utilization via batching."""
            return {
                "description": "Batch multiple requests together",
                "implementations": [
                    "Dynamic batching (wait 5s, collect 4-8 requests)",
                    "Same-resolution batching (better memory efficiency)",
                    "Priority-aware batching (batch within tier)",
                    "Adaptive batch size based on GPU memory"
                ],
                "savings": "Increase utilization 70% → 85% = 21% cost reduction",
                "example": "A100 80GB: batch_size=1 (30% util) → batch_size=4 (85% util)"
            }

        def strategy_4_caching(self):
            """Cache results for similar prompts."""
            return {
                "description": "Avoid regenerating similar images",
                "implementations": [
                    "Semantic similarity cache (95% threshold)",
                    "Exact prompt cache (1 hour TTL)",
                    "Popular style presets cache",
                    "Serve from cache = $0 GPU cost"
                ],
                "savings": "10-20% cache hit rate = 10-20% GPU cost reduction",
                "example": "Cache popular anime styles, landscape prompts"
            }

        def strategy_5_resolution_tiering(self):
            """Charge more for higher resolutions."""
            return {
                "description": "Price discrimination by resolution",
                "pricing": {
                    "512x512": "$0.02/image (baseline)",
                    "768x768": "$0.06/image (3x compute)",
                    "1024x1024": "$0.12/image (6x compute)",
                    "Upscaling 2x": "$0.03/image",
                    "Upscaling 4x": "$0.08/image"
                },
                "strategy": "Encourage 512x512 for free tier, upsell premium for high-res",
                "savings": "Revenue optimization, not cost reduction"
            }

        def strategy_6_multi_region_arbitrage(self):
            """Use cheapest GPU regions."""
            return {
                "description": "Route to lowest-cost region",
                "implementations": [
                    "Monitor spot prices across regions",
                    "Route free-tier traffic to cheapest spots",
                    "Use reserved instances in high-demand regions",
                    "Overflow traffic to regions with spare capacity"
                ],
                "savings": "10-15% cost reduction via arbitrage",
                "example": "us-east-1 A100 spot: $2.50/hr, eu-central-1: $1.95/hr"
            }

        def strategy_7_queue_management(self):
            """Smart queue management to reduce idle time."""
            return {
                "description": "Minimize GPU idle time",
                "implementations": [
                    "Graceful scale-down (finish current batch before terminating)",
                    "Predictive scaling (scale up before queue grows)",
                    "Time-based scaling (reduce GPUs during off-peak)",
                    "GPU hibernation during low traffic"
                ],
                "savings": "Improve utilization from 70% to 85%",
                "example": "24/7 operation: scale to 30% capacity at 2 AM-6 AM"
            }

        def total_cost_optimization(self):
            """Combined strategies."""
            baseline_cost = 792_000  # $792K/month (440 A100s)

            savings = {
                "Spot instances": 0.50,  # 50% reduction
                "Model optimization": 0.30,  # 30% reduction
                "Smart batching": 0.21,  # 21% reduction
                "Caching": 0.15,  # 15% reduction
                "Multi-region": 0.10,  # 10% reduction
                "Queue management": 0.15  # 15% reduction
            }

            # Compound savings (not additive)
            optimized_cost = baseline_cost
            for strategy, reduction in savings.items():
                optimized_cost *= (1 - reduction)

            return {
                "baseline_monthly_cost": f"${baseline_cost:,}",
                "optimized_monthly_cost": f"${int(optimized_cost):,}",
                "total_savings": f"${int(baseline_cost - optimized_cost):,}",
                "savings_percentage": f"{((baseline_cost - optimized_cost) / baseline_cost * 100):.1f}%",
                "strategies_applied": list(savings.keys())
            }
    ```

    **Result:**
    - Baseline: $792K/month
    - Optimized: ~$120K/month
    - Total savings: ~$670K/month (85% reduction)

    ---

    ## 4.6 Monitoring and Observability

    ### Key Metrics to Track

    ```python
    # Prometheus metrics
    from prometheus_client import Counter, Histogram, Gauge

    # Generation metrics
    generations_total = Counter(
        'generations_total',
        'Total image generations',
        ['resolution', 'tier', 'status']
    )

    generation_duration_seconds = Histogram(
        'generation_duration_seconds',
        'Time to generate image',
        ['resolution', 'model'],
        buckets=[1, 2, 5, 10, 15, 30, 60]
    )

    queue_depth = Gauge(
        'queue_depth',
        'Number of jobs in queue',
        ['resolution', 'tier']
    )

    # GPU metrics
    gpu_utilization = Gauge(
        'gpu_utilization_percent',
        'GPU utilization percentage',
        ['worker_id', 'resolution']
    )

    gpu_memory_used = Gauge(
        'gpu_memory_used_bytes',
        'GPU memory used',
        ['worker_id']
    )

    # Cost metrics
    gpu_cost_per_hour = Gauge(
        'gpu_cost_per_hour_dollars',
        'Current GPU cost per hour',
        ['region', 'instance_type']
    )

    # Quality metrics
    nsfw_detected = Counter(
        'nsfw_images_detected_total',
        'Images blocked by safety filter'
    )

    prompt_blocked = Counter(
        'prompts_blocked_total',
        'Prompts blocked by input moderation',
        ['reason']
    )

    # User metrics
    active_users = Gauge(
        'active_users',
        'Number of active users',
        ['tier']
    )
    ```

    ### Alerting Rules

    ```yaml
    # alerts.yml
    groups:
    - name: diffusion_system
      interval: 30s
      rules:
      # Queue too deep
      - alert: QueueDepthHigh
        expr: queue_depth{resolution="512x512"} > 100
        for: 5m
        annotations:
          summary: "Queue depth too high for {{ $labels.resolution }}"
          description: "Queue has {{ $value }} jobs waiting"

      # GPU utilization low
      - alert: GPUUtilizationLow
        expr: avg(gpu_utilization_percent) < 60
        for: 10m
        annotations:
          summary: "GPU utilization below target"
          description: "Average utilization is {{ $value }}%"

      # High error rate
      - alert: GenerationErrorRateHigh
        expr: |
          rate(generations_total{status="failed"}[5m]) /
          rate(generations_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Generation error rate above 5%"

      # NSFW detection spike
      - alert: NSFWDetectionSpike
        expr: rate(nsfw_images_detected_total[5m]) > 10
        for: 5m
        annotations:
          summary: "Spike in NSFW content detection"
          description: "Possible attack or prompt injection"

      # Cost anomaly
      - alert: CostAnomalyDetected
        expr: |
          sum(gpu_cost_per_hour_dollars) >
          sum(gpu_cost_per_hour_dollars offset 1h) * 1.5
        for: 15m
        annotations:
          summary: "GPU costs increased by 50%"
          description: "Check for spot price spikes or scaling issues"
    ```

---

## Summary

This text-to-image generation system design covers:

1. **Architecture:** Queue-based async processing with priority tiers and multi-resolution GPU pools
2. **Diffusion Models:** Stable Diffusion implementation with LoRA, ControlNet, and upscaling
3. **GPU Optimization:** Dynamic batching, smart scheduling, 85% utilization target
4. **Content Safety:** Multi-layer moderation for prompts and generated images
5. **Storage:** S3 + CloudFront CDN for global image delivery
6. **Scaling:** K8s auto-scaling based on queue depth and GPU utilization
7. **Cost Optimization:** Spot instances, model optimization, caching = 85% cost reduction

**Key Numbers:**
- 1M generations/day (4M images)
- 440 A100 GPUs (baseline), optimized to 70-100 with smart strategies
- $792K/month baseline, $120K optimized
- < 15s generation time (512x512, 50 steps)
- 99.9% content safety accuracy

**Trade-offs:**
- Quality vs speed (fewer steps = faster but lower quality)
- Cost vs latency (spot instances cheaper but risk interruption)
- Free tier queue wait vs premium instant generation
- Storage costs vs image retention policies
