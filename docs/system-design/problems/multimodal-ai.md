# Design a Multi-modal AI System (GPT-4V, Gemini)

A scalable multi-modal AI platform that processes and understands multiple input modalities (text, images, video, audio) to generate contextual responses, enabling vision-language understanding, audio transcription, and cross-modal reasoning at scale.

**Difficulty:** 🔴 Hard | **Frequency:** ⭐⭐⭐⭐⭐ Very High | **Time:** 60-75 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10M daily active users, 100M requests/day, 50M multi-modal requests/day, 500K concurrent users |
| **Key Challenges** | Multi-modal encoding at scale, unified embedding space, cross-modal attention, streaming multi-modal responses, GPU optimization for vision models, content-addressable media storage |
| **Core Concepts** | Vision transformers (ViT), CLIP, Whisper, cross-modal attention, unified embeddings, Q-Former, video frame sampling, streaming responses, modal-specific caching |
| **Companies** | OpenAI (GPT-4V), Google (Gemini), Anthropic (Claude 3), Meta (ImageBind), Microsoft (Florence) |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Text-Image Understanding** | Process images with text prompts, answer questions about images | P0 (Must have) |
    | **Video Understanding** | Analyze video content, temporal reasoning across frames | P0 (Must have) |
    | **Audio Processing** | Transcribe and understand audio, integrate with text/visual context | P0 (Must have) |
    | **Multi-modal Prompting** | Accept any combination of text/image/audio/video inputs | P0 (Must have) |
    | **Streaming Responses** | Stream text tokens as generated, even for multi-modal inputs | P0 (Must have) |
    | **Cross-modal Search** | Search across modalities (find images by text, text by image) | P0 (Must have) |
    | **Unified Embedding Space** | Project all modalities into shared embedding space | P0 (Must have) |
    | **Image Captioning** | Generate detailed descriptions of images | P1 (Should have) |
    | **Visual Question Answering** | Answer specific questions about image/video content | P1 (Should have) |
    | **OCR + Understanding** | Extract and understand text within images | P1 (Should have) |
    | **Audio-Visual Alignment** | Synchronize and reason across audio and video | P1 (Should have) |
    | **Content Moderation** | Filter harmful content across all modalities | P1 (Should have) |
    | **Multi-modal Generation** | Generate images/audio based on multi-modal context | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - Model training from scratch (assume pre-trained multi-modal models)
    - Real-time video streaming processing (< 100ms latency)
    - 3D model understanding and generation
    - Live camera feed processing
    - Multi-modal model fine-tuning infrastructure
    - Custom multi-modal dataset curation

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Text-only)** | < 500ms p95 TTFT | Competitive with text-only LLMs |
    | **Latency (Single Image)** | < 2s p95 TTFT | Additional encoding time acceptable |
    | **Latency (Video)** | < 5s p95 for 30-second video | Frame sampling and encoding overhead |
    | **Latency (Audio)** | < 1s p95 for 30-second audio | Whisper transcription time |
    | **Availability** | 99.95% uptime | Critical for production applications |
    | **Image Quality** | Support up to 4K resolution (3840×2160) | High-quality visual understanding |
    | **Video Length** | Support up to 5-minute videos | Balance quality and processing time |
    | **Audio Length** | Support up to 30-minute audio | Long-form transcription support |
    | **GPU Utilization** | > 75% average utilization | GPUs expensive, maximize efficiency |
    | **Cost per Request** | < $0.05 per multi-modal request | Keep margins healthy |
    | **Scalability** | Handle 5x traffic spikes | New feature launches, viral content |
    | **Security** | Multi-tenant isolation, no content leakage | Protect user privacy across modalities |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 10M
    Monthly Active Users (MAU): 25M

    Request breakdown by modality:
    - Text-only requests: 50M/day (50%)
    - Text + Image: 30M/day (30%)
    - Text + Video: 10M/day (10%)
    - Text + Audio: 8M/day (8%)
    - Multi-modal (2+ non-text): 2M/day (2%)
    - Total: 100M requests/day

    Request QPS:
    - Average: 100M / 86,400 = ~1,160 req/sec
    - Peak QPS: 3x average = ~3,480 req/sec

    Multi-modal QPS:
    - Multi-modal requests: 50M/day (excluding text-only)
    - Average: 50M / 86,400 = ~580 multi-modal req/sec
    - Peak: ~1,740 multi-modal req/sec

    Image processing:
    - Images per request: 1.5 average (some requests have multiple images)
    - Daily images: 32M × 1.5 = 48M images/day
    - Image QPS: 48M / 86,400 = ~555 images/sec
    - Average image size: 2 MB (before encoding)
    - Peak: ~1,665 images/sec

    Video processing:
    - Daily videos: 10M videos/day
    - Average video duration: 30 seconds
    - Frames sampled per video: 15 frames (0.5 fps)
    - Total frames processed: 10M × 15 = 150M frames/day
    - Frame QPS: 150M / 86,400 = ~1,736 frames/sec
    - Average video size: 15 MB (before encoding)

    Audio processing:
    - Daily audio clips: 10M clips/day (8M requests + 2M from multi-modal)
    - Average audio duration: 30 seconds
    - Audio QPS: 10M / 86,400 = ~115 audio/sec
    - Average audio size: 500 KB (before encoding)

    Token generation:
    - Average output tokens per request: 400 tokens
    - Total tokens/day: 100M × 400 = 40B tokens/day
    - Token throughput: 40B / 86,400 = ~463K tokens/sec

    Streaming connections:
    - 90% of requests use streaming
    - Average stream duration: 10 seconds
    - Concurrent streams: 3,480 × 0.9 × 10 = ~31,320 concurrent streams peak

    Read/Write ratio: 15:1 (viewing/searching >> processing)
    ```

    ### Storage Estimates

    ```
    User-uploaded media:

    Images:
    - Daily uploads: 48M images
    - Original image size: 2 MB average
    - Compressed/optimized: 800 KB
    - Thumbnail (256×256): 50 KB
    - Total per image: 850 KB
    - Daily storage: 48M × 850 KB = 40.8 TB/day
    - Monthly: 40.8 TB × 30 = 1.22 PB/month
    - 6 months retention: 7.3 PB

    Videos:
    - Daily uploads: 10M videos
    - Original video size: 15 MB average
    - Compressed: 8 MB
    - Extracted keyframes (15 frames × 100 KB): 1.5 MB
    - Thumbnail: 50 KB
    - Total per video: 9.5 MB
    - Daily storage: 10M × 9.5 MB = 95 TB/day
    - Monthly: 2.85 PB/month
    - 6 months retention: 17.1 PB

    Audio:
    - Daily uploads: 10M audio clips
    - Original audio: 500 KB average
    - Compressed: 300 KB
    - Daily storage: 10M × 300 KB = 3 TB/day
    - Monthly: 90 TB/month
    - 6 months retention: 540 TB

    Embeddings (vector storage):
    - Image embeddings: 48M/day × 512 dimensions × 4 bytes = 98 GB/day
    - Video frame embeddings: 150M/day × 512 × 4 = 307 GB/day
    - Audio embeddings: 10M/day × 512 × 4 = 20 GB/day
    - Text embeddings: 100M/day × 512 × 4 = 205 GB/day
    - Total embeddings: 630 GB/day = 18.9 TB/month
    - 6 months retention: 113 TB

    Conversation/request metadata:
    - 100M requests/day × 2 KB = 200 GB/day
    - 1 year retention: 200 GB × 365 = 73 TB

    Model weights:
    - Vision encoder (ViT-L/14): 1.2 GB
    - CLIP text encoder: 500 MB
    - Audio encoder (Whisper-large): 3 GB
    - Cross-modal transformer: 15 GB
    - Language model (7B params): 14 GB
    - Q-Former/projection layers: 2 GB
    - Total per node: ~35 GB
    - 500 nodes × 35 GB = 17.5 TB

    Cache layers:
    - Image embedding cache: 100M hot embeddings × 2 KB = 200 GB
    - Prompt cache: 50M active contexts × 10 KB = 500 GB
    - Response cache: 10M cached responses × 3 KB = 30 GB
    - Total cache: 730 GB

    Total storage: 7.3 PB (images) + 17.1 PB (videos) + 540 TB (audio) + 113 TB (embeddings) + 73 TB (metadata) + 17.5 TB (models) + 730 GB (cache) ≈ 25 PB
    ```

    ### Compute Estimates (GPU)

    ```
    GPU requirements for multi-modal processing:

    Vision encoding (image understanding):
    - Images per second: 555 images/sec average, 1,665 peak
    - ViT-L/14 inference time: 50ms per image on A100
    - Throughput per A100: 20 images/sec (with batching)
    - GPUs needed: 1,665 / 20 = ~84 A100 GPUs (peak)
    - With batching efficiency: ~60 A100 GPUs

    Video encoding (frame processing):
    - Frames per second: 1,736 frames/sec average
    - ViT inference: 50ms per frame
    - Throughput per A100: 20 frames/sec
    - GPUs needed: 1,736 / 20 = ~87 A100 GPUs
    - With batching: ~65 A100 GPUs

    Audio encoding (Whisper):
    - Audio clips per second: 115 clips/sec
    - Whisper-large inference: 500ms per 30-second clip
    - Throughput per A100: 2 clips/sec
    - GPUs needed: 115 / 2 = ~58 A100 GPUs
    - With batching: ~40 A100 GPUs

    Cross-modal transformer + language generation:
    - Multi-modal requests: 580 req/sec average, 1,740 peak
    - Processing time: 200ms per request (cross-attention + generation)
    - Throughput per A100: 5 requests/sec
    - GPUs needed: 1,740 / 5 = 348 A100 GPUs
    - With batching and KV cache: ~250 A100 GPUs

    Text-only language generation (50% of requests):
    - Text-only requests: 1,160 req/sec average
    - Standard LLM inference: 100ms per request
    - Throughput per A100: 10 requests/sec
    - GPUs needed: 1,160 / 10 = 116 A100 GPUs
    - With batching: ~80 A100 GPUs

    Total GPUs needed:
    - Vision: 60 A100s
    - Video: 65 A100s
    - Audio: 40 A100s
    - Multi-modal generation: 250 A100s
    - Text-only generation: 80 A100s
    - Total: 495 A100 GPUs (80GB)

    Cost: 495 × $2.50/hour = $1,237/hour = $29,700/day = $891K/month

    GPU memory requirements:
    - Vision encoder (ViT): 1.2 GB
    - Audio encoder (Whisper): 3 GB
    - Cross-modal layers: 15 GB
    - Language model: 14 GB
    - KV cache per request: 100 MB
    - Batch size: 16 concurrent requests
    - Memory per GPU: 33 GB + (16 × 100 MB) = ~35 GB (fits A100 80GB)
    ```

    ### Bandwidth Estimates

    ```
    Request ingress:
    - Text-only: 580 req/sec × 3 KB = 1.74 MB/sec
    - Images: 555 images/sec × 2 MB = 1,110 MB/sec ≈ 8.9 Gbps
    - Videos: 115 videos/sec × 15 MB = 1,725 MB/sec ≈ 13.8 Gbps
    - Audio: 115 audio/sec × 500 KB = 57.5 MB/sec ≈ 460 Mbps
    - Total ingress: ~23 Gbps

    Response egress (streaming):
    - Text responses: 1,160 req/sec × 1.5 KB = 1.74 MB/sec ≈ 14 Mbps
    - Streaming overhead (SSE): +30% = 18 Mbps

    Media egress (retrieval/playback):
    - Image views: 10x generation rate = 5,550 views/sec × 800 KB = 4.4 GB/sec ≈ 35 Gbps
    - Video views: 1,000 views/sec × 8 MB = 8 GB/sec ≈ 64 Gbps
    - Total egress: ~99 Gbps

    Internal bandwidth (GPU ↔ services):
    - Embeddings: (555 + 1,736 + 115) images/frames/audio × 2 KB = 4.8 MB/sec
    - Cross-modal attention: 580 req/sec × 50 KB = 29 MB/sec
    - KV cache sync: 1,160 req/sec × 100 KB = 116 MB/sec
    - Total internal: ~150 MB/sec ≈ 1.2 Gbps

    Total ingress: ~23 Gbps
    Total egress: ~99 Gbps
    Internal: ~1.2 Gbps
    ```

    ### Memory Estimates (Caching)

    ```
    GPU memory (KV cache):
    - Active multi-modal requests: 5K concurrent
    - KV cache per request: 100 MB
    - Total: 5K × 100 MB = 500 GB (distributed across GPUs)

    Image embedding cache (hot cache):
    - Recently processed images: 100M embeddings
    - Embedding size: 2 KB (512 dims × 4 bytes)
    - Total: 100M × 2 KB = 200 GB (Redis/Memcached)

    Video frame cache:
    - Recently extracted frames: 50M frames
    - Frame embedding: 2 KB
    - Total: 50M × 2 KB = 100 GB

    Audio embedding cache:
    - Recent audio clips: 10M embeddings
    - Embedding size: 2 KB
    - Total: 10M × 2 KB = 20 GB

    Response cache:
    - Common multi-modal queries: 10M cached responses
    - Average response: 3 KB
    - Total: 10M × 3 KB = 30 GB

    Cross-modal index (vector similarity):
    - Unified embedding index: 500M vectors
    - HNSW index overhead: 2x vector size
    - Total: 500M × 2 KB × 2 = 2 TB (distributed)

    Total cache: 500 GB (GPU) + 200 GB (image) + 100 GB (video) + 20 GB (audio) + 30 GB (response) + 2 TB (index) ≈ 2.85 TB
    ```

---

=== "🏗️ Step 2: High-Level Design"

    ## Architecture Overview

    ```mermaid
    graph TB
        subgraph "Client Layer"
            Client[Client Apps<br/>Web/Mobile/API]
        end

        subgraph "API Gateway Layer"
            LB[Load Balancer<br/>nginx/ALB]
            Gateway[API Gateway<br/>Rate limiting, Auth]
        end

        subgraph "Input Processing Service"
            Router[Request Router<br/>Modal detection]
            TextProc[Text Processor<br/>Tokenization]
            ImageProc[Image Processor<br/>Resize, normalize]
            VideoProc[Video Processor<br/>Frame extraction]
            AudioProc[Audio Processor<br/>Format conversion]
        end

        subgraph "Encoding Layer (GPU Cluster)"
            VisionEncoder[Vision Encoder<br/>ViT-L/14, CLIP]
            AudioEncoder[Audio Encoder<br/>Whisper-large]
            TextEncoder[Text Encoder<br/>CLIP text encoder]
            EmbedCache[(Embedding Cache<br/>Redis Cluster)]
        end

        subgraph "Unified Embedding Space"
            Projector[Modal Projection<br/>Q-Former/Perceiver]
            UnifiedEmbed[Unified Embedding<br/>Shared latent space]
        end

        subgraph "Cross-Modal Processing"
            CrossAttn[Cross-Modal Attention<br/>Multi-head attention]
            ModalFusion[Modal Fusion<br/>Weighted combination]
        end

        subgraph "Language Generation"
            LLM[Multi-modal LLM<br/>Flamingo/GPT-4V arch]
            KVCache[(KV Cache<br/>Distributed)]
            StreamService[Streaming Service<br/>SSE/WebSocket]
        end

        subgraph "Storage Layer"
            MediaStore[(Media Storage<br/>S3/GCS)]
            VectorDB[(Vector DB<br/>Pinecone/Weaviate)]
            MetaDB[(Metadata DB<br/>PostgreSQL)]
            ConvDB[(Conversation DB<br/>Cassandra)]
        end

        subgraph "Supporting Services"
            Moderation[Content Moderation<br/>Multi-modal safety]
            Analytics[Analytics Service<br/>Usage tracking]
            CDN[CDN<br/>CloudFront/Cloudflare]
        end

        Client --> LB
        LB --> Gateway
        Gateway --> Router

        Router --> TextProc
        Router --> ImageProc
        Router --> VideoProc
        Router --> AudioProc

        TextProc --> TextEncoder
        ImageProc --> VisionEncoder
        VideoProc --> VisionEncoder
        AudioProc --> AudioEncoder

        VisionEncoder --> EmbedCache
        AudioEncoder --> EmbedCache
        TextEncoder --> EmbedCache

        EmbedCache --> Projector
        Projector --> UnifiedEmbed
        UnifiedEmbed --> CrossAttn

        CrossAttn --> ModalFusion
        ModalFusion --> LLM

        LLM --> KVCache
        LLM --> StreamService
        StreamService --> Gateway
        Gateway --> Client

        ImageProc --> MediaStore
        VideoProc --> MediaStore
        AudioProc --> MediaStore

        UnifiedEmbed --> VectorDB
        Router --> MetaDB
        LLM --> ConvDB

        Router --> Moderation
        Analytics -.-> MetaDB
        MediaStore --> CDN
    ```

    ## Component Responsibilities

    ### 1. Input Processing Service

    **Responsibilities:**
    - Detect input modalities (text, image, video, audio)
    - Route to appropriate modal processors
    - Validate media formats and sizes
    - Apply preprocessing (resizing, normalization, frame extraction)

    **Technology:**
    - **Framework:** FastAPI/Flask for REST endpoints
    - **Image processing:** Pillow, OpenCV
    - **Video processing:** FFmpeg, cv2 for frame extraction
    - **Audio processing:** librosa, pydub

    **Key Design Decisions:**
    - Async processing for large media files
    - Content-based deduplication (perceptual hashing)
    - Progressive image loading for large files
    - Frame sampling strategies for videos (uniform, scene detection)

    ---

    ### 2. Multi-Modal Encoding Layer

    **Responsibilities:**
    - Encode images/videos using vision transformers
    - Transcribe and encode audio using Whisper
    - Encode text using CLIP text encoder
    - Cache embeddings for repeated media

    **Technology:**
    - **Vision:** ViT-L/14, CLIP vision encoder
    - **Audio:** Whisper-large-v3
    - **Text:** CLIP text encoder, BERT
    - **Caching:** Redis cluster with TTL

    **Key Design Decisions:**
    - Batch processing for GPU efficiency
    - Content-addressable caching (hash-based)
    - Dynamic batching based on modality
    - GPU memory pooling

    ---

    ### 3. Unified Embedding Space

    **Responsibilities:**
    - Project modal-specific embeddings to shared space
    - Enable cross-modal similarity computation
    - Align embeddings across modalities

    **Technology:**
    - **Architecture:** Q-Former (from BLIP-2), Perceiver
    - **Projection:** Learned linear/MLP projections
    - **Training:** Contrastive learning (CLIP-style)

    **Key Design Decisions:**
    - Fixed-size unified embeddings (512-1024 dims)
    - Modality-specific learnable queries
    - Temperature-scaled cosine similarity

    ---

    ### 4. Cross-Modal Processing

    **Responsibilities:**
    - Attend across different modalities
    - Fuse multi-modal information
    - Generate cross-modal representations

    **Technology:**
    - **Attention:** Multi-head cross-attention
    - **Fusion:** Gated fusion, adaptive weighting
    - **Architecture:** Flamingo-style perceiver resampler

    **Key Design Decisions:**
    - Late fusion (after encoding)
    - Learnable modality weights
    - Hierarchical attention (intra-modal → cross-modal)

    ---

    ### 5. Multi-Modal Language Generation

    **Responsibilities:**
    - Generate text responses conditioned on multi-modal inputs
    - Stream tokens in real-time
    - Maintain conversation context

    **Technology:**
    - **Model:** GPT-style decoder with cross-attention
    - **Streaming:** Server-Sent Events (SSE)
    - **Caching:** vLLM for KV cache management

    **Key Design Decisions:**
    - Separate text-only and multi-modal inference paths
    - Prefix caching for multi-modal embeddings
    - Dynamic batching with priority queuing

    ---

    ### 6. Storage & Retrieval

    **Responsibilities:**
    - Store user-uploaded media
    - Index embeddings for similarity search
    - Store conversation history and metadata

    **Technology:**
    - **Media:** S3/GCS with lifecycle policies
    - **Vectors:** Pinecone, Weaviate, or Milvus
    - **Metadata:** PostgreSQL (structured data)
    - **Conversations:** Cassandra (time-series writes)

    **Key Design Decisions:**
    - Content-addressable storage (dedupe by hash)
    - Hot/warm/cold storage tiers
    - HNSW indexing for vector similarity
    - Sharding by user_id for conversations

    ---

    ## Data Flow: Multi-Modal Request

    ```mermaid
    sequenceDiagram
        participant C as Client
        participant G as API Gateway
        participant R as Router
        participant I as Image Processor
        participant V as Vision Encoder
        participant T as Text Encoder
        participant P as Projector
        participant X as Cross-Attention
        participant L as LLM
        participant S as Stream Service

        C->>G: POST /chat (text + image)
        G->>R: Route request
        R->>R: Detect modalities

        par Image Processing
            R->>I: Process image
            I->>I: Resize, normalize
            I->>V: Encode image
            V->>V: ViT forward pass
            V->>P: Image embedding [768d]
        and Text Processing
            R->>T: Encode text
            T->>T: Tokenize + encode
            T->>P: Text embedding [768d]
        end

        P->>P: Project to unified space [512d]
        P->>X: Unified embeddings
        X->>X: Cross-modal attention
        X->>L: Fused representation

        L->>L: Generate tokens (streaming)
        loop Stream tokens
            L->>S: Token chunk
            S->>G: SSE event
            G->>C: Token stream
        end

        L->>S: [DONE]
        S->>G: Close stream
        G->>C: Connection closed
    ```

    ## API Design

    ### Multi-Modal Chat Completion

    ```http
    POST /v1/chat/completions
    Content-Type: multipart/form-data

    {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What's in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": "data:image/jpeg;base64,/9j/4AAQ...",
                "detail": "high"
              }
            }
          ]
        }
      ],
      "model": "gpt-4-vision-preview",
      "max_tokens": 500,
      "stream": true
    }
    ```

    **Response (Streaming SSE):**
    ```
    data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4-vision-preview","choices":[{"index":0,"delta":{"role":"assistant","content":"This"},"finish_reason":null}]}

    data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4-vision-preview","choices":[{"index":0,"delta":{"content":" image"},"finish_reason":null}]}

    data: [DONE]
    ```

    ### Video Understanding

    ```http
    POST /v1/video/analyze
    Content-Type: multipart/form-data

    {
      "video_url": "https://example.com/video.mp4",
      "prompt": "Summarize the key events in this video",
      "frame_sampling": "uniform",  // uniform, keyframes, scene
      "max_frames": 20,
      "stream": true
    }
    ```

    ### Audio Transcription + Understanding

    ```http
    POST /v1/audio/transcribe
    Content-Type: multipart/form-data

    {
      "audio_file": <binary>,
      "prompt": "Summarize the main points discussed",
      "language": "en",
      "include_timestamps": true,
      "format": "text"  // text, json, srt
    }
    ```

    ### Cross-Modal Search

    ```http
    POST /v1/search
    Content-Type: application/json

    {
      "query": {
        "type": "text",
        "content": "sunset over mountains"
      },
      "search_modalities": ["image", "video"],
      "top_k": 10,
      "filters": {
        "user_id": "user-123",
        "date_range": "last_30_days"
      }
    }
    ```

    **Response:**
    ```json
    {
      "results": [
        {
          "id": "img-456",
          "type": "image",
          "url": "https://cdn.example.com/img-456.jpg",
          "similarity_score": 0.92,
          "metadata": {
            "uploaded_at": "2024-01-15T10:30:00Z",
            "resolution": "1920x1080"
          }
        },
        {
          "id": "vid-789",
          "type": "video",
          "url": "https://cdn.example.com/vid-789.mp4",
          "similarity_score": 0.88,
          "matched_frames": [12, 45, 67],
          "metadata": {
            "duration": 120,
            "uploaded_at": "2024-01-14T15:20:00Z"
          }
        }
      ]
    }
    ```

---

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Vision Encoding with CLIP & Vision Transformers

    ### Image Encoding Pipeline

    ```python
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPVisionModel
    import hashlib
    import redis

    class VisionEncoder:
        def __init__(self, model_name="openai/clip-vit-large-patch14", cache_ttl=3600):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            # Redis cache for embeddings
            self.cache = redis.Redis(host='localhost', port=6379, db=0)
            self.cache_ttl = cache_ttl

        def compute_image_hash(self, image_bytes):
            """Content-addressable hash for deduplication"""
            return hashlib.sha256(image_bytes).hexdigest()

        def preprocess_image(self, image_path, max_size=1024):
            """Load and preprocess image"""
            image = Image.open(image_path).convert("RGB")

            # Resize if too large (maintain aspect ratio)
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)

            return image

        def encode_single(self, image):
            """Encode a single image"""
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use pooled output (CLS token) or last hidden state
                embedding = outputs.pooler_output  # [1, 768]

            return embedding.cpu().numpy()

        def encode_batch(self, images):
            """Batch encoding for efficiency"""
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output  # [batch_size, 768]

            return embeddings.cpu().numpy()

        def encode_with_cache(self, image_path):
            """Encode with caching support"""
            # Read image bytes
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # Check cache
            image_hash = self.compute_image_hash(image_bytes)
            cache_key = f"vision_emb:{image_hash}"

            cached = self.cache.get(cache_key)
            if cached is not None:
                return np.frombuffer(cached, dtype=np.float32)

            # Encode image
            image = self.preprocess_image(image_path)
            embedding = self.encode_single(image)

            # Cache result
            self.cache.setex(
                cache_key,
                self.cache_ttl,
                embedding.tobytes()
            )

            return embedding

    # Usage
    encoder = VisionEncoder()
    image_embedding = encoder.encode_with_cache("path/to/image.jpg")
    print(f"Image embedding shape: {image_embedding.shape}")  # (768,)
    ```

    ### High-Resolution Image Processing (GPT-4V Style)

    GPT-4V uses a multi-crop strategy for high-resolution images:

    ```python
    class HighResVisionEncoder:
        def __init__(self, base_encoder, crop_size=336, low_res_size=224):
            self.base_encoder = base_encoder
            self.crop_size = crop_size
            self.low_res_size = low_res_size

        def create_image_crops(self, image):
            """
            Create multiple crops + low-res version
            Similar to GPT-4V's multi-scale approach
            """
            width, height = image.size
            crops = []

            # Low-resolution full image
            low_res = image.resize((self.low_res_size, self.low_res_size), Image.LANCZOS)
            crops.append(low_res)

            # High-resolution crops
            num_crops_x = max(1, width // self.crop_size)
            num_crops_y = max(1, height // self.crop_size)

            for i in range(num_crops_x):
                for j in range(num_crops_y):
                    left = i * self.crop_size
                    top = j * self.crop_size
                    right = min(left + self.crop_size, width)
                    bottom = min(top + self.crop_size, height)

                    crop = image.crop((left, top, right, bottom))
                    crop = crop.resize((self.crop_size, self.crop_size), Image.LANCZOS)
                    crops.append(crop)

            return crops

        def encode_high_res(self, image_path):
            """Encode high-res image with multi-crop strategy"""
            image = Image.open(image_path).convert("RGB")
            crops = self.create_image_crops(image)

            # Encode all crops in batch
            embeddings = self.base_encoder.encode_batch(crops)

            # Combine embeddings (various strategies)
            # 1. Concatenation (GPT-4V approach)
            combined = embeddings.flatten()

            # 2. Average pooling
            # combined = embeddings.mean(axis=0)

            # 3. Attention-weighted pooling
            # combined = self.attention_pool(embeddings)

            return combined, embeddings
    ```

    ---

    ## 3.2 Video Understanding with Frame Sampling

    ### Video Frame Extraction & Encoding

    ```python
    import cv2
    import numpy as np
    from typing import List, Tuple

    class VideoEncoder:
        def __init__(self, vision_encoder, max_frames=20):
            self.vision_encoder = vision_encoder
            self.max_frames = max_frames

        def extract_frames_uniform(self, video_path, num_frames=None):
            """Extract frames at uniform intervals"""
            if num_frames is None:
                num_frames = self.max_frames

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps

            # Calculate frame indices
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = []
            frame_times = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                    frame_times.append(idx / fps)

            cap.release()
            return frames, frame_times, duration

        def extract_keyframes(self, video_path, threshold=30.0):
            """Extract keyframes based on scene changes"""
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_times = []
            prev_frame = None
            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS)

            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_score = np.mean(diff)

                    if diff_score > threshold:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        frame_times.append(frame_idx / fps)

                prev_frame = frame.copy()
                frame_idx += 1

            cap.release()

            # If too few keyframes, fall back to uniform sampling
            if len(frames) < 5:
                return self.extract_frames_uniform(video_path)

            return frames, frame_times, frame_idx / fps

        def encode_video(self, video_path, sampling="uniform"):
            """Encode video into frame embeddings"""
            # Extract frames
            if sampling == "uniform":
                frames, times, duration = self.extract_frames_uniform(video_path)
            elif sampling == "keyframes":
                frames, times, duration = self.extract_keyframes(video_path)
            else:
                raise ValueError(f"Unknown sampling method: {sampling}")

            # Encode frames in batch
            frame_embeddings = self.vision_encoder.encode_batch(frames)

            return {
                'embeddings': frame_embeddings,  # [num_frames, 768]
                'timestamps': times,
                'num_frames': len(frames),
                'duration': duration,
                'sampling_method': sampling
            }

        def encode_video_temporal(self, video_path):
            """Encode video with temporal context"""
            video_data = self.encode_video(video_path, sampling="uniform")

            # Add positional encoding for temporal order
            embeddings = video_data['embeddings']
            num_frames = len(embeddings)

            # Sinusoidal positional encoding
            position = np.arange(num_frames)[:, np.newaxis]
            div_term = np.exp(np.arange(0, embeddings.shape[1], 2) *
                             -(np.log(10000.0) / embeddings.shape[1]))

            pos_encoding = np.zeros_like(embeddings)
            pos_encoding[:, 0::2] = np.sin(position * div_term)
            pos_encoding[:, 1::2] = np.cos(position * div_term)

            # Combine frame embeddings with positional encoding
            temporal_embeddings = embeddings + pos_encoding

            video_data['temporal_embeddings'] = temporal_embeddings
            return video_data

    # Usage
    vision_enc = VisionEncoder()
    video_enc = VideoEncoder(vision_enc, max_frames=20)

    video_data = video_enc.encode_video_temporal("path/to/video.mp4")
    print(f"Encoded {video_data['num_frames']} frames")
    print(f"Embeddings shape: {video_data['temporal_embeddings'].shape}")
    ```

    ---

    ## 3.3 Audio Processing with Whisper

    ### Audio Transcription & Encoding

    ```python
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
    import torch

    class AudioEncoder:
        def __init__(self, model_name="openai/whisper-large-v3"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

        def load_audio(self, audio_path, sr=16000):
            """Load audio file and resample to 16kHz"""
            audio, _ = librosa.load(audio_path, sr=sr)
            return audio

        def transcribe(self, audio_path, language="en", return_timestamps=False):
            """Transcribe audio to text"""
            audio = self.load_audio(audio_path)

            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = inputs.input_features.to(self.device)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs,
                    language=language,
                    return_timestamps=return_timestamps
                )

            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            return transcription

        def encode_audio(self, audio_path):
            """Extract audio embeddings from encoder"""
            audio = self.load_audio(audio_path)

            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = inputs.input_features.to(self.device)

            with torch.no_grad():
                # Get encoder hidden states
                encoder_outputs = self.model.model.encoder(inputs)
                # Use mean pooling over time dimension
                audio_embedding = encoder_outputs.last_hidden_state.mean(dim=1)

            return audio_embedding.cpu().numpy()

        def transcribe_with_timestamps(self, audio_path):
            """Transcribe with word-level timestamps"""
            audio = self.load_audio(audio_path)

            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = inputs.input_features.to(self.device)

            with torch.no_grad():
                # Generate with timestamps
                predicted_ids = self.model.generate(
                    inputs,
                    return_timestamps=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode with timestamps
            transcription = self.processor.batch_decode(
                predicted_ids.sequences,
                skip_special_tokens=False
            )[0]

            # Parse timestamps (Whisper format: <|0.00|> text <|5.00|>)
            segments = self._parse_timestamps(transcription)

            return segments

        def _parse_timestamps(self, text):
            """Parse Whisper timestamp format"""
            import re
            pattern = r'<\|(\d+\.\d+)\|>([^<]+)'
            matches = re.findall(pattern, text)

            segments = []
            for i, (start_time, text_segment) in enumerate(matches):
                end_time = matches[i + 1][0] if i + 1 < len(matches) else None
                segments.append({
                    'start': float(start_time),
                    'end': float(end_time) if end_time else None,
                    'text': text_segment.strip()
                })

            return segments

    # Usage
    audio_enc = AudioEncoder()

    # Transcription
    transcription = audio_enc.transcribe("path/to/audio.mp3", language="en")
    print(f"Transcription: {transcription}")

    # Audio embedding
    audio_embedding = audio_enc.encode_audio("path/to/audio.mp3")
    print(f"Audio embedding shape: {audio_embedding.shape}")

    # Transcription with timestamps
    segments = audio_enc.transcribe_with_timestamps("path/to/audio.mp3")
    for seg in segments:
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
    ```

    ---

    ## 3.4 Unified Embedding Space with Q-Former

    ### Cross-Modal Projection

    ```python
    import torch
    import torch.nn as nn

    class QFormer(nn.Module):
        """
        Q-Former from BLIP-2: Queries cross-modal information
        Projects vision/audio embeddings to language-aligned space
        """
        def __init__(
            self,
            vision_dim=768,
            audio_dim=768,
            text_dim=768,
            hidden_dim=768,
            num_queries=32,
            num_layers=6,
            num_heads=12
        ):
            super().__init__()
            self.num_queries = num_queries

            # Learnable query tokens
            self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

            # Modal-specific projections
            self.vision_proj = nn.Linear(vision_dim, hidden_dim)
            self.audio_proj = nn.Linear(audio_dim, hidden_dim)
            self.text_proj = nn.Linear(text_dim, hidden_dim)

            # Transformer layers for cross-attention
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection to unified space
            self.output_proj = nn.Linear(hidden_dim, text_dim)

        def forward(self, vision_emb=None, audio_emb=None, text_emb=None):
            """
            Forward pass with any combination of modalities

            Args:
                vision_emb: [batch, seq_len_v, vision_dim] or [batch, vision_dim]
                audio_emb: [batch, seq_len_a, audio_dim] or [batch, audio_dim]
                text_emb: [batch, seq_len_t, text_dim] or [batch, text_dim]

            Returns:
                unified_emb: [batch, num_queries, text_dim]
            """
            batch_size = 1
            modal_embeddings = []

            # Process vision embeddings
            if vision_emb is not None:
                if len(vision_emb.shape) == 2:
                    vision_emb = vision_emb.unsqueeze(1)  # [batch, 1, dim]
                batch_size = vision_emb.shape[0]
                vision_emb = self.vision_proj(vision_emb)  # [batch, seq, hidden]
                modal_embeddings.append(vision_emb)

            # Process audio embeddings
            if audio_emb is not None:
                if len(audio_emb.shape) == 2:
                    audio_emb = audio_emb.unsqueeze(1)
                batch_size = audio_emb.shape[0]
                audio_emb = self.audio_proj(audio_emb)
                modal_embeddings.append(audio_emb)

            # Process text embeddings
            if text_emb is not None:
                if len(text_emb.shape) == 2:
                    text_emb = text_emb.unsqueeze(1)
                batch_size = text_emb.shape[0]
                text_emb = self.text_proj(text_emb)
                modal_embeddings.append(text_emb)

            # Concatenate all modal embeddings
            if len(modal_embeddings) == 0:
                raise ValueError("At least one modality must be provided")

            all_embeddings = torch.cat(modal_embeddings, dim=1)  # [batch, total_seq, hidden]

            # Expand query tokens for batch
            queries = self.query_tokens.expand(batch_size, -1, -1)  # [batch, num_queries, hidden]

            # Concatenate queries with modal embeddings
            input_emb = torch.cat([queries, all_embeddings], dim=1)  # [batch, num_queries + total_seq, hidden]

            # Apply transformer (queries attend to modal embeddings)
            output = self.transformer(input_emb)  # [batch, num_queries + total_seq, hidden]

            # Extract only query outputs
            query_output = output[:, :self.num_queries, :]  # [batch, num_queries, hidden]

            # Project to unified space
            unified_emb = self.output_proj(query_output)  # [batch, num_queries, text_dim]

            return unified_emb


    class UnifiedEmbeddingSpace:
        """Manages unified embedding space for all modalities"""
        def __init__(self, q_former_config=None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            config = q_former_config or {
                'vision_dim': 768,
                'audio_dim': 768,
                'text_dim': 768,
                'hidden_dim': 768,
                'num_queries': 32
            }

            self.q_former = QFormer(**config).to(self.device)
            self.q_former.eval()

        def project_to_unified(self, vision_emb=None, audio_emb=None, text_emb=None):
            """Project modal embeddings to unified space"""
            # Convert numpy to torch if needed
            if vision_emb is not None and not isinstance(vision_emb, torch.Tensor):
                vision_emb = torch.from_numpy(vision_emb).float().to(self.device)
            if audio_emb is not None and not isinstance(audio_emb, torch.Tensor):
                audio_emb = torch.from_numpy(audio_emb).float().to(self.device)
            if text_emb is not None and not isinstance(text_emb, torch.Tensor):
                text_emb = torch.from_numpy(text_emb).float().to(self.device)

            with torch.no_grad():
                unified_emb = self.q_former(
                    vision_emb=vision_emb,
                    audio_emb=audio_emb,
                    text_emb=text_emb
                )

            return unified_emb.cpu().numpy()

        def compute_similarity(self, emb1, emb2, temperature=0.07):
            """Compute cosine similarity between embeddings"""
            emb1 = torch.from_numpy(emb1) if not isinstance(emb1, torch.Tensor) else emb1
            emb2 = torch.from_numpy(emb2) if not isinstance(emb2, torch.Tensor) else emb2

            # Flatten to [batch, features]
            emb1_flat = emb1.reshape(emb1.shape[0], -1)
            emb2_flat = emb2.reshape(emb2.shape[0], -1)

            # L2 normalize
            emb1_norm = torch.nn.functional.normalize(emb1_flat, dim=-1)
            emb2_norm = torch.nn.functional.normalize(emb2_flat, dim=-1)

            # Cosine similarity
            similarity = torch.matmul(emb1_norm, emb2_norm.T) / temperature

            return similarity.numpy()

    # Usage
    unified_space = UnifiedEmbeddingSpace()

    # Example: Image + Text
    image_emb = vision_encoder.encode_single(image)  # [1, 768]
    text_emb = text_encoder.encode("A beautiful sunset")  # [1, 768]

    unified_emb = unified_space.project_to_unified(
        vision_emb=torch.from_numpy(image_emb),
        text_emb=torch.from_numpy(text_emb)
    )
    print(f"Unified embedding shape: {unified_emb.shape}")  # [1, 32, 768]

    # Cross-modal similarity
    similarity = unified_space.compute_similarity(image_emb, text_emb)
    print(f"Image-text similarity: {similarity}")
    ```

    ---

    ## 3.5 Cross-Modal Attention & Fusion

    ### Multi-Modal Transformer

    ```python
    class CrossModalAttention(nn.Module):
        """Cross-attention between different modalities"""
        def __init__(self, dim=768, num_heads=12, dropout=0.1):
            super().__init__()
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, query, key_value, attention_mask=None):
            """
            Args:
                query: [batch, seq_q, dim] - modality that attends
                key_value: [batch, seq_kv, dim] - modality being attended to
                attention_mask: [batch, seq_q, seq_kv]
            """
            # Cross-attention
            attn_output, attn_weights = self.multihead_attn(
                query=query,
                key=key_value,
                value=key_value,
                attn_mask=attention_mask
            )

            # Residual connection + norm
            output = self.norm(query + self.dropout(attn_output))

            return output, attn_weights


    class MultiModalFusion(nn.Module):
        """Fuse multiple modalities with learned weights"""
        def __init__(self, dim=768, num_modalities=3):
            super().__init__()
            self.num_modalities = num_modalities

            # Modality-specific transformations
            self.modal_transforms = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim)
                )
                for _ in range(num_modalities)
            ])

            # Learned fusion weights (gating mechanism)
            self.fusion_gate = nn.Sequential(
                nn.Linear(dim * num_modalities, dim),
                nn.GELU(),
                nn.Linear(dim, num_modalities),
                nn.Softmax(dim=-1)
            )

            self.output_proj = nn.Linear(dim, dim)

        def forward(self, modal_embeddings, modal_mask=None):
            """
            Args:
                modal_embeddings: list of [batch, seq, dim] for each modality
                modal_mask: [batch, num_modalities] - which modalities are present

            Returns:
                fused: [batch, seq, dim]
            """
            batch_size = modal_embeddings[0].shape[0]

            # Transform each modality
            transformed = []
            for i, (emb, transform) in enumerate(zip(modal_embeddings, self.modal_transforms)):
                if modal_mask is None or modal_mask[:, i].any():
                    transformed.append(transform(emb))
                else:
                    transformed.append(torch.zeros_like(emb))

            # Compute fusion weights
            # Concatenate all modalities for gate input
            concat_emb = torch.cat([t.mean(dim=1) for t in transformed], dim=-1)  # [batch, dim * num_mod]
            fusion_weights = self.fusion_gate(concat_emb)  # [batch, num_modalities]

            # Weighted sum of modalities
            fused = torch.zeros_like(transformed[0])
            for i, t_emb in enumerate(transformed):
                weight = fusion_weights[:, i:i+1].unsqueeze(1)  # [batch, 1, 1]
                fused = fused + weight * t_emb

            # Output projection
            output = self.output_proj(fused)

            return output, fusion_weights


    class MultiModalTransformer(nn.Module):
        """Complete multi-modal transformer with cross-attention and fusion"""
        def __init__(
            self,
            dim=768,
            num_layers=6,
            num_heads=12,
            num_modalities=3
        ):
            super().__init__()

            # Cross-attention layers (each modality attends to others)
            self.cross_attn_layers = nn.ModuleList([
                CrossModalAttention(dim, num_heads)
                for _ in range(num_layers)
            ])

            # Self-attention layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                batch_first=True
            )
            self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Fusion layer
            self.fusion = MultiModalFusion(dim, num_modalities)

        def forward(self, vision_emb=None, audio_emb=None, text_emb=None):
            """
            Process and fuse multiple modalities

            Returns:
                fused_emb: [batch, seq, dim] - unified multi-modal representation
                attention_weights: dict - attention weights for interpretability
            """
            modalities = []
            modal_names = []

            if vision_emb is not None:
                modalities.append(vision_emb)
                modal_names.append("vision")
            if audio_emb is not None:
                modalities.append(audio_emb)
                modal_names.append("audio")
            if text_emb is not None:
                modalities.append(text_emb)
                modal_names.append("text")

            if len(modalities) == 0:
                raise ValueError("At least one modality required")

            # Apply cross-attention between modalities
            attention_weights = {}
            processed_modalities = list(modalities)

            for layer in self.cross_attn_layers:
                new_modalities = []
                for i, query_mod in enumerate(processed_modalities):
                    # Attend to all other modalities
                    key_value = torch.cat(
                        [processed_modalities[j] for j in range(len(processed_modalities)) if j != i],
                        dim=1
                    )
                    output, attn_w = layer(query_mod, key_value)
                    new_modalities.append(output)
                    attention_weights[f"{modal_names[i]}_layer"] = attn_w

                processed_modalities = new_modalities

            # Fuse modalities
            fused_emb, fusion_weights = self.fusion(processed_modalities)
            attention_weights['fusion_weights'] = fusion_weights

            # Final self-attention
            output = self.self_attn(fused_emb)

            return output, attention_weights

    # Usage
    multi_modal_transformer = MultiModalTransformer(
        dim=768,
        num_layers=6,
        num_heads=12,
        num_modalities=3
    )

    # Process multi-modal input
    vision_emb = torch.randn(1, 32, 768)  # Vision embedding from Q-Former
    audio_emb = torch.randn(1, 16, 768)   # Audio embedding
    text_emb = torch.randn(1, 50, 768)    # Text embedding

    fused_output, attn_weights = multi_modal_transformer(
        vision_emb=vision_emb,
        audio_emb=audio_emb,
        text_emb=text_emb
    )

    print(f"Fused output shape: {fused_output.shape}")
    print(f"Fusion weights: {attn_weights['fusion_weights']}")
    ```

    ---

    ## 3.6 Streaming Multi-Modal Responses

    ### Multi-Modal Language Model with Streaming

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from typing import Iterator
    import asyncio
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import json

    class MultiModalLLM:
        """Multi-modal language model with streaming support"""
        def __init__(
            self,
            model_name="meta-llama/Llama-2-7b-chat-hf",
            multi_modal_transformer=None
        ):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()

            self.multi_modal_transformer = multi_modal_transformer

        def prepare_multi_modal_input(
            self,
            text_prompt,
            vision_emb=None,
            audio_emb=None
        ):
            """Prepare input with multi-modal prefix"""
            # Tokenize text
            text_tokens = self.tokenizer(
                text_prompt,
                return_tensors="pt",
                add_special_tokens=True
            ).input_ids.to(self.device)

            # Get text embeddings
            text_emb = self.model.get_input_embeddings()(text_tokens)

            if vision_emb is None and audio_emb is None:
                return text_emb

            # Process multi-modal inputs
            with torch.no_grad():
                fused_emb, _ = self.multi_modal_transformer(
                    vision_emb=vision_emb,
                    audio_emb=audio_emb,
                    text_emb=text_emb
                )

            return fused_emb

        def generate_stream(
            self,
            text_prompt,
            vision_emb=None,
            audio_emb=None,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9
        ) -> Iterator[str]:
            """
            Generate tokens in streaming fashion

            Yields:
                token: str - each generated token
            """
            # Prepare input embeddings
            input_emb = self.prepare_multi_modal_input(
                text_prompt,
                vision_emb=vision_emb,
                audio_emb=audio_emb
            )

            # Initialize generation
            past_key_values = None
            generated_tokens = []

            for _ in range(max_new_tokens):
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        inputs_embeds=input_emb if past_key_values is None else None,
                        input_ids=generated_tokens[-1:] if past_key_values is not None else None,
                        past_key_values=past_key_values,
                        use_cache=True
                    )

                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-p filtering
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(next_token)

                # Decode and yield token
                token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                yield token_str

                # Update past key values for next iteration
                past_key_values = outputs.past_key_values

    # FastAPI streaming endpoint
    app = FastAPI()

    @app.post("/v1/chat/completions/stream")
    async def chat_completion_stream(request: Request):
        """Streaming chat completion endpoint"""
        data = await request.json()

        # Extract request parameters
        messages = data.get("messages", [])
        model = data.get("model", "gpt-4-vision-preview")
        max_tokens = data.get("max_tokens", 500)
        temperature = data.get("temperature", 0.7)

        # Process multi-modal inputs
        vision_emb = None
        audio_emb = None
        text_content = []

        for message in messages:
            if isinstance(message.get("content"), list):
                for content_part in message["content"]:
                    if content_part["type"] == "text":
                        text_content.append(content_part["text"])
                    elif content_part["type"] == "image_url":
                        # Encode image
                        image_url = content_part["image_url"]["url"]
                        # ... encode image to vision_emb
                    elif content_part["type"] == "audio":
                        # Encode audio
                        audio_url = content_part["audio"]["url"]
                        # ... encode audio to audio_emb
            else:
                text_content.append(message["content"])

        text_prompt = "\n".join(text_content)

        # Create streaming generator
        async def generate():
            """SSE event generator"""
            request_id = f"chatcmpl-{torch.randint(0, 1000000, (1,)).item()}"

            for token in llm.generate_stream(
                text_prompt,
                vision_emb=vision_emb,
                audio_emb=audio_emb,
                max_new_tokens=max_tokens,
                temperature=temperature
            ):
                # Format as SSE
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }

                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

            # Send final chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # Initialize models
    multi_modal_transformer = MultiModalTransformer()
    llm = MultiModalLLM(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        multi_modal_transformer=multi_modal_transformer
    )
    ```

    ---

    ## 3.7 Cross-Modal Search & Retrieval

    ### Vector Similarity Search

    ```python
    from typing import List, Dict
    import numpy as np
    import faiss
    import pickle

    class CrossModalSearchEngine:
        """Vector similarity search across modalities"""
        def __init__(self, embedding_dim=768):
            self.embedding_dim = embedding_dim

            # Separate FAISS indices for each modality
            self.indices = {
                'image': faiss.IndexFlatIP(embedding_dim),  # Inner product (cosine sim)
                'video': faiss.IndexFlatIP(embedding_dim),
                'audio': faiss.IndexFlatIP(embedding_dim),
                'text': faiss.IndexFlatIP(embedding_dim)
            }

            # Metadata storage (maps index ID to metadata)
            self.metadata = {
                'image': {},
                'video': {},
                'audio': {},
                'text': {}
            }

            # ID counters
            self.id_counters = {
                'image': 0,
                'video': 0,
                'audio': 0,
                'text': 0
            }

        def normalize_embedding(self, embedding):
            """L2 normalize for cosine similarity"""
            norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
            return embedding / (norm + 1e-8)

        def add_embedding(self, modality: str, embedding: np.ndarray, metadata: Dict):
            """Add embedding to index"""
            # Normalize
            embedding = self.normalize_embedding(embedding)

            # Add to FAISS index
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)

            self.indices[modality].add(embedding.astype('float32'))

            # Store metadata
            current_id = self.id_counters[modality]
            self.metadata[modality][current_id] = metadata
            self.id_counters[modality] += 1

            return current_id

        def search(
            self,
            query_embedding: np.ndarray,
            search_modalities: List[str],
            top_k: int = 10,
            threshold: float = 0.0
        ) -> List[Dict]:
            """
            Search across specified modalities

            Returns:
                results: list of {modality, id, score, metadata}
            """
            query_embedding = self.normalize_embedding(query_embedding)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            all_results = []

            for modality in search_modalities:
                if modality not in self.indices:
                    continue

                # Search in FAISS index
                scores, indices = self.indices[modality].search(
                    query_embedding.astype('float32'),
                    top_k
                )

                # Collect results
                for score, idx in zip(scores[0], indices[0]):
                    if score > threshold and idx != -1:
                        result = {
                            'modality': modality,
                            'id': int(idx),
                            'score': float(score),
                            'metadata': self.metadata[modality].get(idx, {})
                        }
                        all_results.append(result)

            # Sort by score descending
            all_results.sort(key=lambda x: x['score'], reverse=True)

            return all_results[:top_k]

        def search_multi_modal(
            self,
            query_embeddings: Dict[str, np.ndarray],
            search_modalities: List[str],
            top_k: int = 10,
            modality_weights: Dict[str, float] = None
        ) -> List[Dict]:
            """
            Multi-modal query (e.g., text + image query)

            Args:
                query_embeddings: {'text': emb1, 'image': emb2}
                search_modalities: modalities to search in
                modality_weights: importance weights for each query modality
            """
            if modality_weights is None:
                modality_weights = {k: 1.0 for k in query_embeddings.keys()}

            # Normalize weights
            total_weight = sum(modality_weights.values())
            modality_weights = {k: v / total_weight for k, v in modality_weights.items()}

            # Get results for each query modality
            all_results = {}
            for query_mod, query_emb in query_embeddings.items():
                results = self.search(
                    query_emb,
                    search_modalities,
                    top_k=top_k * 2  # Get more results for merging
                )

                # Weight scores
                weight = modality_weights.get(query_mod, 1.0)
                for result in results:
                    result['score'] *= weight

                # Merge results by ID
                for result in results:
                    key = (result['modality'], result['id'])
                    if key in all_results:
                        all_results[key]['score'] += result['score']
                    else:
                        all_results[key] = result

            # Sort and return top-k
            merged_results = list(all_results.values())
            merged_results.sort(key=lambda x: x['score'], reverse=True)

            return merged_results[:top_k]

        def save(self, path: str):
            """Save indices and metadata"""
            # Save FAISS indices
            for modality, index in self.indices.items():
                faiss.write_index(index, f"{path}_{modality}.faiss")

            # Save metadata
            with open(f"{path}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_counters': self.id_counters
                }, f)

        def load(self, path: str):
            """Load indices and metadata"""
            # Load FAISS indices
            for modality in self.indices.keys():
                try:
                    self.indices[modality] = faiss.read_index(f"{path}_{modality}.faiss")
                except:
                    pass

            # Load metadata
            with open(f"{path}_metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_counters = data['id_counters']

    # Usage
    search_engine = CrossModalSearchEngine(embedding_dim=768)

    # Index some embeddings
    image_emb = np.random.randn(768)
    search_engine.add_embedding(
        'image',
        image_emb,
        metadata={'url': 'https://example.com/img1.jpg', 'user_id': 'user-123'}
    )

    video_emb = np.random.randn(768)
    search_engine.add_embedding(
        'video',
        video_emb,
        metadata={'url': 'https://example.com/vid1.mp4', 'duration': 120}
    )

    # Search with text query
    text_query_emb = np.random.randn(768)
    results = search_engine.search(
        text_query_emb,
        search_modalities=['image', 'video'],
        top_k=5
    )

    print("Search results:")
    for result in results:
        print(f"{result['modality']}: {result['score']:.3f} - {result['metadata']}")

    # Multi-modal query (text + image)
    multi_results = search_engine.search_multi_modal(
        query_embeddings={
            'text': text_query_emb,
            'image': image_emb
        },
        search_modalities=['image', 'video'],
        top_k=5,
        modality_weights={'text': 0.6, 'image': 0.4}
    )
    ```

---

=== "📈 Step 4: Scale & Optimize"

    ## 4.1 Modal-Specific Caching Strategy

    ### Multi-Tier Caching

    ```python
    import hashlib
    import redis
    from typing import Optional
    import numpy as np

    class ModalCache:
        """Multi-tier caching for multi-modal embeddings"""
        def __init__(self, redis_host='localhost', redis_port=6379):
            # Redis for hot cache (recent embeddings)
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=False
            )

            # In-memory cache for very hot items
            self.memory_cache = {}
            self.memory_cache_size = 10000

            # TTL by modality (seconds)
            self.ttl = {
                'image': 3600,      # 1 hour
                'video': 7200,      # 2 hours (more expensive to recompute)
                'audio': 3600,      # 1 hour
                'text': 1800        # 30 minutes
            }

        def compute_content_hash(self, content: bytes, modality: str) -> str:
            """Content-addressable hash for deduplication"""
            hasher = hashlib.sha256()
            hasher.update(content)
            hasher.update(modality.encode())
            return hasher.hexdigest()

        def get_embedding(
            self,
            content_hash: str,
            modality: str
        ) -> Optional[np.ndarray]:
            """Retrieve cached embedding"""
            # Check memory cache first
            cache_key = f"{modality}:{content_hash}"
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]

            # Check Redis
            redis_key = f"emb:{modality}:{content_hash}"
            cached = self.redis_client.get(redis_key)

            if cached is not None:
                embedding = np.frombuffer(cached, dtype=np.float32)

                # Promote to memory cache
                if len(self.memory_cache) < self.memory_cache_size:
                    self.memory_cache[cache_key] = embedding

                return embedding

            return None

        def set_embedding(
            self,
            content_hash: str,
            modality: str,
            embedding: np.ndarray
        ):
            """Cache embedding"""
            cache_key = f"{modality}:{content_hash}"
            redis_key = f"emb:{modality}:{content_hash}"

            # Store in Redis with TTL
            self.redis_client.setex(
                redis_key,
                self.ttl[modality],
                embedding.astype(np.float32).tobytes()
            )

            # Store in memory cache
            if len(self.memory_cache) < self.memory_cache_size:
                self.memory_cache[cache_key] = embedding

        def get_or_encode(
            self,
            content: bytes,
            modality: str,
            encoder_func
        ) -> np.ndarray:
            """Get from cache or encode if not present"""
            content_hash = self.compute_content_hash(content, modality)

            # Try cache first
            cached_emb = self.get_embedding(content_hash, modality)
            if cached_emb is not None:
                return cached_emb

            # Encode and cache
            embedding = encoder_func(content)
            self.set_embedding(content_hash, modality, embedding)

            return embedding

        def clear_memory_cache(self):
            """Clear in-memory cache"""
            self.memory_cache.clear()

        def get_cache_stats(self):
            """Get cache statistics"""
            return {
                'memory_cache_size': len(self.memory_cache),
                'redis_keys': self.redis_client.dbsize()
            }

    # Usage
    cache = ModalCache()

    def encode_image(image_bytes):
        # Simulate expensive encoding
        return np.random.randn(768).astype(np.float32)

    # First call: encodes
    image_data = b"image_binary_data"
    emb1 = cache.get_or_encode(image_data, 'image', encode_image)

    # Second call: cached
    emb2 = cache.get_or_encode(image_data, 'image', encode_image)

    print(f"Cache stats: {cache.get_cache_stats()}")
    ```

    ---

    ## 4.2 GPU Optimization & Batching

    ### Dynamic Batching by Modality

    ```python
    import asyncio
    from collections import defaultdict
    from typing import List, Tuple
    import time

    class DynamicBatcher:
        """Dynamic batching for multi-modal GPU inference"""
        def __init__(
            self,
            max_batch_size=32,
            max_wait_ms=50,
            modal_batch_sizes=None
        ):
            self.max_batch_size = max_batch_size
            self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds

            # Different batch sizes for different modalities
            self.modal_batch_sizes = modal_batch_sizes or {
                'image': 32,
                'video_frame': 64,
                'audio': 16,
                'text': 64
            }

            # Queues for each modality
            self.queues = defaultdict(list)
            self.queue_times = defaultdict(float)

            # Processing locks
            self.locks = defaultdict(asyncio.Lock)

        async def add_to_batch(
            self,
            modality: str,
            data,
            encoder_func
        ):
            """Add item to batch queue and wait for result"""
            # Create future for this item
            future = asyncio.Future()

            # Add to queue
            self.queues[modality].append((data, future))

            # Set queue time if first item
            if len(self.queues[modality]) == 1:
                self.queue_times[modality] = time.time()

            # Check if we should process batch
            should_process = (
                len(self.queues[modality]) >= self.modal_batch_sizes[modality] or
                (time.time() - self.queue_times[modality]) >= self.max_wait_ms
            )

            if should_process:
                # Try to acquire lock and process
                if not self.locks[modality].locked():
                    asyncio.create_task(self._process_batch(modality, encoder_func))

            # Wait for result
            return await future

        async def _process_batch(self, modality: str, encoder_func):
            """Process a batch of items"""
            async with self.locks[modality]:
                if not self.queues[modality]:
                    return

                # Get batch
                batch_size = self.modal_batch_sizes[modality]
                batch = self.queues[modality][:batch_size]
                self.queues[modality] = self.queues[modality][batch_size:]

                # Reset queue time if items remaining
                if self.queues[modality]:
                    self.queue_times[modality] = time.time()

                # Extract data and futures
                data_batch = [item[0] for item in batch]
                futures = [item[1] for item in batch]

                try:
                    # Run encoder (should be async or run in executor)
                    results = await asyncio.to_thread(encoder_func, data_batch)

                    # Set results
                    for future, result in zip(futures, results):
                        future.set_result(result)

                except Exception as e:
                    # Set exception for all futures
                    for future in futures:
                        future.set_exception(e)

    # Usage with FastAPI
    from fastapi import FastAPI, UploadFile

    app = FastAPI()
    batcher = DynamicBatcher()
    vision_encoder = VisionEncoder()

    def encode_image_batch(images: List[Image.Image]):
        """Batch encode images"""
        return vision_encoder.encode_batch(images)

    @app.post("/encode/image")
    async def encode_image_endpoint(file: UploadFile):
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Add to batch (will wait for batch to fill or timeout)
        embedding = await batcher.add_to_batch(
            'image',
            image,
            encode_image_batch
        )

        return {'embedding': embedding.tolist()}
    ```

    ### GPU Memory Pool Management

    ```python
    import torch
    from contextlib import contextmanager

    class GPUMemoryPool:
        """Manage GPU memory allocation for different models"""
        def __init__(self):
            self.allocated = {}
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        @contextmanager
        def allocate(self, model_name: str, memory_mb: int):
            """Context manager for GPU memory allocation"""
            try:
                # Reserve memory
                if model_name not in self.allocated:
                    # Allocate buffer
                    buffer_size = memory_mb * 1024 * 1024 // 4  # Convert to float32 elements
                    buffer = torch.empty(buffer_size, dtype=torch.float32, device=self.device)
                    self.allocated[model_name] = buffer

                yield

            finally:
                # Clear cache after use
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def clear_all(self):
            """Clear all allocated memory"""
            self.allocated.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def get_memory_stats(self):
            """Get current memory usage"""
            if not torch.cuda.is_available():
                return {}

            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }

    # Usage
    gpu_pool = GPUMemoryPool()

    with gpu_pool.allocate('vision_encoder', memory_mb=2000):
        # Run vision encoding
        embeddings = vision_encoder.encode_batch(images)

    with gpu_pool.allocate('llm', memory_mb=15000):
        # Run language generation
        output = llm.generate(prompt)

    print(f"GPU stats: {gpu_pool.get_memory_stats()}")
    ```

    ---

    ## 4.3 Content-Addressable Media Storage

    ### Deduplication & Storage Optimization

    ```python
    import boto3
    from typing import Optional
    import hashlib

    class ContentAddressableStorage:
        """S3-based content-addressable storage with deduplication"""
        def __init__(self, bucket_name: str, prefix: str = "media"):
            self.s3_client = boto3.client('s3')
            self.bucket_name = bucket_name
            self.prefix = prefix

            # DynamoDB for metadata
            self.dynamodb = boto3.resource('dynamodb')
            self.metadata_table = self.dynamodb.Table('media-metadata')

        def compute_sha256(self, content: bytes) -> str:
            """Compute SHA-256 hash of content"""
            return hashlib.sha256(content).hexdigest()

        def get_s3_key(self, content_hash: str, modality: str) -> str:
            """Generate S3 key from content hash"""
            # Shard by first 2 chars of hash for better distribution
            shard = content_hash[:2]
            return f"{self.prefix}/{modality}/{shard}/{content_hash}"

        def upload(
            self,
            content: bytes,
            modality: str,
            metadata: dict
        ) -> Tuple[str, bool]:
            """
            Upload content with deduplication

            Returns:
                (content_hash, was_uploaded)
            """
            # Compute content hash
            content_hash = self.compute_sha256(content)

            # Check if already exists
            s3_key = self.get_s3_key(content_hash, modality)

            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
                # Already exists, just update metadata reference count
                self._increment_reference(content_hash)
                return content_hash, False

            except self.s3_client.exceptions.NoSuchKey:
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=content,
                    ContentType=self._get_content_type(modality),
                    StorageClass='INTELLIGENT_TIERING'  # Auto-optimize storage class
                )

                # Store metadata
                self._store_metadata(content_hash, modality, metadata, s3_key)

                return content_hash, True

        def download(self, content_hash: str, modality: str) -> bytes:
            """Download content by hash"""
            s3_key = self.get_s3_key(content_hash, modality)

            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            return response['Body'].read()

        def get_cdn_url(self, content_hash: str, modality: str) -> str:
            """Generate CloudFront CDN URL"""
            s3_key = self.get_s3_key(content_hash, modality)
            cdn_domain = "d123456abcdef.cloudfront.net"
            return f"https://{cdn_domain}/{s3_key}"

        def _store_metadata(
            self,
            content_hash: str,
            modality: str,
            metadata: dict,
            s3_key: str
        ):
            """Store metadata in DynamoDB"""
            self.metadata_table.put_item(
                Item={
                    'content_hash': content_hash,
                    'modality': modality,
                    's3_key': s3_key,
                    'upload_time': int(time.time()),
                    'reference_count': 1,
                    **metadata
                }
            )

        def _increment_reference(self, content_hash: str):
            """Increment reference count for deduplicated content"""
            self.metadata_table.update_item(
                Key={'content_hash': content_hash},
                UpdateExpression='ADD reference_count :inc',
                ExpressionAttributeValues={':inc': 1}
            )

        def _get_content_type(self, modality: str) -> str:
            """Get MIME type for modality"""
            content_types = {
                'image': 'image/jpeg',
                'video': 'video/mp4',
                'audio': 'audio/mpeg',
                'text': 'text/plain'
            }
            return content_types.get(modality, 'application/octet-stream')

    # Usage
    storage = ContentAddressableStorage(bucket_name='my-multimodal-bucket')

    # Upload image (will deduplicate if already exists)
    with open('image.jpg', 'rb') as f:
        image_data = f.read()

    content_hash, was_uploaded = storage.upload(
        image_data,
        'image',
        metadata={'user_id': 'user-123', 'width': 1920, 'height': 1080}
    )

    print(f"Content hash: {content_hash}")
    print(f"Was uploaded: {was_uploaded}")
    print(f"CDN URL: {storage.get_cdn_url(content_hash, 'image')}")
    ```

    ---

    ## 4.4 Monitoring & Observability

    ### Multi-Modal Metrics

    ```python
    from prometheus_client import Counter, Histogram, Gauge
    import time

    class MultiModalMetrics:
        """Prometheus metrics for multi-modal system"""

        def __init__(self):
            # Request metrics
            self.requests_total = Counter(
                'multimodal_requests_total',
                'Total requests by modality',
                ['modality', 'endpoint']
            )

            self.request_duration = Histogram(
                'multimodal_request_duration_seconds',
                'Request duration by modality',
                ['modality', 'stage'],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
            )

            # Encoding metrics
            self.encoding_duration = Histogram(
                'multimodal_encoding_duration_seconds',
                'Encoding duration by modality',
                ['modality'],
                buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
            )

            self.batch_size = Histogram(
                'multimodal_batch_size',
                'Batch sizes used for encoding',
                ['modality'],
                buckets=(1, 2, 4, 8, 16, 32, 64, 128)
            )

            # Cache metrics
            self.cache_hits = Counter(
                'multimodal_cache_hits_total',
                'Cache hits by modality',
                ['modality', 'cache_tier']
            )

            self.cache_misses = Counter(
                'multimodal_cache_misses_total',
                'Cache misses by modality',
                ['modality']
            )

            # GPU metrics
            self.gpu_utilization = Gauge(
                'multimodal_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id', 'model']
            )

            self.gpu_memory_used = Gauge(
                'multimodal_gpu_memory_used_bytes',
                'GPU memory used in bytes',
                ['gpu_id']
            )

            # Generation metrics
            self.tokens_generated = Counter(
                'multimodal_tokens_generated_total',
                'Total tokens generated',
                ['model']
            )

            self.generation_latency = Histogram(
                'multimodal_generation_latency_seconds',
                'Time to first token and total generation time',
                ['model', 'metric_type'],  # metric_type: ttft or total
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
            )

            # Error metrics
            self.errors_total = Counter(
                'multimodal_errors_total',
                'Total errors by type',
                ['error_type', 'modality']
            )

        def track_request(self, modality: str, endpoint: str):
            """Track a request"""
            self.requests_total.labels(modality=modality, endpoint=endpoint).inc()

        def track_encoding(self, modality: str, duration: float, batch_size: int):
            """Track encoding operation"""
            self.encoding_duration.labels(modality=modality).observe(duration)
            self.batch_size.labels(modality=modality).observe(batch_size)

        def track_cache_access(self, modality: str, hit: bool, tier: str = 'redis'):
            """Track cache access"""
            if hit:
                self.cache_hits.labels(modality=modality, cache_tier=tier).inc()
            else:
                self.cache_misses.labels(modality=modality).inc()

        def update_gpu_metrics(self, gpu_id: int, utilization: float, memory_used: int, model: str):
            """Update GPU metrics"""
            self.gpu_utilization.labels(gpu_id=str(gpu_id), model=model).set(utilization)
            self.gpu_memory_used.labels(gpu_id=str(gpu_id)).set(memory_used)

        def track_generation(self, model: str, ttft: float, total_time: float, num_tokens: int):
            """Track generation metrics"""
            self.generation_latency.labels(model=model, metric_type='ttft').observe(ttft)
            self.generation_latency.labels(model=model, metric_type='total').observe(total_time)
            self.tokens_generated.labels(model=model).inc(num_tokens)

        def track_error(self, error_type: str, modality: str):
            """Track an error"""
            self.errors_total.labels(error_type=error_type, modality=modality).inc()

    # Usage in request handler
    metrics = MultiModalMetrics()

    async def process_multimodal_request(text, image, audio):
        start_time = time.time()

        # Track request
        modality = "text+image+audio"
        metrics.track_request(modality, "/v1/chat/completions")

        try:
            # Encode image
            if image:
                encode_start = time.time()
                image_emb = await encode_image_with_cache(image)
                metrics.track_encoding('image', time.time() - encode_start, 1)

            # Encode audio
            if audio:
                encode_start = time.time()
                audio_emb = await encode_audio_with_cache(audio)
                metrics.track_encoding('audio', time.time() - encode_start, 1)

            # Generate response
            gen_start = time.time()
            first_token_time = None

            async for token in generate_stream(text, image_emb, audio_emb):
                if first_token_time is None:
                    first_token_time = time.time()
                yield token

            # Track generation metrics
            ttft = first_token_time - gen_start
            total_time = time.time() - gen_start
            metrics.track_generation('gpt-4v', ttft, total_time, num_tokens=100)

        except Exception as e:
            metrics.track_error(type(e).__name__, modality)
            raise

        finally:
            duration = time.time() - start_time
            metrics.request_duration.labels(modality=modality, stage='total').observe(duration)
    ```

    ---

    ## 4.5 Load Balancing & Auto-Scaling

    ### GPU Pool Load Balancer

    ```python
    import random
    from typing import List, Dict
    import asyncio

    class GPUPoolLoadBalancer:
        """Load balancer for GPU inference pools"""

        def __init__(self):
            self.gpu_nodes = {}  # node_id -> node_info
            self.health_checks = {}  # node_id -> last_health_check_time
            self.metrics = {}  # node_id -> metrics

        def register_node(
            self,
            node_id: str,
            endpoint: str,
            gpu_type: str,
            capabilities: List[str]
        ):
            """Register a GPU node"""
            self.gpu_nodes[node_id] = {
                'endpoint': endpoint,
                'gpu_type': gpu_type,
                'capabilities': capabilities,
                'active_requests': 0,
                'total_requests': 0,
                'healthy': True
            }

        def select_node(self, modality: str, strategy: str = 'least_loaded') -> str:
            """
            Select a GPU node for a request

            Strategies:
            - least_loaded: Choose node with fewest active requests
            - round_robin: Rotate through nodes
            - random: Random selection
            - weighted: Based on GPU capability
            """
            # Filter nodes by capability
            capable_nodes = [
                node_id for node_id, info in self.gpu_nodes.items()
                if modality in info['capabilities'] and info['healthy']
            ]

            if not capable_nodes:
                raise RuntimeError(f"No healthy nodes available for {modality}")

            if strategy == 'least_loaded':
                return min(
                    capable_nodes,
                    key=lambda n: self.gpu_nodes[n]['active_requests']
                )

            elif strategy == 'round_robin':
                # Use total_requests as counter
                return min(
                    capable_nodes,
                    key=lambda n: self.gpu_nodes[n]['total_requests']
                )

            elif strategy == 'random':
                return random.choice(capable_nodes)

            elif strategy == 'weighted':
                # Prefer more powerful GPUs (A100 > V100 > T4)
                gpu_weights = {'A100': 3, 'V100': 2, 'T4': 1}
                weighted_nodes = [
                    (n, gpu_weights.get(self.gpu_nodes[n]['gpu_type'], 1))
                    for n in capable_nodes
                ]
                return random.choices(
                    [n for n, w in weighted_nodes],
                    weights=[w for n, w in weighted_nodes]
                )[0]

        async def send_request(self, node_id: str, request_data: Dict):
            """Send request to GPU node"""
            node = self.gpu_nodes[node_id]
            node['active_requests'] += 1
            node['total_requests'] += 1

            try:
                # Send HTTP request to node
                # response = await http_client.post(node['endpoint'], json=request_data)
                # return response

                # Simulate for demo
                await asyncio.sleep(0.1)
                return {'result': 'success'}

            finally:
                node['active_requests'] -= 1

        async def health_check(self, node_id: str):
            """Check node health"""
            try:
                # Send health check request
                # response = await http_client.get(f"{node['endpoint']}/health")
                # healthy = response.status_code == 200

                # Simulate for demo
                healthy = True

                self.gpu_nodes[node_id]['healthy'] = healthy
                self.health_checks[node_id] = time.time()

            except Exception:
                self.gpu_nodes[node_id]['healthy'] = False

        async def health_check_loop(self, interval: int = 30):
            """Periodically check all nodes"""
            while True:
                for node_id in list(self.gpu_nodes.keys()):
                    await self.health_check(node_id)

                await asyncio.sleep(interval)

        def get_pool_stats(self) -> Dict:
            """Get statistics about the GPU pool"""
            total_nodes = len(self.gpu_nodes)
            healthy_nodes = sum(1 for n in self.gpu_nodes.values() if n['healthy'])
            total_active_requests = sum(n['active_requests'] for n in self.gpu_nodes.values())

            return {
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_nodes,
                'total_active_requests': total_active_requests,
                'nodes': {
                    node_id: {
                        'healthy': info['healthy'],
                        'active_requests': info['active_requests'],
                        'total_requests': info['total_requests']
                    }
                    for node_id, info in self.gpu_nodes.items()
                }
            }

    # Usage
    load_balancer = GPUPoolLoadBalancer()

    # Register GPU nodes
    load_balancer.register_node(
        'gpu-node-1',
        'http://gpu-node-1:8000',
        'A100',
        ['image', 'video', 'audio', 'text']
    )
    load_balancer.register_node(
        'gpu-node-2',
        'http://gpu-node-2:8000',
        'V100',
        ['image', 'audio', 'text']
    )

    # Select node for request
    node_id = load_balancer.select_node('image', strategy='least_loaded')
    print(f"Selected node: {node_id}")

    # Send request
    result = await load_balancer.send_request(node_id, {'image': '...'})

    # Get pool stats
    stats = load_balancer.get_pool_stats()
    print(f"Pool stats: {stats}")
    ```

    ---

    ## Architecture Diagram: Scaled System

    ```mermaid
    graph TB
        subgraph "Edge Layer"
            CDN[CDN<br/>CloudFront]
            WAF[WAF<br/>DDoS Protection]
        end

        subgraph "API Layer (Multi-AZ)"
            LB1[Load Balancer<br/>ALB]
            API1[API Gateway 1]
            API2[API Gateway 2]
            API3[API Gateway N]
        end

        subgraph "Processing Layer"
            Router[Request Router<br/>Modal Detection]

            subgraph "Image Pipeline"
                ImgQ[Image Queue<br/>SQS]
                ImgProc1[Image Processor 1]
                ImgProc2[Image Processor N]
            end

            subgraph "Video Pipeline"
                VidQ[Video Queue<br/>SQS]
                VidProc1[Video Processor 1]
                VidProc2[Video Processor N]
            end

            subgraph "Audio Pipeline"
                AudQ[Audio Queue<br/>SQS]
                AudProc1[Audio Processor 1]
                AudProc2[Audio Processor N]
            end
        end

        subgraph "GPU Cluster (Multi-Region)"
            GPULB[GPU Load Balancer]

            subgraph "Vision GPU Pool"
                VisionGPU1[Vision GPU 1<br/>A100 x 8]
                VisionGPU2[Vision GPU N<br/>A100 x 8]
            end

            subgraph "Audio GPU Pool"
                AudioGPU1[Audio GPU 1<br/>A100 x 4]
                AudioGPU2[Audio GPU N<br/>A100 x 4]
            end

            subgraph "LLM GPU Pool"
                LLMGPU1[LLM GPU 1<br/>A100 x 8]
                LLMGPU2[LLM GPU N<br/>A100 x 8]
            end
        end

        subgraph "Caching Layer"
            Redis1[(Redis Cluster 1<br/>Embeddings)]
            Redis2[(Redis Cluster 2<br/>KV Cache)]
            Memcached[(Memcached<br/>Response Cache)]
        end

        subgraph "Storage Layer"
            S3[(S3<br/>Media Storage<br/>Intelligent Tiering)]
            Pinecone[(Pinecone<br/>Vector DB)]
            RDS[(RDS Aurora<br/>Metadata)]
            Cassandra[(Cassandra<br/>Conversations)]
        end

        subgraph "Monitoring"
            Prometheus[Prometheus<br/>Metrics]
            Grafana[Grafana<br/>Dashboards]
            Datadog[Datadog<br/>APM]
        end

        CDN --> WAF
        WAF --> LB1
        LB1 --> API1
        LB1 --> API2
        LB1 --> API3

        API1 --> Router
        API2 --> Router
        API3 --> Router

        Router --> ImgQ
        Router --> VidQ
        Router --> AudQ

        ImgQ --> ImgProc1
        ImgQ --> ImgProc2
        VidQ --> VidProc1
        VidQ --> VidProc2
        AudQ --> AudProc1
        AudQ --> AudProc2

        ImgProc1 --> GPULB
        ImgProc2 --> GPULB
        VidProc1 --> GPULB
        VidProc2 --> GPULB
        AudProc1 --> GPULB
        AudProc2 --> GPULB

        GPULB --> VisionGPU1
        GPULB --> VisionGPU2
        GPULB --> AudioGPU1
        GPULB --> AudioGPU2
        GPULB --> LLMGPU1
        GPULB --> LLMGPU2

        VisionGPU1 --> Redis1
        AudioGPU1 --> Redis1
        LLMGPU1 --> Redis2

        Router --> Redis1
        Router --> Memcached
        Router --> S3
        Router --> Pinecone
        Router --> RDS
        Router --> Cassandra

        API1 -.-> Prometheus
        GPULB -.-> Prometheus
        Prometheus --> Grafana
        API1 -.-> Datadog
    ```

---

## Summary

This multi-modal AI system design demonstrates:

1. **Multi-Modal Processing Pipeline**: Unified architecture for processing text, images, video, and audio inputs
2. **Vision Encoding**: CLIP and ViT-based image encoding with high-resolution support (GPT-4V style)
3. **Video Understanding**: Frame sampling strategies (uniform, keyframes) with temporal encoding
4. **Audio Processing**: Whisper-based transcription and audio embedding generation
5. **Unified Embedding Space**: Q-Former architecture for projecting all modalities to shared space
6. **Cross-Modal Attention**: Multi-head attention and gated fusion for multi-modal reasoning
7. **Streaming Generation**: Real-time token streaming for multi-modal responses (SSE)
8. **Vector Search**: Cross-modal similarity search with FAISS indexing
9. **Scalability**: Modal-specific caching, dynamic batching, GPU pooling, content-addressable storage
10. **Observability**: Comprehensive metrics for multi-modal system monitoring

**Key Technologies**: PyTorch, Transformers, CLIP, Whisper, ViT, FAISS, Redis, S3, FastAPI, SSE

**References**:
- GPT-4V (OpenAI): Multi-modal vision-language model with high-res image understanding
- Gemini (Google): Native multi-modal architecture with unified encoding
- Claude 3 (Anthropic): Vision + text understanding with streaming responses
- BLIP-2: Q-Former architecture for vision-language alignment
- Flamingo: Few-shot multi-modal learning with perceiver resampler
- ImageBind (Meta): Joint embedding space for 6 modalities
