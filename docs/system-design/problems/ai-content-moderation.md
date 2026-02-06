# Design an AI Content Moderation System

A scalable AI-powered content moderation platform that automatically detects and filters harmful content (text, images, videos) including toxicity, hate speech, NSFW material, violence, and spam across social media platforms, supporting real-time and batch processing with human-in-the-loop review.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1B content items/day, 100M users, 500K reports/day, 10K human reviewers |
| **Key Challenges** | Multi-modal classification, real-time detection (<500ms), high accuracy (99.5%), appeal workflow, false positive management, adversarial attacks, model drift |
| **Core Concepts** | BERT toxicity detection, ResNet NSFW classification, video frame sampling, human-in-the-loop, active learning, feedback loops, tiered review, content hashing |
| **Companies** | OpenAI (Moderation API), Google (Perspective API), Meta, TikTok, Discord, Reddit, YouTube |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Text Moderation** | Detect toxicity, hate speech, profanity, threats, sexual content | P0 (Must have) |
    | **Image Moderation** | Detect NSFW, violence, gore, self-harm, inappropriate content | P0 (Must have) |
    | **Video Moderation** | Frame-by-frame analysis + audio transcription moderation | P0 (Must have) |
    | **Real-time Processing** | Moderate content before publishing (<500ms) | P0 (Must have) |
    | **Batch Processing** | Retroactive moderation of existing content | P0 (Must have) |
    | **Human Review Queue** | Escalate uncertain cases (confidence 40-60%) | P0 (Must have) |
    | **Appeal Workflow** | Users can appeal moderation decisions | P0 (Must have) |
    | **Multi-language Support** | Support 50+ languages for text moderation | P1 (Should have) |
    | **Context-aware Moderation** | Consider user history, community context | P1 (Should have) |
    | **Severity Scoring** | Rate content harm level (0-1 score) | P1 (Should have) |
    | **Action Recommendation** | Suggest actions (remove, warn, restrict, allow) | P1 (Should have) |
    | **Duplicate Detection** | Hash-based detection of previously moderated content | P1 (Should have) |
    | **Feedback Loop** | Human decisions retrain models | P1 (Should have) |
    | **Analytics Dashboard** | Moderation metrics, trends, false positive rates | P2 (Nice to have) |

    **Explicitly Out of Scope** (mention in interview):

    - User account management system
    - Content creation/upload infrastructure
    - Payment processing for premium moderation
    - Live streaming moderation in real-time (<100ms)
    - Audio-only content moderation
    - Copyright detection (DMCA)
    - Spam detection (covered separately)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Latency (Real-time)** | < 500ms p95 | Must not delay user experience |
    | **Latency (Batch)** | < 1 hour for full catalog scan | Timely removal of harmful content |
    | **Availability** | 99.95% uptime | Critical for platform safety and compliance |
    | **Accuracy (Precision)** | 99.5% for auto-removal | Minimize false positives |
    | **Accuracy (Recall)** | 95% for harmful content detection | Minimize false negatives |
    | **Throughput** | 1B items/day, 11.5K items/sec average | Handle platform scale |
    | **Human Review Time** | < 30 seconds per item | Efficient reviewer productivity |
    | **Appeal Response** | < 24 hours for decision | User experience and legal compliance |
    | **Scalability** | Support 10x traffic spikes | Viral content, bot attacks |
    | **Model Update Frequency** | Daily retraining, hourly incremental | Combat evolving abuse patterns |
    | **Multi-region** | < 100ms latency globally | Serve international users |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Daily Active Users (DAU): 100M
    Monthly Active Users (MAU): 200M

    Content submissions:
    - Text posts: 500M/day (comments, posts, messages)
    - Images: 400M/day
    - Videos: 100M/day
    - Total: 1B content items/day
    - Average QPS: 1B / 86,400 = ~11,574 items/sec
    - Peak QPS: 3x average = ~35,000 items/sec

    Moderation breakdown:
    - Real-time (pre-publish): 70% = 700M items/day = ~8,100 QPS
    - Batch (post-publish): 30% = 300M items/day
    - User reports: 500K/day = ~6 reports/sec

    Detection rates:
    - Auto-approve (high confidence safe): 85% = 850M items/day
    - Auto-reject (high confidence harmful): 10% = 100M items/day
    - Human review (uncertain): 5% = 50M items/day
    - Human review QPS: 50M / 86,400 = ~578 items/sec

    Human reviewers:
    - Review rate: 200 items/hour/reviewer = 3.3 items/min
    - Daily items needing review: 50M
    - Reviewer hours needed: 50M / 200 = 250K hours/day
    - 8-hour shifts: 250K / 8 = 31,250 reviewers needed
    - With 20% overhead + training: ~37,500 reviewers
    - Actual reviewers (distributed globally): 10K (assume better ML reduces queue)

    Appeals:
    - Appeal rate: 1% of auto-rejected = 1M appeals/day
    - Appeal QPS: 1M / 86,400 = ~11.5 appeals/sec
    - Appeal review time: 2 minutes/appeal
    - Specialized appeal reviewers: 500

    Model inference:
    - Text classification: 500M/day = 5,787 QPS
    - Image classification: 400M/day = 4,630 QPS
    - Video analysis: 100M/day = 1,157 QPS
    - Total ML inference: 11,574 QPS

    Read/Write ratio: 1:20 (write-heavy for moderation events)
    ```

    ### Storage Estimates

    ```
    Content metadata:
    - Per item: 500 bytes (content_id, user_id, type, timestamps, flags)
    - 1B items/day × 500 bytes = 500 GB/day
    - 1 year retention: 500 GB × 365 = 182 TB

    Moderation results:
    - Per result: 1 KB (scores, labels, confidence, action, reviewer_id)
    - 1B results/day × 1 KB = 1 TB/day
    - 3 years retention (compliance): 1 TB × 365 × 3 = 1 PB

    Content hashes (duplicate detection):
    - Per hash: 32 bytes (SHA-256) + 100 bytes metadata
    - 1B items × 132 bytes = 132 GB/day
    - 1 year retention: 132 GB × 365 = 48 TB

    Human review queue:
    - Active queue: 50M items × 2 KB = 100 GB
    - Historical reviews: 50M/day × 5 KB × 365 = 91 TB/year

    Appeal records:
    - 1M appeals/day × 3 KB = 3 GB/day
    - 3 years: 3 GB × 365 × 3 = 3.3 TB

    Model artifacts:
    - Text models (BERT-based): 500 MB
    - Image models (ResNet50): 100 MB
    - Video models: 200 MB
    - Multi-language models: 50 × 500 MB = 25 GB
    - Model versions (10 versions): 26 GB × 10 = 260 GB
    - Training data samples: 100 GB

    Feature store:
    - User reputation scores: 200M users × 500 bytes = 100 GB
    - Content embeddings: 1B items × 512 bytes = 512 GB (TTL 30 days)

    Total storage: 182 TB (metadata) + 1 PB (results) + 48 TB (hashes) + 91 TB (reviews) + 3.3 TB (appeals) + 260 GB (models) + 612 GB (features) ≈ 1.3 PB
    ```

    ### Bandwidth Estimates

    ```
    Content ingestion:
    - Text: 5,787 QPS × 2 KB = 11.6 MB/sec = 93 Mbps
    - Images: 4,630 QPS × 200 KB = 926 MB/sec = 7.4 Gbps
    - Videos: 1,157 QPS × 5 MB = 5.8 GB/sec = 46 Gbps

    Model inference:
    - Embeddings lookup: 11,574 QPS × 512 bytes = 5.9 MB/sec = 47 Mbps
    - Feature fetching: 11,574 QPS × 1 KB = 11.6 MB/sec = 93 Mbps

    Results storage:
    - Moderation results: 11,574 QPS × 1 KB = 11.6 MB/sec = 93 Mbps

    Human review interface:
    - Item fetching: 578 QPS × 300 KB = 173 MB/sec = 1.4 Gbps
    - Decision submissions: 578 QPS × 2 KB = 1.2 MB/sec = 9.6 Mbps

    Total ingress: ~54 Gbps (mostly video content)
    Total egress: ~1.5 Gbps (mostly human review)
    ```

    ### Compute Estimates

    ```
    Text moderation (BERT inference):
    - Inference time: 20ms per item (batch size 32)
    - Daily: 500M texts
    - GPU hours: 500M × 20ms / 3600s = 2,778 GPU-hours/day
    - T4 GPUs needed: 2,778 / 24 = ~116 T4 GPUs

    Image moderation (ResNet inference):
    - Inference time: 15ms per image (batch size 64)
    - Daily: 400M images
    - GPU hours: 400M × 15ms / 3600s = 1,667 GPU-hours/day
    - T4 GPUs needed: 1,667 / 24 = ~70 T4 GPUs

    Video moderation:
    - Frame extraction: 10 frames/video = 1B frames/day
    - Frame classification: 1B × 15ms / 3600s = 4,167 GPU-hours/day
    - Audio transcription: 100M videos × 50ms = 139 GPU-hours/day
    - Total: 4,306 GPU-hours/day
    - T4 GPUs: 4,306 / 24 = ~180 T4 GPUs

    Total GPUs: 116 + 70 + 180 = 366 T4 GPUs
    With 30% overhead: 366 × 1.3 = ~476 T4 GPUs

    GPU cost: 476 × $0.35/hour = $167/hour = $4,000/day = $120K/month
    ```

    ---

    ## Key Assumptions

    1. 1% of content is harmful and requires action (10M items/day)
    2. 5% of content is borderline and needs human review (50M items/day)
    3. Multi-language support covers 95% of global user base
    4. Content hashing detects 30% of repeat violations instantly
    5. Human reviewers handle 200 items/hour with 98% accuracy
    6. Appeal rate is 1% of moderated content (manageable volume)
    7. Model retraining occurs daily with human feedback
    8. False positive rate < 0.5% for auto-removal actions
    9. Video analysis via frame sampling (10 frames/video) is sufficient
    10. Real-time moderation prioritized for public content, batch for private

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Multi-modal pipeline:** Unified architecture for text, image, and video
    2. **Tiered decision making:** Auto-approve → Auto-reject → Human review
    3. **Human-in-the-loop:** Continuous feedback improves models
    4. **Low latency:** Pre-publish moderation must not impact UX
    5. **High accuracy:** Minimize false positives (user frustration) and false negatives (harmful content)
    6. **Duplicate detection:** Hash-based caching for previously moderated content

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Content Sources"
            Client[Client Apps]
            ReportAPI[User Reports]
            BatchJob[Batch Scanner]
        end

        subgraph "Ingestion Layer"
            Gateway[API Gateway]
            ContentQueue[Content Queue<br/>Kafka]
            PriorityQueue[Priority Queue<br/>High priority/reports]
        end

        subgraph "Duplicate Detection"
            HashService[Content Hash Service<br/>Perceptual hashing]
            HashCache[(Hash Cache<br/>Redis)]
        end

        subgraph "ML Classification Layer"
            TextClassifier[Text Classifier<br/>BERT fine-tuned]
            ImageClassifier[Image Classifier<br/>ResNet50 NSFW]
            VideoProcessor[Video Processor<br/>Frame sampler + Audio]
            EnsembleService[Ensemble Service<br/>Combines scores]
        end

        subgraph "Decision Engine"
            RulesEngine[Rules Engine<br/>Thresholds + Context]
            ActionService[Action Service<br/>Allow/Remove/Review]
        end

        subgraph "Human Review"
            ReviewQueue[(Review Queue<br/>Priority sorted)]
            ReviewUI[Review Interface]
            Reviewers[Human Reviewers]
            ActiveLearning[Active Learning<br/>Sample uncertain cases]
        end

        subgraph "Appeal System"
            AppealQueue[(Appeal Queue)]
            AppealUI[Appeal Interface]
            AppealReviewers[Appeal Reviewers]
        end

        subgraph "Feedback Loop"
            FeedbackService[Feedback Service]
            TrainingPipeline[Model Retraining<br/>Nightly + Incremental]
            ModelRegistry[Model Registry<br/>A/B testing]
        end

        subgraph "Storage & Analytics"
            ModerationDB[(Moderation DB<br/>Cassandra)]
            FeatureStore[(Feature Store<br/>User reputation)]
            Analytics[Analytics Service<br/>Metrics + Dashboards]
        end

        Client --> Gateway
        ReportAPI --> PriorityQueue
        BatchJob --> ContentQueue
        Gateway --> ContentQueue
        Gateway --> PriorityQueue

        ContentQueue --> HashService
        PriorityQueue --> HashService
        HashService --> HashCache
        HashCache -->|Cache hit| ActionService

        HashService -->|New content| TextClassifier
        HashService --> ImageClassifier
        HashService --> VideoProcessor

        TextClassifier --> EnsembleService
        ImageClassifier --> EnsembleService
        VideoProcessor --> EnsembleService

        EnsembleService --> RulesEngine
        RulesEngine --> ActionService
        RulesEngine --> FeatureStore

        ActionService -->|Auto-approve/reject| ModerationDB
        ActionService -->|Uncertain| ReviewQueue

        ReviewQueue --> ActiveLearning
        ActiveLearning --> ReviewUI
        ReviewUI --> Reviewers
        Reviewers --> FeedbackService

        FeedbackService --> ModerationDB
        FeedbackService --> TrainingPipeline
        TrainingPipeline --> ModelRegistry
        ModelRegistry --> TextClassifier
        ModelRegistry --> ImageClassifier

        ActionService -->|Rejected| AppealQueue
        AppealQueue --> AppealUI
        AppealUI --> AppealReviewers
        AppealReviewers --> FeedbackService

        ModerationDB --> Analytics
    ```

    ---

    ## Component Details

    ### 1. Ingestion Layer

    **API Gateway:**
    - Rate limiting (prevent abuse)
    - Authentication/authorization
    - Content deduplication at edge
    - Route to priority queue for user reports

    **Content Queue (Kafka):**
    - Topic: `moderation.content.text`, `moderation.content.image`, `moderation.content.video`
    - Partitions: 100 (for parallelism)
    - Retention: 7 days
    - Consumer groups: Text/Image/Video classifiers

    **Priority Queue:**
    - User reports get highest priority
    - Viral content (high engagement) gets elevated priority
    - New user content (higher risk) gets priority

    ---

    ### 2. Duplicate Detection

    **Content Hash Service:**
    - **Text:** SHA-256 hash of normalized text (lowercase, remove whitespace)
    - **Image:** Perceptual hash (pHash) - robust to minor edits
    - **Video:** Hash of keyframes + audio fingerprint
    - Check cache before ML inference (30% cache hit rate)

    **Hash Cache (Redis):**
    - Key: `hash:{content_hash}`
    - Value: `{action: "remove", confidence: 0.98, reason: "hate_speech"}`
    - TTL: 30 days (evict old hashes)
    - Cluster: 20 nodes, 500GB memory

    ---

    ### 3. ML Classification Layer

    **Text Classifier (BERT-based):**
    - Model: BERT-base fine-tuned on moderation datasets
    - Categories: Toxicity, hate speech, threats, sexual content, self-harm, spam
    - Input: Text (max 512 tokens)
    - Output: Multi-label scores [0-1] for each category
    - Latency: 20ms (batch size 32)
    - Multi-language: XLM-RoBERTa for 50+ languages

    **Image Classifier (ResNet50):**
    - Model: ResNet50 fine-tuned on NSFW datasets
    - Categories: Nudity, sexual, violence, gore, self-harm
    - Input: 224x224 RGB image
    - Output: Multi-label scores [0-1]
    - Latency: 15ms (batch size 64)
    - Preprocessing: Face detection + blur for privacy

    **Video Processor:**
    - Frame sampling: Extract 10 evenly-spaced frames
    - Frame classification: Run image classifier on each frame
    - Audio transcription: Whisper model → text moderation
    - Shot detection: Detect scene changes for intelligent sampling
    - Aggregation: Max score across frames + audio
    - Latency: 500ms for 30-second video

    **Ensemble Service:**
    - Combines scores from multiple models
    - Weighted average based on model confidence
    - Context-aware: User reputation, community standards
    - Output: Final score [0-1] + category labels

    ---

    ### 4. Decision Engine

    **Rules Engine:**
    - **Auto-approve:** Score < 0.2 (high confidence safe)
    - **Auto-reject:** Score > 0.8 (high confidence harmful)
    - **Human review:** 0.2 ≤ score ≤ 0.8 (uncertain)
    - Severity thresholds: Critical (>0.95) → Immediate removal
    - Context rules: First-time offender vs repeat violator

    **Action Service:**
    - Actions: `allow`, `remove`, `warn`, `shadowban`, `review_queue`
    - User notification: Explain moderation decision
    - Enforcement: Call content service to remove/hide content
    - Audit log: Record all actions for compliance

    **Feature Store:**
    - User reputation score (0-100)
    - Historical violation count
    - Community context (subreddit rules, server guidelines)
    - Real-time features: Time of day, geolocation

    ---

    ### 5. Human Review System

    **Review Queue (PostgreSQL + Redis):**
    - Priority: User reports > high-engagement content > random sampling
    - Assignment: Load balancing across reviewers
    - SLA tracking: Time in queue, review time
    - Consensus: Multiple reviewers for ambiguous cases

    **Review Interface:**
    - Display content with context (user profile, community, history)
    - Side-by-side ML predictions + confidence scores
    - Action buttons: Approve, Remove, Warn, Escalate
    - Reason selection: Predefined categories + free text
    - Batch actions: Review similar content together

    **Active Learning:**
    - Sample high-uncertainty cases (confidence 45-55%)
    - Sample edge cases where models disagree
    - Ensure diverse coverage (all categories, languages)
    - Goal: Maximize model improvement per review

    ---

    ### 6. Appeal System

    **Appeal Workflow:**
    1. User submits appeal with explanation
    2. Appeal enters queue with original content + decision
    3. Specialized appeal reviewers (higher training)
    4. Decision: Uphold, overturn, or escalate to senior reviewer
    5. Notify user within 24 hours
    6. Log outcome for model retraining

    **Appeal Queue:**
    - FIFO with priority for time-sensitive content
    - Automated triage: Simple cases auto-approved
    - Reviewer assignment: Balance workload + expertise

    ---

    ### 7. Feedback Loop

    **Feedback Service:**
    - Collect human decisions (review queue + appeals)
    - Label quality validation: Inter-rater agreement
    - Data augmentation: Paraphrase, translate, crop images
    - Training data pipeline: Kafka → S3 → Training cluster

    **Model Retraining:**
    - **Nightly:** Full model retrain on 30 days of feedback
    - **Incremental:** Hourly updates for emerging abuse patterns
    - **A/B testing:** Shadow mode → 5% → 50% → 100% rollout
    - **Metrics:** Precision, recall, F1, AUC-ROC, false positive rate

    **Model Registry:**
    - Version control: Model versioning + lineage tracking
    - Rollback: Instant rollback if metrics degrade
    - Monitoring: Latency, throughput, accuracy drift

    ---

    ### 8. Storage & Analytics

    **Moderation DB (Cassandra):**
    - Table: `moderation_events` (content_id, timestamp, action, scores)
    - Partition key: `content_id`
    - Retention: 3 years (compliance)
    - Replication: RF=3, multi-region

    **Analytics Service:**
    - Metrics: Moderation rate, false positive/negative rate, appeal overturn rate
    - Dashboards: Real-time moderation volume, reviewer productivity
    - Alerting: Spike detection, model degradation, SLA violations
    - Reports: Monthly compliance reports for regulators

    ---

    ## API Design

    ### Moderate Content API

    **Request:**
    ```http
    POST /v1/moderation/moderate
    Content-Type: application/json

    {
      "content_id": "post_123456",
      "content_type": "text|image|video",
      "content": "Text content or URL to media",
      "user_id": "user_789",
      "context": {
        "community_id": "subreddit_xyz",
        "is_public": true,
        "parent_content_id": "post_111" // for comments
      },
      "priority": "normal|high|urgent"
    }
    ```

    **Response:**
    ```json
    {
      "moderation_id": "mod_abc123",
      "status": "approved|rejected|pending_review",
      "action": "allow|remove|warn|review_queue",
      "scores": {
        "toxicity": 0.05,
        "hate_speech": 0.02,
        "nsfw": 0.01,
        "violence": 0.03,
        "spam": 0.10
      },
      "categories": ["safe"],
      "confidence": 0.95,
      "reason": "Content appears safe",
      "latency_ms": 120,
      "model_version": "v2.3.1",
      "timestamp": "2026-02-05T10:30:00Z"
    }
    ```

    ### Appeal API

    **Request:**
    ```http
    POST /v1/moderation/appeal
    Content-Type: application/json

    {
      "moderation_id": "mod_abc123",
      "content_id": "post_123456",
      "user_id": "user_789",
      "appeal_reason": "This was taken out of context. The phrase was a quote from a book.",
      "appeal_context": {
        "additional_info": "Book title: XYZ"
      }
    }
    ```

    **Response:**
    ```json
    {
      "appeal_id": "appeal_xyz789",
      "status": "pending|approved|rejected",
      "estimated_review_time_hours": 12,
      "queue_position": 42,
      "timestamp": "2026-02-05T10:35:00Z"
    }
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Text Toxicity Detection with BERT

    ### Model Architecture

    ```python
    import torch
    import torch.nn as nn
    from transformers import BertModel, BertTokenizer

    class ToxicityClassifier(nn.Module):
        """
        BERT-based multi-label toxicity classifier
        Categories: toxicity, hate_speech, threat, sexual, self_harm, spam
        """
        def __init__(self, num_labels=6, dropout=0.3):
            super().__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            # BERT encoding
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output  # [CLS] token representation

            # Dropout + classification
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            scores = self.sigmoid(logits)  # Multi-label: [0, 1] for each category

            return scores

    # Initialize model and tokenizer
    model = ToxicityClassifier(num_labels=6)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Category labels
    CATEGORIES = ['toxicity', 'hate_speech', 'threat', 'sexual', 'self_harm', 'spam']
    ```

    ### Inference Pipeline

    ```python
    def moderate_text(text: str, model, tokenizer, device) -> dict:
        """
        Moderate text content and return moderation scores

        Args:
            text: Input text to moderate
            model: Trained toxicity classifier
            tokenizer: BERT tokenizer
            device: torch device (cpu/cuda)

        Returns:
            dict with scores, categories, and action
        """
        # Preprocessing
        text = text.strip()[:512]  # Limit to 512 chars

        # Tokenization
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Inference
        with torch.no_grad():
            scores = model(input_ids, attention_mask)

        scores = scores.cpu().numpy()[0]  # Convert to numpy

        # Build result
        result = {
            'scores': {cat: float(score) for cat, score in zip(CATEGORIES, scores)},
            'max_score': float(scores.max()),
            'max_category': CATEGORIES[scores.argmax()],
            'categories_detected': [
                CATEGORIES[i] for i, score in enumerate(scores) if score > 0.5
            ]
        }

        # Decision logic
        if result['max_score'] > 0.8:
            result['action'] = 'reject'
            result['status'] = 'rejected'
            result['confidence'] = result['max_score']
        elif result['max_score'] < 0.2:
            result['action'] = 'allow'
            result['status'] = 'approved'
            result['confidence'] = 1.0 - result['max_score']
        else:
            result['action'] = 'review'
            result['status'] = 'pending_review'
            result['confidence'] = 0.5  # Uncertain

        return result

    # Example usage
    text = "I hate you and wish you were dead"
    result = moderate_text(text, model, tokenizer, device)
    print(result)
    # {
    #   'scores': {'toxicity': 0.92, 'hate_speech': 0.85, 'threat': 0.78, ...},
    #   'max_score': 0.92,
    #   'max_category': 'toxicity',
    #   'categories_detected': ['toxicity', 'hate_speech', 'threat'],
    #   'action': 'reject',
    #   'status': 'rejected',
    #   'confidence': 0.92
    # }
    ```

    ### Multi-language Support

    ```python
    from transformers import XLMRobertaModel, XLMRobertaTokenizer

    class MultilingualToxicityClassifier(nn.Module):
        """
        XLM-RoBERTa-based classifier for 50+ languages
        """
        def __init__(self, num_labels=6, dropout=0.3):
            super().__init__()
            self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.xlm_roberta.config.hidden_size, num_labels)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_ids, attention_mask):
            outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return self.sigmoid(logits)

    # Language detection
    from langdetect import detect

    def moderate_multilingual_text(text: str, model, tokenizer, device) -> dict:
        """
        Detect language and moderate text
        """
        try:
            language = detect(text)
        except:
            language = 'unknown'

        result = moderate_text(text, model, tokenizer, device)
        result['language'] = language
        return result
    ```

    ### Contextual Moderation

    ```python
    def moderate_with_context(
        text: str,
        user_id: str,
        community_id: str,
        parent_text: str = None,
        model=None,
        tokenizer=None,
        device=None
    ) -> dict:
        """
        Context-aware moderation considering:
        - User reputation
        - Community standards
        - Conversation context
        """
        # Get base moderation score
        result = moderate_text(text, model, tokenizer, device)

        # Fetch user reputation (from feature store)
        user_reputation = get_user_reputation(user_id)  # 0-100 score

        # Adjust thresholds based on reputation
        if user_reputation > 80:
            # Trusted users: more lenient
            auto_reject_threshold = 0.9
            auto_approve_threshold = 0.3
        elif user_reputation < 20:
            # New/low-reputation users: stricter
            auto_reject_threshold = 0.6
            auto_approve_threshold = 0.15
        else:
            # Regular users: standard thresholds
            auto_reject_threshold = 0.8
            auto_approve_threshold = 0.2

        # Re-evaluate action with adjusted thresholds
        max_score = result['max_score']
        if max_score > auto_reject_threshold:
            result['action'] = 'reject'
            result['status'] = 'rejected'
        elif max_score < auto_approve_threshold:
            result['action'] = 'allow'
            result['status'] = 'approved'
        else:
            result['action'] = 'review'
            result['status'] = 'pending_review'

        # Add context to result
        result['context'] = {
            'user_reputation': user_reputation,
            'community_id': community_id,
            'is_reply': parent_text is not None
        }

        return result

    def get_user_reputation(user_id: str) -> int:
        """
        Fetch user reputation from feature store (Redis)
        Returns score 0-100
        """
        import redis
        r = redis.Redis(host='feature-store', port=6379)
        reputation = r.get(f'user:{user_id}:reputation')
        return int(reputation) if reputation else 50  # Default: 50
    ```

    ---

    ## 3.2 Image NSFW Detection with ResNet

    ### Model Architecture

    ```python
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image

    class NSFWClassifier(nn.Module):
        """
        ResNet50-based NSFW image classifier
        Categories: nudity, sexual, violence, gore, self_harm
        """
        def __init__(self, num_labels=5, pretrained=True):
            super().__init__()
            self.resnet = models.resnet50(pretrained=pretrained)

            # Replace final layer for multi-label classification
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, num_labels),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.resnet(x)

    # Initialize model
    model = NSFWClassifier(num_labels=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    CATEGORIES = ['nudity', 'sexual', 'violence', 'gore', 'self_harm']
    ```

    ### Inference Pipeline

    ```python
    def moderate_image(image_path: str, model, transform, device) -> dict:
        """
        Moderate image content and return NSFW scores

        Args:
            image_path: Path to image file or URL
            model: Trained NSFW classifier
            transform: Image preprocessing transforms
            device: torch device

        Returns:
            dict with scores, categories, and action
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            scores = model(image_tensor)

        scores = scores.cpu().numpy()[0]

        # Build result
        result = {
            'scores': {cat: float(score) for cat, score in zip(CATEGORIES, scores)},
            'max_score': float(scores.max()),
            'max_category': CATEGORIES[scores.argmax()],
            'categories_detected': [
                CATEGORIES[i] for i, score in enumerate(scores) if score > 0.5
            ]
        }

        # Decision logic
        if result['max_score'] > 0.8:
            result['action'] = 'reject'
            result['status'] = 'rejected'
            result['confidence'] = result['max_score']
        elif result['max_score'] < 0.2:
            result['action'] = 'allow'
            result['status'] = 'approved'
            result['confidence'] = 1.0 - result['max_score']
        else:
            result['action'] = 'review'
            result['status'] = 'pending_review'
            result['confidence'] = 0.5

        return result

    # Example usage
    result = moderate_image('user_upload.jpg', model, transform, device)
    print(result)
    # {
    #   'scores': {'nudity': 0.95, 'sexual': 0.88, 'violence': 0.05, ...},
    #   'max_score': 0.95,
    #   'max_category': 'nudity',
    #   'categories_detected': ['nudity', 'sexual'],
    #   'action': 'reject',
    #   'status': 'rejected',
    #   'confidence': 0.95
    # }
    ```

    ### Perceptual Hashing for Duplicate Detection

    ```python
    import imagehash
    from PIL import Image
    import redis

    def compute_image_hash(image_path: str) -> str:
        """
        Compute perceptual hash for duplicate detection
        Robust to resizing, compression, minor edits
        """
        image = Image.open(image_path)
        phash = imagehash.phash(image, hash_size=16)  # 256-bit hash
        return str(phash)

    def check_duplicate_image(image_path: str, redis_client) -> dict:
        """
        Check if image has been previously moderated
        """
        image_hash = compute_image_hash(image_path)

        # Check Redis cache
        cached_result = redis_client.get(f'image_hash:{image_hash}')

        if cached_result:
            import json
            return {
                'is_duplicate': True,
                'cached_result': json.loads(cached_result),
                'hash': image_hash
            }
        else:
            return {
                'is_duplicate': False,
                'hash': image_hash
            }

    def moderate_image_with_cache(image_path: str, model, transform, device, redis_client) -> dict:
        """
        Moderate image with duplicate detection
        """
        # Check cache first
        duplicate_check = check_duplicate_image(image_path, redis_client)

        if duplicate_check['is_duplicate']:
            # Return cached result
            result = duplicate_check['cached_result']
            result['cache_hit'] = True
            return result

        # New image: run ML inference
        result = moderate_image(image_path, model, transform, device)
        result['cache_hit'] = False

        # Cache result
        import json
        redis_client.setex(
            f"image_hash:{duplicate_check['hash']}",
            30 * 24 * 3600,  # TTL: 30 days
            json.dumps(result)
        )

        return result

    # Usage with Redis
    redis_client = redis.Redis(host='hash-cache', port=6379)
    result = moderate_image_with_cache('upload.jpg', model, transform, device, redis_client)
    ```

    ---

    ## 3.3 Video Moderation with Frame Sampling

    ### Frame Extraction and Analysis

    ```python
    import cv2
    import numpy as np
    from typing import List, Dict

    def extract_frames(video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """
        Extract evenly-spaced frames from video

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract

        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        # Calculate frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames

    def moderate_video(video_path: str, image_model, text_model, transform, device) -> dict:
        """
        Moderate video by analyzing frames and audio

        Args:
            video_path: Path to video file
            image_model: Trained NSFW image classifier
            text_model: Trained toxicity text classifier
            transform: Image preprocessing
            device: torch device

        Returns:
            dict with video moderation results
        """
        # Extract frames
        frames = extract_frames(video_path, num_frames=10)

        # Analyze each frame
        frame_scores = []
        for frame in frames:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)

            # Save temporarily and moderate
            temp_path = '/tmp/frame.jpg'
            pil_image.save(temp_path)
            frame_result = moderate_image(temp_path, image_model, transform, device)
            frame_scores.append(frame_result['max_score'])

        # Aggregate frame scores
        max_frame_score = max(frame_scores)
        avg_frame_score = np.mean(frame_scores)

        # Extract and moderate audio (simplified)
        audio_score = moderate_video_audio(video_path, text_model, device)

        # Combine scores
        final_score = max(max_frame_score, audio_score)

        result = {
            'video_score': final_score,
            'max_frame_score': max_frame_score,
            'avg_frame_score': avg_frame_score,
            'audio_score': audio_score,
            'num_frames_analyzed': len(frames)
        }

        # Decision
        if final_score > 0.8:
            result['action'] = 'reject'
            result['status'] = 'rejected'
            result['confidence'] = final_score
        elif final_score < 0.2:
            result['action'] = 'allow'
            result['status'] = 'approved'
            result['confidence'] = 1.0 - final_score
        else:
            result['action'] = 'review'
            result['status'] = 'pending_review'
            result['confidence'] = 0.5

        return result

    def moderate_video_audio(video_path: str, text_model, device) -> float:
        """
        Extract audio, transcribe, and moderate text

        Args:
            video_path: Path to video file
            text_model: Trained toxicity classifier
            device: torch device

        Returns:
            Audio moderation score (0-1)
        """
        # Extract audio using ffmpeg
        import subprocess
        audio_path = '/tmp/audio.wav'
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ], capture_output=True)

        # Transcribe audio (using Whisper or similar)
        transcript = transcribe_audio(audio_path)

        if not transcript:
            return 0.0  # No audio or silent

        # Moderate transcript
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        result = moderate_text(transcript, text_model, tokenizer, device)

        return result['max_score']

    def transcribe_audio(audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper
        """
        import whisper
        model = whisper.load_model('base')
        result = model.transcribe(audio_path)
        return result['text']
    ```

    ### Scene Detection for Intelligent Sampling

    ```python
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector

    def extract_frames_with_scene_detection(video_path: str, max_frames: int = 20) -> List[np.ndarray]:
        """
        Extract keyframes at scene boundaries + uniform sampling
        More intelligent than uniform sampling alone
        """
        # Scene detection
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30))

        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()

        # Get scene boundary frames
        scene_frames = [scene[0].get_frames() for scene in scene_list]

        # If too many scenes, sample uniformly
        if len(scene_frames) > max_frames:
            indices = np.linspace(0, len(scene_frames) - 1, max_frames, dtype=int)
            scene_frames = [scene_frames[i] for i in indices]

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        for frame_num in scene_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames
    ```

    ---

    ## 3.4 Human Review Interface

    ### Review Queue Priority Scoring

    ```python
    from typing import Dict, List
    import heapq
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class ReviewItem:
        """Represents an item in the human review queue"""
        content_id: str
        content_type: str  # text, image, video
        ml_score: float  # 0-1
        submission_time: datetime
        user_reputation: int  # 0-100
        engagement_score: int  # views, likes
        is_user_report: bool
        priority_score: float

        def __lt__(self, other):
            # For heapq (max-heap via negation)
            return self.priority_score > other.priority_score

    def calculate_priority_score(
        ml_score: float,
        submission_time: datetime,
        user_reputation: int,
        engagement_score: int,
        is_user_report: bool
    ) -> float:
        """
        Calculate priority score for review queue
        Higher score = higher priority

        Factors:
        - User reports: highest priority
        - ML uncertainty: prioritize 0.4-0.6 range
        - User reputation: new/low-rep users get priority
        - Engagement: viral content gets priority
        - Time in queue: prevent starvation
        """
        priority = 0.0

        # User reports always prioritized
        if is_user_report:
            priority += 100.0

        # ML uncertainty (prefer scores near 0.5)
        uncertainty = 1 - abs(ml_score - 0.5) * 2  # Peak at 0.5
        priority += uncertainty * 50.0

        # User reputation (lower = higher priority)
        reputation_factor = (100 - user_reputation) / 100.0
        priority += reputation_factor * 30.0

        # Engagement (log scale)
        if engagement_score > 0:
            import math
            engagement_factor = math.log10(engagement_score + 1) / 6.0  # Normalize
            priority += engagement_factor * 20.0

        # Time in queue (prevent starvation)
        time_in_queue = (datetime.utcnow() - submission_time).total_seconds() / 3600.0  # hours
        time_factor = min(time_in_queue / 24.0, 1.0)  # Max boost after 24h
        priority += time_factor * 30.0

        return priority

    class ReviewQueue:
        """Priority queue for human review items"""
        def __init__(self):
            self.queue = []
            self.items_map = {}  # content_id -> ReviewItem

        def add_item(
            self,
            content_id: str,
            content_type: str,
            ml_score: float,
            user_reputation: int,
            engagement_score: int,
            is_user_report: bool = False
        ):
            """Add item to review queue"""
            submission_time = datetime.utcnow()
            priority = calculate_priority_score(
                ml_score, submission_time, user_reputation,
                engagement_score, is_user_report
            )

            item = ReviewItem(
                content_id=content_id,
                content_type=content_type,
                ml_score=ml_score,
                submission_time=submission_time,
                user_reputation=user_reputation,
                engagement_score=engagement_score,
                is_user_report=is_user_report,
                priority_score=priority
            )

            heapq.heappush(self.queue, item)
            self.items_map[content_id] = item

        def get_next_item(self) -> ReviewItem:
            """Get highest priority item from queue"""
            if not self.queue:
                return None
            item = heapq.heappop(self.queue)
            del self.items_map[item.content_id]
            return item

        def size(self) -> int:
            return len(self.queue)

    # Example usage
    queue = ReviewQueue()

    # Add items
    queue.add_item(
        content_id='post_123',
        content_type='text',
        ml_score=0.55,  # Uncertain
        user_reputation=20,  # Low rep
        engagement_score=1000,  # Some engagement
        is_user_report=False
    )

    queue.add_item(
        content_id='post_456',
        content_type='image',
        ml_score=0.75,
        user_reputation=80,
        engagement_score=50000,  # Viral
        is_user_report=True  # User reported
    )

    # Get next item to review
    next_item = queue.get_next_item()
    print(f"Review: {next_item.content_id}, Priority: {next_item.priority_score:.2f}")
    # Review: post_456, Priority: 150.25 (user report + viral)
    ```

    ### Active Learning Sample Selection

    ```python
    def select_active_learning_samples(
        pending_items: List[Dict],
        current_model_version: str,
        num_samples: int = 100
    ) -> List[Dict]:
        """
        Select most informative samples for human review
        to maximize model improvement

        Strategies:
        1. Uncertainty sampling: High entropy predictions
        2. Diversity sampling: Cover all categories
        3. Disagreement sampling: Ensemble models disagree
        4. Error sampling: Previous model errors
        """
        selected_samples = []

        # Strategy 1: Uncertainty sampling (confidence 0.45-0.55)
        uncertain_samples = [
            item for item in pending_items
            if 0.45 <= item['ml_score'] <= 0.55
        ]
        selected_samples.extend(uncertain_samples[:num_samples // 2])

        # Strategy 2: Diversity sampling (one from each category)
        categories = {}
        for item in pending_items:
            cat = item['max_category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)

        for cat, items in categories.items():
            if len(selected_samples) < num_samples:
                selected_samples.append(items[0])

        # Strategy 3: Boundary samples (scores near decision thresholds)
        boundary_samples = [
            item for item in pending_items
            if (0.18 <= item['ml_score'] <= 0.22) or (0.78 <= item['ml_score'] <= 0.82)
        ]
        remaining = num_samples - len(selected_samples)
        selected_samples.extend(boundary_samples[:remaining])

        return selected_samples[:num_samples]
    ```

    ---

    ## 3.5 Appeal Workflow

    ### Appeal Processing

    ```python
    from enum import Enum
    from typing import Optional

    class AppealStatus(Enum):
        PENDING = 'pending'
        APPROVED = 'approved'
        REJECTED = 'rejected'
        ESCALATED = 'escalated'

    class AppealDecision(Enum):
        UPHOLD = 'uphold'  # Original decision correct
        OVERTURN = 'overturn'  # Original decision wrong
        ESCALATE = 'escalate'  # Needs senior review

    @dataclass
    class Appeal:
        appeal_id: str
        content_id: str
        user_id: str
        original_moderation_id: str
        original_action: str
        appeal_reason: str
        submission_time: datetime
        status: AppealStatus
        decision: Optional[AppealDecision] = None
        reviewer_id: Optional[str] = None
        review_time: Optional[datetime] = None
        review_notes: Optional[str] = None

    def process_appeal(appeal: Appeal, content, original_moderation) -> Appeal:
        """
        Process user appeal with automated triage

        Automated overturn cases:
        - Original decision was borderline (score 0.75-0.85)
        - User has excellent reputation (>90)
        - Similar appeals were overturned recently

        Automated uphold cases:
        - Original score >0.95 (very high confidence)
        - User has poor reputation (<10)
        - Repeat offender

        Otherwise: Human review
        """
        # Fetch context
        user_reputation = get_user_reputation(appeal.user_id)
        original_score = original_moderation['max_score']

        # Automated overturn
        if (0.75 <= original_score <= 0.85 and user_reputation > 90):
            appeal.status = AppealStatus.APPROVED
            appeal.decision = AppealDecision.OVERTURN
            appeal.review_notes = 'Automated overturn: borderline score + excellent reputation'
            return appeal

        # Automated uphold
        if original_score > 0.95 and user_reputation < 20:
            appeal.status = AppealStatus.REJECTED
            appeal.decision = AppealDecision.UPHOLD
            appeal.review_notes = 'Automated uphold: high confidence violation + low reputation'
            return appeal

        # Escalate to human reviewer
        appeal.status = AppealStatus.PENDING
        return appeal

    def reviewer_decide_appeal(
        appeal: Appeal,
        decision: AppealDecision,
        reviewer_id: str,
        notes: str
    ) -> Appeal:
        """
        Human reviewer makes final decision on appeal
        """
        appeal.decision = decision
        appeal.reviewer_id = reviewer_id
        appeal.review_time = datetime.utcnow()
        appeal.review_notes = notes

        if decision == AppealDecision.UPHOLD:
            appeal.status = AppealStatus.REJECTED
        elif decision == AppealDecision.OVERTURN:
            appeal.status = AppealStatus.APPROVED
            # Restore content
            restore_content(appeal.content_id)
            # Update user reputation positively
            update_user_reputation(appeal.user_id, delta=+5)
        elif decision == AppealDecision.ESCALATE:
            appeal.status = AppealStatus.ESCALATED
            # Send to senior reviewer queue

        # Log decision for model retraining
        log_appeal_outcome(appeal)

        return appeal

    def restore_content(content_id: str):
        """Restore previously removed content"""
        # Implementation: Update content status in database
        pass

    def log_appeal_outcome(appeal: Appeal):
        """Log appeal outcome for model retraining"""
        import kafka
        producer = kafka.KafkaProducer(bootstrap_servers=['kafka:9092'])

        event = {
            'event_type': 'appeal_outcome',
            'appeal_id': appeal.appeal_id,
            'content_id': appeal.content_id,
            'decision': appeal.decision.value,
            'timestamp': appeal.review_time.isoformat()
        }

        producer.send('moderation.feedback', value=json.dumps(event).encode())
    ```

    ---

    ## 3.6 Integration with External APIs

    ### OpenAI Moderation API

    ```python
    import openai

    def moderate_with_openai(text: str, api_key: str) -> dict:
        """
        Use OpenAI Moderation API as a secondary signal
        """
        openai.api_key = api_key

        response = openai.Moderation.create(input=text)
        result = response['results'][0]

        return {
            'flagged': result['flagged'],
            'categories': result['categories'],
            'category_scores': result['category_scores'],
            'source': 'openai'
        }

    # Example usage
    text = "I will kill you"
    openai_result = moderate_with_openai(text, api_key='sk-...')
    print(openai_result)
    # {
    #   'flagged': True,
    #   'categories': {'violence': True, 'hate': False, ...},
    #   'category_scores': {'violence': 0.98, 'hate': 0.05, ...},
    #   'source': 'openai'
    # }
    ```

    ### Google Perspective API

    ```python
    import requests

    def moderate_with_perspective(text: str, api_key: str) -> dict:
        """
        Use Google Perspective API for toxicity analysis
        """
        url = f'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}'

        data = {
            'comment': {'text': text},
            'languages': ['en'],
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'IDENTITY_ATTACK': {},
                'INSULT': {},
                'PROFANITY': {},
                'THREAT': {}
            }
        }

        response = requests.post(url, json=data)
        result = response.json()

        scores = {}
        for attr, data in result['attributeScores'].items():
            scores[attr.lower()] = data['summaryScore']['value']

        return {
            'scores': scores,
            'max_score': max(scores.values()),
            'source': 'perspective'
        }

    # Example usage
    text = "You are an idiot"
    perspective_result = moderate_with_perspective(text, api_key='...')
    print(perspective_result)
    # {
    #   'scores': {'toxicity': 0.87, 'insult': 0.92, 'threat': 0.05, ...},
    #   'max_score': 0.92,
    #   'source': 'perspective'
    # }
    ```

    ### Ensemble Multiple APIs

    ```python
    def ensemble_moderation(text: str, internal_model, openai_key: str, perspective_key: str) -> dict:
        """
        Combine multiple moderation signals for higher accuracy
        """
        # Get scores from all sources
        internal_result = moderate_text(text, internal_model, tokenizer, device)
        openai_result = moderate_with_openai(text, openai_key)
        perspective_result = moderate_with_perspective(text, perspective_key)

        # Weighted ensemble
        internal_score = internal_result['max_score']
        openai_score = max(openai_result['category_scores'].values()) if openai_result['flagged'] else 0.0
        perspective_score = perspective_result['max_score']

        # Weighted average: internal 50%, OpenAI 25%, Perspective 25%
        ensemble_score = (
            0.5 * internal_score +
            0.25 * openai_score +
            0.25 * perspective_score
        )

        return {
            'ensemble_score': ensemble_score,
            'internal_score': internal_score,
            'openai_score': openai_score,
            'perspective_score': perspective_score,
            'action': 'reject' if ensemble_score > 0.8 else ('allow' if ensemble_score < 0.2 else 'review')
        }
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## 4.1 Caching Strategies

    ### Multi-Level Caching

    ```python
    import redis
    import hashlib
    from typing import Optional

    class ModerationCache:
        """
        Multi-level cache for moderation results

        L1: In-memory cache (hot content)
        L2: Redis cache (warm content)
        L3: Database (cold content)
        """
        def __init__(self, redis_client):
            self.redis = redis_client
            self.local_cache = {}  # LRU cache
            self.max_local_size = 10000

        def get(self, content: str, content_type: str) -> Optional[dict]:
            """Get cached moderation result"""
            cache_key = self._compute_key(content, content_type)

            # L1: Check local cache
            if cache_key in self.local_cache:
                return self.local_cache[cache_key]

            # L2: Check Redis
            cached = self.redis.get(f'moderation:{cache_key}')
            if cached:
                import json
                result = json.loads(cached)
                self._update_local_cache(cache_key, result)
                return result

            # L3: Miss - need to compute
            return None

        def set(self, content: str, content_type: str, result: dict, ttl: int = 86400):
            """Cache moderation result"""
            cache_key = self._compute_key(content, content_type)

            # Store in Redis (L2)
            import json
            self.redis.setex(f'moderation:{cache_key}', ttl, json.dumps(result))

            # Store in local cache (L1)
            self._update_local_cache(cache_key, result)

        def _compute_key(self, content: str, content_type: str) -> str:
            """Compute cache key from content"""
            if content_type == 'text':
                # Normalize text
                normalized = content.lower().strip()
                return hashlib.sha256(normalized.encode()).hexdigest()
            elif content_type == 'image':
                # Use perceptual hash
                return compute_image_hash(content)
            elif content_type == 'video':
                # Hash of keyframes
                return compute_video_hash(content)

        def _update_local_cache(self, key: str, value: dict):
            """Update local LRU cache"""
            if len(self.local_cache) >= self.max_local_size:
                # Evict oldest
                self.local_cache.pop(next(iter(self.local_cache)))
            self.local_cache[key] = value

    # Usage
    redis_client = redis.Redis(host='cache', port=6379)
    cache = ModerationCache(redis_client)

    # Check cache before moderation
    result = cache.get(text, 'text')
    if result:
        print('Cache hit!')
    else:
        result = moderate_text(text, model, tokenizer, device)
        cache.set(text, 'text', result)
    ```

    ### Content Hash Deduplication

    ```python
    def moderate_with_deduplication(
        content: str,
        content_type: str,
        model,
        cache: ModerationCache
    ) -> dict:
        """
        Moderate content with deduplication
        30% cache hit rate expected
        """
        # Check cache
        cached_result = cache.get(content, content_type)

        if cached_result:
            cached_result['cache_hit'] = True
            cached_result['latency_ms'] = 5  # Cache lookup latency
            return cached_result

        # Cache miss: run moderation
        import time
        start = time.time()

        if content_type == 'text':
            result = moderate_text(content, model, tokenizer, device)
        elif content_type == 'image':
            result = moderate_image(content, model, transform, device)
        elif content_type == 'video':
            result = moderate_video(content, model, model, transform, device)

        latency = (time.time() - start) * 1000
        result['cache_hit'] = False
        result['latency_ms'] = latency

        # Cache result
        cache.set(content, content_type, result)

        return result
    ```

    ---

    ## 4.2 Tiered Review System

    ### Auto-Approve / Auto-Reject / Human Review

    ```python
    class TieredModerationSystem:
        """
        Three-tier moderation for efficiency

        Tier 1: Auto-approve (85% of content)
        Tier 2: Auto-reject (10% of content)
        Tier 3: Human review (5% of content)
        """
        def __init__(
            self,
            auto_approve_threshold: float = 0.2,
            auto_reject_threshold: float = 0.8
        ):
            self.auto_approve_threshold = auto_approve_threshold
            self.auto_reject_threshold = auto_reject_threshold
            self.metrics = {
                'auto_approved': 0,
                'auto_rejected': 0,
                'human_review': 0
            }

        def process(self, content: str, content_type: str, model, cache) -> dict:
            """Process content through tiered system"""
            # Get moderation score
            result = moderate_with_deduplication(content, content_type, model, cache)
            score = result['max_score']

            # Tier 1: Auto-approve
            if score < self.auto_approve_threshold:
                result['tier'] = 'auto_approve'
                result['action'] = 'allow'
                result['requires_review'] = False
                self.metrics['auto_approved'] += 1

            # Tier 2: Auto-reject
            elif score > self.auto_reject_threshold:
                result['tier'] = 'auto_reject'
                result['action'] = 'remove'
                result['requires_review'] = False
                self.metrics['auto_rejected'] += 1

            # Tier 3: Human review
            else:
                result['tier'] = 'human_review'
                result['action'] = 'pending'
                result['requires_review'] = True
                self.metrics['human_review'] += 1
                # Add to review queue
                add_to_review_queue(content, result)

            return result

        def get_metrics(self) -> dict:
            """Get system metrics"""
            total = sum(self.metrics.values())
            return {
                'auto_approve_rate': self.metrics['auto_approved'] / total,
                'auto_reject_rate': self.metrics['auto_rejected'] / total,
                'human_review_rate': self.metrics['human_review'] / total,
                'automation_rate': (self.metrics['auto_approved'] + self.metrics['auto_rejected']) / total
            }

    # Usage
    system = TieredModerationSystem()
    result = system.process(text, 'text', model, cache)
    print(f"Tier: {result['tier']}, Action: {result['action']}")

    # Check automation rate
    metrics = system.get_metrics()
    print(f"Automation rate: {metrics['automation_rate']:.2%}")  # Target: 95%
    ```

    ---

    ## 4.3 Model Retraining Pipeline

    ### Continuous Learning

    ```python
    from datetime import datetime, timedelta
    from typing import List, Tuple
    import torch.optim as optim

    class ModelRetrainingPipeline:
        """
        Automated pipeline for model retraining

        Nightly: Full retrain on 30 days of data
        Hourly: Incremental updates on last hour
        """
        def __init__(self, model, optimizer, device):
            self.model = model
            self.optimizer = optimizer
            self.device = device
            self.feedback_buffer = []

        def add_feedback(
            self,
            content: str,
            predicted_label: int,
            true_label: int,
            confidence: float
        ):
            """Add human feedback to training buffer"""
            self.feedback_buffer.append({
                'content': content,
                'predicted': predicted_label,
                'true': true_label,
                'confidence': confidence,
                'timestamp': datetime.utcnow()
            })

        def should_retrain(self, mode: str = 'nightly') -> bool:
            """Check if retraining should occur"""
            if mode == 'nightly':
                # Retrain at 2 AM daily
                now = datetime.utcnow()
                return now.hour == 2 and now.minute < 10
            elif mode == 'incremental':
                # Retrain every hour if enough feedback
                return len(self.feedback_buffer) >= 1000

        def fetch_training_data(self, days: int = 30) -> Tuple[List, List]:
            """Fetch training data from last N days"""
            # Query database for human-labeled examples
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Fetch from moderation DB
            query = f"""
                SELECT content, labels, scores
                FROM moderation_events
                WHERE timestamp >= '{cutoff.isoformat()}'
                AND reviewer_id IS NOT NULL
                LIMIT 1000000
            """
            # Execute query and return (contents, labels)
            pass

        def retrain_full(self):
            """Full model retrain on 30 days of data"""
            print("[Retraining] Starting full model retrain...")

            # Fetch training data
            contents, labels = self.fetch_training_data(days=30)

            # Create DataLoader
            from torch.utils.data import DataLoader, TensorDataset
            dataset = TensorDataset(contents, labels)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Training loop
            self.model.train()
            for epoch in range(3):  # 3 epochs
                total_loss = 0
                for batch in dataloader:
                    self.optimizer.zero_grad()
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    loss = nn.BCELoss()(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

            self.model.eval()

            # Save model
            self.save_model(version=f'nightly_{datetime.utcnow().strftime("%Y%m%d")}')
            print("[Retraining] Full retrain complete!")

        def retrain_incremental(self):
            """Incremental update on recent feedback"""
            print("[Retraining] Starting incremental update...")

            if len(self.feedback_buffer) < 100:
                print("[Retraining] Not enough feedback, skipping")
                return

            # Use feedback buffer
            contents = [f['content'] for f in self.feedback_buffer]
            labels = [f['true'] for f in self.feedback_buffer]

            # Fine-tune for 1 epoch
            # ... training code ...

            # Clear buffer
            self.feedback_buffer = []

            print("[Retraining] Incremental update complete!")

        def save_model(self, version: str):
            """Save model to registry"""
            path = f'/models/toxicity_classifier_{version}.pt'
            torch.save(self.model.state_dict(), path)
            print(f"[Model] Saved to {path}")

        def evaluate_model(self, test_data) -> dict:
            """Evaluate model on test set"""
            self.model.eval()

            correct = 0
            total = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            with torch.no_grad():
                for inputs, targets in test_data:
                    outputs = self.model(inputs)
                    predictions = (outputs > 0.5).float()

                    correct += (predictions == targets).sum().item()
                    total += targets.numel()

                    true_positives += ((predictions == 1) & (targets == 1)).sum().item()
                    false_positives += ((predictions == 1) & (targets == 0)).sum().item()
                    false_negatives += ((predictions == 0) & (targets == 1)).sum().item()

            accuracy = correct / total
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    ```

    ### A/B Testing New Models

    ```python
    import random

    class ModelABTesting:
        """
        A/B test new model versions

        Shadow mode (0%) → Canary (5%) → Half (50%) → Full (100%)
        """
        def __init__(self):
            self.model_a = load_model('v2.3.1')  # Current production
            self.model_b = load_model('v2.4.0')  # New candidate
            self.rollout_percentage = 5  # Start with 5%
            self.metrics_a = {'correct': 0, 'total': 0}
            self.metrics_b = {'correct': 0, 'total': 0}

        def get_model(self) -> tuple:
            """Select model based on rollout percentage"""
            if random.random() < self.rollout_percentage / 100.0:
                return self.model_b, 'model_b'
            else:
                return self.model_a, 'model_a'

        def moderate_with_ab_test(self, content: str) -> dict:
            """Moderate content with A/B testing"""
            model, model_name = self.get_model()

            result = moderate_text(content, model, tokenizer, device)
            result['model_version'] = model_name

            return result

        def log_outcome(self, model_name: str, is_correct: bool):
            """Log A/B test outcome"""
            if model_name == 'model_a':
                self.metrics_a['total'] += 1
                if is_correct:
                    self.metrics_a['correct'] += 1
            else:
                self.metrics_b['total'] += 1
                if is_correct:
                    self.metrics_b['correct'] += 1

        def should_increase_rollout(self) -> bool:
            """Decide if model B is performing well enough to increase rollout"""
            if self.metrics_b['total'] < 1000:
                return False  # Not enough data

            accuracy_a = self.metrics_a['correct'] / self.metrics_a['total']
            accuracy_b = self.metrics_b['correct'] / self.metrics_b['total']

            # Model B must be at least as good as A
            return accuracy_b >= accuracy_a * 0.99  # 99% of A's accuracy

        def increase_rollout(self):
            """Gradually increase rollout percentage"""
            if self.rollout_percentage < 100:
                self.rollout_percentage = min(self.rollout_percentage * 2, 100)
                print(f"[A/B Test] Increased rollout to {self.rollout_percentage}%")
    ```

    ---

    ## 4.4 GPU Optimization

    ### Batching for Higher Throughput

    ```python
    import asyncio
    from collections import deque
    from typing import List

    class BatchProcessor:
        """
        Batch multiple requests for GPU efficiency
        Wait up to 50ms or batch size 32 before processing
        """
        def __init__(self, model, batch_size: int = 32, max_wait_ms: int = 50):
            self.model = model
            self.batch_size = batch_size
            self.max_wait_ms = max_wait_ms
            self.queue = deque()
            self.results = {}

        async def moderate_async(self, content: str, content_id: str) -> dict:
            """Add request to batch queue"""
            future = asyncio.Future()
            self.queue.append({'content': content, 'content_id': content_id, 'future': future})

            # Trigger batch processing if queue full
            if len(self.queue) >= self.batch_size:
                await self._process_batch()
            else:
                # Wait for max_wait_ms
                await asyncio.sleep(self.max_wait_ms / 1000.0)
                if self.queue:
                    await self._process_batch()

            return await future

        async def _process_batch(self):
            """Process batch of requests"""
            if not self.queue:
                return

            # Collect batch
            batch = []
            futures = []
            while self.queue and len(batch) < self.batch_size:
                item = self.queue.popleft()
                batch.append(item['content'])
                futures.append(item['future'])

            # Tokenize batch
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding='max_length'
            )

            # GPU inference (batch)
            with torch.no_grad():
                outputs = self.model(
                    inputs['input_ids'].to(device),
                    inputs['attention_mask'].to(device)
                )

            # Parse results
            for i, future in enumerate(futures):
                scores = outputs[i].cpu().numpy()
                result = {
                    'scores': {cat: float(scores[j]) for j, cat in enumerate(CATEGORIES)},
                    'max_score': float(scores.max())
                }
                future.set_result(result)

    # Usage
    batch_processor = BatchProcessor(model, batch_size=32, max_wait_ms=50)

    async def handle_request(content: str, content_id: str):
        result = await batch_processor.moderate_async(content, content_id)
        return result

    # Process 100 concurrent requests
    tasks = [handle_request(f"Text {i}", f"id_{i}") for i in range(100)]
    results = await asyncio.gather(*tasks)
    ```

    ### Model Quantization

    ```python
    import torch.quantization as quantization

    def quantize_model(model):
        """
        Quantize model to INT8 for 4x faster inference
        Minimal accuracy loss (<1%)
        """
        # Prepare model for quantization
        model.eval()
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)

        # Calibrate with sample data
        calibration_data = load_calibration_data()  # 1000 samples
        with torch.no_grad():
            for inputs in calibration_data:
                model(inputs)

        # Convert to quantized model
        quantized_model = quantization.convert(model, inplace=False)

        return quantized_model

    # Usage
    quantized_model = quantize_model(model)

    # Inference is 4x faster
    # Model size reduced from 500MB to 125MB
    ```

    ---

    ## 4.5 Monitoring and Alerting

    ### Real-time Metrics

    ```python
    from prometheus_client import Counter, Histogram, Gauge
    import time

    # Define metrics
    moderation_requests = Counter('moderation_requests_total', 'Total moderation requests', ['content_type', 'action'])
    moderation_latency = Histogram('moderation_latency_seconds', 'Moderation latency', ['content_type'])
    cache_hit_rate = Gauge('moderation_cache_hit_rate', 'Cache hit rate')
    human_review_queue_size = Gauge('human_review_queue_size', 'Human review queue size')
    model_accuracy = Gauge('model_accuracy', 'Model accuracy', ['model_version'])

    def moderate_with_metrics(content: str, content_type: str, model, cache) -> dict:
        """Moderate content with metrics tracking"""
        start = time.time()

        # Moderate
        result = moderate_with_deduplication(content, content_type, model, cache)

        # Track latency
        latency = time.time() - start
        moderation_latency.labels(content_type=content_type).observe(latency)

        # Track action
        moderation_requests.labels(
            content_type=content_type,
            action=result['action']
        ).inc()

        # Track cache hit rate
        if result.get('cache_hit'):
            cache_hit_rate.set(cache_hit_rate._value.get() * 0.99 + 0.01)  # Exponential moving average
        else:
            cache_hit_rate.set(cache_hit_rate._value.get() * 0.99)

        return result

    def alert_if_needed(metrics: dict):
        """Send alerts for anomalies"""
        # Alert if cache hit rate drops
        if metrics['cache_hit_rate'] < 0.25:  # Expected: 30%
            send_alert('Low cache hit rate', severity='warning')

        # Alert if latency increases
        if metrics['p95_latency'] > 500:  # ms
            send_alert('High moderation latency', severity='critical')

        # Alert if false positive rate increases
        if metrics['false_positive_rate'] > 0.01:  # 1%
            send_alert('High false positive rate', severity='critical')

    def send_alert(message: str, severity: str):
        """Send alert to PagerDuty/Slack"""
        print(f"[ALERT] {severity.upper()}: {message}")
        # Integration with alerting system
    ```

    ---

    ## 4.6 Cost Optimization

    ### GPU Cost Analysis

    ```
    Current cost: 476 T4 GPUs × $0.35/hour = $4,000/day = $120K/month

    Optimization strategies:

    1. Caching (30% cache hit rate):
       - Reduce GPU load by 30%
       - New cost: $84K/month
       - Savings: $36K/month

    2. Batching (2x throughput increase):
       - Reduce GPUs needed by 50%
       - New cost: $42K/month
       - Savings: $42K/month

    3. Model quantization (4x faster):
       - Reduce GPUs needed by 75%
       - New cost: $10.5K/month
       - Savings: $31.5K/month

    4. Tiered processing (95% auto-decisions):
       - Only 5% needs full ML processing
       - New cost: $6K/month
       - Savings: $4.5K/month

    Total optimized cost: ~$10K/month (92% reduction)
    ```

    ---

    ## Summary

    **Key Optimizations:**

    1. **Caching:** 30% cache hit rate via content hashing
    2. **Tiered Review:** 95% automated decisions (auto-approve/reject)
    3. **Batching:** 32-item batches for 2x GPU throughput
    4. **Quantization:** INT8 quantization for 4x speedup
    5. **Model Retraining:** Daily retraining with human feedback
    6. **A/B Testing:** Gradual rollout of new models
    7. **Multi-level Cache:** Local + Redis + Database
    8. **Active Learning:** Sample high-uncertainty cases for review

    **Final Architecture Characteristics:**

    - Latency: <100ms p95 (real-time), <1 hour (batch)
    - Throughput: 1B items/day, 35K QPS peak
    - Accuracy: 99.5% precision, 95% recall
    - Automation: 95% auto-decisions, 5% human review
    - Cost: ~$10K/month GPU, ~$5K/month human reviewers
    - Scalability: Horizontal scaling of ML inference services

---

## References & Resources

### External APIs

1. **OpenAI Moderation API**
   - Endpoint: `https://api.openai.com/v1/moderations`
   - Categories: hate, hate/threatening, self-harm, sexual, sexual/minors, violence, violence/graphic
   - Free tier: 150K requests/month
   - [Documentation](https://platform.openai.com/docs/guides/moderation)

2. **Google Perspective API**
   - Endpoint: `https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze`
   - Attributes: TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, THREAT
   - Free tier: 1 QPS, paid tiers available
   - [Documentation](https://developers.perspectiveapi.com/)

### Models & Datasets

1. **Text Moderation:**
   - BERT-base fine-tuned on Jigsaw Toxic Comment Dataset
   - XLM-RoBERTa for multilingual (50+ languages)
   - DistilBERT for faster inference

2. **Image Moderation:**
   - ResNet50 fine-tuned on NSFW datasets
   - EfficientNet for better accuracy/speed tradeoff
   - CLIP for zero-shot classification

3. **Video Moderation:**
   - Frame sampling + ResNet50 for visual
   - Whisper for audio transcription
   - Scene detection for intelligent sampling

### Papers

1. "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
2. "Deep Residual Learning for Image Recognition" (He et al., 2015)
3. "Perspective API: Toxicity Detection in Online Conversations" (Jigsaw, 2017)
4. "Content Moderation at Scale" (Meta AI, 2020)
5. "Active Learning for Content Moderation" (Duarte et al., 2021)

### Similar Systems

1. **Meta Content Moderation:** 15K human reviewers + AI classifiers
2. **YouTube Moderation:** 10M videos/day, 500K hours/day
3. **TikTok Moderation:** 10K+ reviewers, real-time + batch
4. **Discord Moderation:** AutoMod + user reports
5. **Reddit Moderation:** Community moderators + AI assist

---

## Follow-up Questions

1. **How would you handle adversarial attacks?** (Users circumventing moderation with creative spelling, images with embedded text, etc.)

2. **How would you ensure fairness and reduce bias?** (Avoid over-moderating certain groups, languages, or topics)

3. **How would you handle evolving abuse patterns?** (New slang, memes, coded language that emerges over time)

4. **How would you moderate private vs. public content?** (Different risk tolerances, privacy concerns)

5. **How would you handle cross-platform abuse?** (Users coordinating attacks across multiple platforms)

6. **How would you support different community standards?** (Subreddit-specific rules, Discord server guidelines)

7. **How would you handle false positives at scale?** (1M false positives/day at 0.1% FPR with 1B items)

8. **How would you integrate legal compliance?** (GDPR, COPPA, regional content laws)

---

## Interview Tips

1. **Start with requirements:** Clarify real-time vs. batch, accuracy targets, content types
2. **Discuss tradeoffs:** Precision (fewer false positives) vs. Recall (catch all harmful content)
3. **Emphasize human-in-the-loop:** ML alone insufficient, humans crucial for edge cases
4. **Talk about feedback loops:** How human decisions improve models over time
5. **Address scale:** 1B items/day requires heavy optimization (caching, batching, tiering)
6. **Consider context:** User reputation, community norms, conversation context matter
7. **Plan for evolution:** Abuse patterns evolve, models need continuous retraining
8. **Discuss appeals:** Users must have recourse for false positives
9. **Cover multi-modal:** Text, image, and video all have different challenges
10. **Mention external APIs:** OpenAI, Perspective can supplement internal models

**Common mistakes to avoid:**
- Relying only on ML without human review
- Ignoring false positive impact on user experience
- Not considering cache/deduplication for repeated content
- Forgetting about appeal workflow
- Underestimating compute costs (GPU inference expensive)
- Not planning for adversarial attacks
- Ignoring context (first-time offender vs. repeat violator)
