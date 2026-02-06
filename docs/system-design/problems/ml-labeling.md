# Design an ML Data Labeling Platform

A scalable data annotation platform that manages workforce, task distribution, quality control, and consensus mechanisms to generate high-quality labeled datasets for training machine learning models across images, text, video, and audio modalities.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐⭐ High | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10K concurrent labelers, 10M tasks/month, 1B annotations, multi-tenant SaaS platform |
| **Key Challenges** | Quality control with consensus, task routing with skill matching, inter-annotator agreement, active learning for task selection, fraud detection, payment optimization |
| **Core Concepts** | Task assignment algorithms, consensus mechanisms (majority/weighted voting), inter-annotator agreement (Cohen's kappa, Fleiss' kappa), active learning, dispute resolution, workforce management |
| **Companies** | Labelbox, Scale AI, Amazon SageMaker Ground Truth, Snorkel AI, Appen, CloudFactory, Sama |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Task Upload & Management** | Bulk upload tasks (images, text, video) with instructions | P0 (Must have) |
    | **Task Assignment & Routing** | Intelligent routing based on labeler skills, availability | P0 (Must have) |
    | **Labeling Interfaces** | Image bounding boxes, polygon segmentation, text classification, NER | P0 (Must have) |
    | **Quality Control** | Multiple annotators per task, consensus mechanisms | P0 (Must have) |
    | **Workforce Management** | Labeler onboarding, skill assessment, performance tracking | P0 (Must have) |
    | **Consensus Engine** | Majority voting, weighted voting based on accuracy | P0 (Must have) |
    | **Inter-Annotator Agreement** | Calculate Cohen's kappa, Fleiss' kappa, Krippendorff's alpha | P0 (Must have) |
    | **Dispute Resolution** | Expert review for disagreements, golden tasks for validation | P1 (Should have) |
    | **Active Learning** | Prioritize uncertain tasks for labeling efficiency | P1 (Should have) |
    | **Labeler Training** | Tutorial tasks, certification tests, ongoing calibration | P1 (Should have) |
    | **Payment & Incentives** | Fair compensation, bonuses for quality, fraud detection | P1 (Should have) |
    | **API & SDK** | Programmatic task submission, label retrieval | P1 (Should have) |
    | **Analytics Dashboard** | Task progress, quality metrics, cost tracking | P2 (Nice to have) |
    | **Collaborative Labeling** | Team annotations, review workflows | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Model training infrastructure (separate ML pipeline)
    - Data storage for ML models
    - Model deployment and inference
    - Data collection/scraping services
    - Direct payment processing (use Stripe/PayPal)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Task Assignment Latency** | < 100ms p95 | Labelers need instant task delivery |
    | **Label Submission Latency** | < 200ms p95 | Real-time feedback on submissions |
    | **Availability** | 99.9% uptime | Global 24/7 workforce availability |
    | **Label Quality** | > 95% accuracy on golden tasks | High-quality training data requirement |
    | **Inter-Annotator Agreement** | Cohen's kappa > 0.75 | Substantial agreement threshold |
    | **Consensus Turnaround** | < 1 hour for 3 annotators | Fast iteration for ML teams |
    | **Fraud Detection** | < 0.1% fraudulent labels | Protect data quality |
    | **Payment Processing** | < 24 hours payment delay | Fair compensation for workers |
    | **Cost Efficiency** | < $0.10 per label average | Competitive pricing |
    | **Scalability** | Support 100K concurrent labelers | Handle demand spikes |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Labelers:
    - Active labelers: 10,000 concurrent labelers
    - Peak labelers: 20,000 concurrent (campaign launches)
    - Labeler sessions: 4 hours average
    - Labelers per day: 30,000 unique labelers

    Tasks:
    - Tasks per month: 10M tasks
    - Tasks per day: 10M / 30 = 333K tasks/day
    - Task creation QPS: 333K / 86,400 = 3.8 req/sec
    - Peak task creation: 50 req/sec (batch uploads)

    Task Assignments:
    - Tasks per labeler per hour: 20 tasks
    - Task requests per hour: 10K labelers × 20 = 200K requests/hour
    - Assignment QPS: 200K / 3,600 = 55 req/sec
    - Peak assignment QPS: 200 req/sec

    Label Submissions:
    - Annotations per task: 3 annotators for consensus
    - Total annotations per day: 333K × 3 = 1M annotations/day
    - Submission QPS: 1M / 86,400 = 11.5 req/sec
    - Peak submission QPS: 50 req/sec

    Quality Control:
    - Consensus calculations per task: 333K/day
    - Golden task validations: 10% of tasks = 33K/day
    - Inter-annotator agreement: computed per batch (1K tasks)

    Active Learning:
    - Model uncertainty scoring: every 10K new labels
    - Task prioritization: recalculate every hour

    Read/Write ratio: 5:1 (task reads vs label writes)
    ```

    ### Storage Estimates

    ```
    Tasks:
    - Task metadata: 2 KB per task (instructions, type, status)
    - Image tasks (70%): 333K × 0.7 = 233K tasks/day
      - Image references (URLs): included in metadata
      - Bounding box annotations: 500 bytes per annotation
    - Text tasks (20%): 67K tasks/day
      - Text content: 5 KB average
      - Classification labels: 200 bytes
    - Video tasks (10%): 33K tasks/day
      - Video frame references: 1 KB per task
      - Temporal annotations: 1 KB per annotation

    Task storage per day:
    - Metadata: 333K × 2 KB = 666 MB
    - Text content: 67K × 5 KB = 335 MB
    - Total: ~1 GB/day
    - 90-day retention: 90 GB

    Annotations:
    - Annotations per day: 1M annotations
    - Annotation data: 1 KB per annotation (label + metadata)
    - Daily annotations: 1M × 1 KB = 1 GB/day
    - 1-year retention: 365 GB ≈ 400 GB

    Labeler Profiles:
    - Labelers: 50,000 total labelers
    - Profile data: 5 KB per labeler (skills, stats, history)
    - 50K × 5 KB = 250 MB

    Quality Metrics:
    - Golden task results: 10% of tasks = 33K/day
    - Agreement scores: 100 bytes per task
    - Performance history: 33K × 100 bytes = 3.3 MB/day
    - 90-day retention: 300 MB

    Model State (Active Learning):
    - Uncertainty scores: 4 bytes per task
    - 333K × 4 bytes = 1.3 MB/day
    - Keep only active tasks: 100K × 4 bytes = 400 KB

    Audit Logs:
    - Assignment logs, submission logs: 500 bytes per event
    - 1M events/day × 500 bytes = 500 MB/day
    - 30-day retention: 15 GB

    Total: 90 GB (tasks) + 400 GB (annotations) + 250 MB (profiles) + 300 MB (quality) + 15 GB (logs) ≈ 500 GB
    ```

    ### Compute Estimates

    ```
    API Services:
    - Task assignment service: 20 instances × 4 vCPUs = 80 vCPUs
    - Label submission service: 15 instances × 4 vCPUs = 60 vCPUs
    - Quality control service: 10 instances × 8 vCPUs = 80 vCPUs
    - API gateway: 10 instances × 4 vCPUs = 40 vCPUs
    - Total: 260 vCPUs
    - Cost: 260 × $0.04/hour = $10.40/hour = $250/day

    Consensus Engine:
    - Consensus calculations: 333K tasks/day
    - CPU time per consensus: 100ms
    - Total compute: 333K × 100ms = 9.25 hours
    - Parallel workers: 10 instances × 8 vCPUs = 80 vCPUs
    - Cost: $64/day

    Active Learning Service:
    - Model inference for uncertainty: every 10K labels
    - GPU inference: 1 GPU × 24 hours = 1 GPU-day
    - Cost: 1 × $1.50/hour × 24 = $36/day

    Inter-Annotator Agreement:
    - Batch calculations: 333K tasks / 1K batch = 333 batches
    - CPU time per batch: 500ms
    - Total compute: 333 × 500ms = 2.8 hours
    - Workers: 5 instances × 4 vCPUs = 20 vCPUs
    - Cost: $16/day

    Fraud Detection:
    - Pattern analysis: 30K labelers/day
    - ML model inference: 1 GPU × 4 hours = 0.17 GPU-days
    - Cost: $6/day

    Total Daily Compute Cost: $250 + $64 + $36 + $16 + $6 = $372/day
    Cost per task: $372 / 333K = $0.001/task (compute only)
    ```

    ### Network Estimates

    ```
    Task uploads:
    - Image task references: 333K × 2 KB = 666 MB/day (URLs, not images)
    - Average: 666 MB / 86,400 = 7.7 KB/sec ≈ 0.06 Mbps
    - Peak (batch uploads): 10 Mbps

    Task assignments (reads):
    - 200K assignments/hour × 5 KB = 1 GB/hour = 278 KB/sec ≈ 2.2 Mbps
    - Peak: 10 Mbps

    Label submissions:
    - 1M submissions/day × 1 KB = 1 GB/day = 11.6 KB/sec ≈ 0.09 Mbps
    - Peak: 5 Mbps

    Labeling interface assets (images, videos):
    - Image loading: 200K assignments/hour × 500 KB = 100 GB/hour = 222 Mbps
    - Video loading (10%): 20K assignments/hour × 10 MB = 200 GB/hour = 444 Mbps
    - Total media: 666 Mbps average (CDN-served)

    Total bandwidth: 2.2 + 0.09 + 666 ≈ 670 Mbps average (mostly media)
    CDN bandwidth: 600 Mbps
    Origin bandwidth: 70 Mbps
    ```

---

=== "🏗️ Step 2: High-Level Design"

    ## Architecture Diagram

    ```mermaid
    graph TB
        User["👤 ML Team/Requester"]
        Labeler["👥 Labelers"]
        WebUI["🖥️ Web UI/Labeling Interface"]
        APIGateway["🚪 API Gateway"]

        subgraph "Task Management"
            TaskIngestion["📥 Task Ingestion Service"]
            TaskQueue["📋 Task Queue<br/>(Priority Queue)"]
            TaskRouter["🎯 Task Router<br/>(Skill Matching)"]
            ActiveLearning["🧠 Active Learning Service<br/>(Uncertainty Sampling)"]
        end

        subgraph "Assignment Engine"
            AssignmentService["🎲 Assignment Service"]
            SkillMatcher["🎓 Skill Matcher"]
            LoadBalancer["⚖️ Load Balancer<br/>(Fair Distribution)"]
            GoldenTaskInjector["⭐ Golden Task Injector"]
        end

        subgraph "Labeling Services"
            ImageAnnotator["🖼️ Image Annotator<br/>(Bbox, Polygon)"]
            TextAnnotator["📝 Text Annotator<br/>(Classification, NER)"]
            VideoAnnotator["🎥 Video Annotator<br/>(Temporal Segmentation)"]
            LabelValidator["✓ Label Validator"]
        end

        subgraph "Quality Control"
            ConsensusEngine["🤝 Consensus Engine<br/>(Majority/Weighted Voting)"]
            IAA["📊 Inter-Annotator Agreement<br/>(Cohen's Kappa)"]
            DisputeResolver["⚖️ Dispute Resolver<br/>(Expert Review)"]
            FraudDetector["🚨 Fraud Detector"]
        end

        subgraph "Workforce Management"
            LabelerRegistry["👤 Labeler Registry"]
            SkillTracker["🎯 Skill Tracker<br/>(Performance Metrics)"]
            TrainingService["📚 Training Service<br/>(Tutorials, Certification)"]
            PaymentService["💰 Payment Service<br/>(Fair Compensation)"]
        end

        subgraph "Storage"
            TaskDB["💾 Task DB<br/>(PostgreSQL)"]
            AnnotationDB["💾 Annotation DB<br/>(PostgreSQL)"]
            LabelerDB["💾 Labeler DB<br/>(PostgreSQL)"]
            QualityDB["📈 Quality Metrics DB<br/>(TimescaleDB)"]
            CacheLayer["⚡ Cache<br/>(Redis)"]
            ObjectStore["☁️ Object Storage<br/>(S3/GCS)"]
        end

        subgraph "Analytics & Monitoring"
            Dashboard["📊 Analytics Dashboard"]
            MetricsCollector["📈 Metrics Collector<br/>(Prometheus)"]
            AuditLogger["📝 Audit Logger"]
        end

        User --> APIGateway
        Labeler --> WebUI
        WebUI --> APIGateway

        APIGateway --> TaskIngestion
        APIGateway --> AssignmentService
        APIGateway --> LabelValidator

        TaskIngestion --> TaskQueue
        TaskQueue --> ActiveLearning
        ActiveLearning --> TaskRouter
        TaskRouter --> AssignmentService

        AssignmentService --> SkillMatcher
        AssignmentService --> LoadBalancer
        AssignmentService --> GoldenTaskInjector
        SkillMatcher --> LabelerRegistry

        LabelValidator --> ImageAnnotator
        LabelValidator --> TextAnnotator
        LabelValidator --> VideoAnnotator

        ImageAnnotator --> ConsensusEngine
        TextAnnotator --> ConsensusEngine
        VideoAnnotator --> ConsensusEngine

        ConsensusEngine --> IAA
        ConsensusEngine --> DisputeResolver
        ConsensusEngine --> FraudDetector

        IAA --> QualityDB
        FraudDetector --> SkillTracker
        SkillTracker --> PaymentService

        TaskIngestion --> TaskDB
        ConsensusEngine --> AnnotationDB
        LabelerRegistry --> LabelerDB

        AssignmentService --> CacheLayer
        ConsensusEngine --> CacheLayer

        TaskIngestion --> ObjectStore

        MetricsCollector --> Dashboard
        ConsensusEngine --> AuditLogger
    ```

    ---

    ## Component Responsibilities

    ### Task Management

    **Task Ingestion Service**
    - Accept bulk task uploads via API/SDK
    - Validate task format and instructions
    - Extract metadata (type, difficulty, requester)
    - Store task data in Task DB and ObjectStore
    - Enqueue tasks in Task Queue with priority

    **Task Queue**
    - Priority queue based on:
      - Requester SLA (enterprise > standard)
      - Task urgency (deadline-based)
      - Active learning score (high uncertainty first)
    - Distribute tasks across shards for scalability
    - Support task batching for related items

    **Task Router**
    - Match tasks to qualified labelers based on:
      - Required skills (image annotation, medical text, etc.)
      - Labeler performance history
      - Language requirements
      - Availability and timezone
    - Implement skill-based routing algorithms
    - Balance task distribution across labelers

    **Active Learning Service**
    - Integrate with ML models to compute uncertainty scores
    - Prioritize high-uncertainty tasks for labeling
    - Support multiple uncertainty metrics:
      - Prediction entropy
      - Margin sampling (difference between top-2 predictions)
      - Query by committee (variance across ensemble)
    - Trigger retraining when sufficient new labels collected

    ---

    ### Assignment Engine

    **Assignment Service**
    - Fetch next task for labeler based on routing rules
    - Track task assignments per labeler (prevent duplicate work)
    - Handle task timeouts and reassignments
    - Log all assignments for audit trail
    - Cache active assignments in Redis for fast lookup

    **Skill Matcher**
    - Query labeler skills from Labeler Registry
    - Match task requirements to labeler capabilities
    - Consider certification status and test scores
    - Apply filters (language, domain expertise)

    **Load Balancer**
    - Ensure fair task distribution across labelers
    - Prevent task hoarding (limit active tasks per labeler)
    - Balance between specialist and generalist labelers
    - Adjust based on real-time performance

    **Golden Task Injector**
    - Insert pre-labeled validation tasks (10% of stream)
    - Track labeler accuracy on golden tasks
    - Use for continuous performance monitoring
    - Trigger retraining if accuracy drops below threshold

    ---

    ### Quality Control

    **Consensus Engine**
    - Collect annotations from multiple labelers (typically 3)
    - Apply consensus algorithms:
      - **Majority voting**: Simple majority for discrete labels
      - **Weighted voting**: Weight by labeler accuracy
      - **DAWID-SKENE model**: EM algorithm for multi-class
      - **Bounding box IoU**: Average or max IoU for object detection
    - Resolve disagreements automatically when possible
    - Route disputed tasks to expert reviewers

    **Inter-Annotator Agreement (IAA)**
    - Calculate agreement metrics:
      - **Cohen's kappa**: Agreement between 2 annotators
      - **Fleiss' kappa**: Agreement among 3+ annotators
      - **Krippendorff's alpha**: General-purpose metric
      - **IoU**: For bounding box agreement
    - Identify problematic tasks (low agreement)
    - Flag ambiguous instructions for requester review

    **Dispute Resolver**
    - Identify tasks with low consensus (e.g., 33% agreement on 3 annotations)
    - Route to expert reviewers or senior labelers
    - Maintain queue of disputed tasks
    - Provide side-by-side annotation comparison
    - Store final adjudicated label

    **Fraud Detector**
    - Detect suspicious patterns:
      - Unusually fast submissions (< 5 seconds per task)
      - Low accuracy on golden tasks (< 70%)
      - Random clicking patterns
      - Copy-paste from other labelers (collusion)
    - Flag or ban fraudulent labelers
    - Invalidate fraudulent annotations

    ---

    ### Workforce Management

    **Labeler Registry**
    - Store labeler profiles (skills, languages, certifications)
    - Track onboarding status and training completion
    - Maintain performance history (accuracy, speed, consistency)
    - Manage availability and timezone preferences

    **Skill Tracker**
    - Compute real-time accuracy from golden tasks
    - Track inter-annotator agreement participation
    - Calculate earnings and bonus eligibility
    - Identify top performers for promotion to expert reviewers

    **Training Service**
    - Provide interactive tutorials for each annotation type
    - Administer certification tests
    - Deliver calibration tasks for ongoing training
    - Track training completion and recertification

    **Payment Service**
    - Calculate earnings based on task completion
    - Apply bonuses for high accuracy (e.g., 20% bonus for 95%+ accuracy)
    - Withhold payment for fraudulent work
    - Integrate with payment providers (Stripe, PayPal)
    - Handle international payments and currencies

    ---

    ## Data Flow

    ### Task Lifecycle

    ```mermaid
    sequenceDiagram
        participant Requester
        participant TaskIngestion
        participant TaskQueue
        participant ActiveLearning
        participant Assignment
        participant Labeler
        participant Consensus
        participant Quality

        Requester->>TaskIngestion: Upload tasks (images, text)
        TaskIngestion->>TaskQueue: Enqueue with metadata
        TaskQueue->>ActiveLearning: Score uncertainty
        ActiveLearning->>TaskQueue: Update priority

        Labeler->>Assignment: Request next task
        Assignment->>TaskQueue: Fetch high-priority task
        TaskQueue-->>Assignment: Return task + golden task
        Assignment-->>Labeler: Deliver task with instructions

        Labeler->>Consensus: Submit annotation

        alt 3 annotations collected
            Consensus->>Quality: Calculate IAA
            Quality->>Consensus: Kappa score

            alt High agreement (kappa > 0.75)
                Consensus->>Requester: Return consensus label
            else Low agreement (kappa < 0.5)
                Consensus->>DisputeResolver: Route to expert
                DisputeResolver->>Requester: Return adjudicated label
            end
        else < 3 annotations
            Consensus->>TaskQueue: Re-enqueue for more annotators
        end
    ```

    ---

    ## API Design

    ### Task Management APIs

    ```python
    # Upload tasks
    POST /api/v1/tasks/batch
    {
      "project_id": "proj_123",
      "tasks": [
        {
          "type": "image_bbox",
          "data": {"image_url": "s3://bucket/img1.jpg"},
          "instructions": "Draw boxes around all cars",
          "metadata": {"priority": "high"}
        }
      ]
    }
    Response: {"task_ids": ["task_001", "task_002"]}

    # Get task status
    GET /api/v1/tasks/{task_id}
    Response: {
      "task_id": "task_001",
      "status": "completed",
      "annotations": [
        {"labeler_id": "lab_1", "label": {...}},
        {"labeler_id": "lab_2", "label": {...}}
      ],
      "consensus": {...},
      "agreement_score": 0.85
    }
    ```

    ### Labeler Assignment APIs

    ```python
    # Get next task
    GET /api/v1/labelers/{labeler_id}/next_task
    Response: {
      "task_id": "task_001",
      "type": "image_bbox",
      "data": {"image_url": "..."},
      "instructions": "...",
      "is_golden": false
    }

    # Submit annotation
    POST /api/v1/tasks/{task_id}/annotations
    {
      "labeler_id": "lab_1",
      "annotation": {
        "type": "bbox",
        "boxes": [
          {"x": 100, "y": 150, "w": 200, "h": 180, "class": "car"}
        ]
      },
      "time_spent_sec": 45
    }
    Response: {"annotation_id": "ann_001", "golden_feedback": {...}}
    ```

    ### Quality Control APIs

    ```python
    # Get consensus
    GET /api/v1/tasks/{task_id}/consensus
    Response: {
      "consensus_label": {...},
      "confidence": 0.92,
      "agreement_score": 0.85,
      "method": "weighted_voting",
      "annotations": [...]
    }

    # Get IAA metrics
    GET /api/v1/projects/{project_id}/iaa
    Response: {
      "cohens_kappa": 0.78,
      "fleiss_kappa": 0.82,
      "tasks_analyzed": 1000
    }
    ```

---

=== "🔍 Step 3: Deep Dive"

    ## Task Routing with Skill Matching

    ### Routing Algorithm

    Match tasks to labelers based on a scoring function that considers:
    1. **Skill match**: Labeler has required skills
    2. **Performance**: Historical accuracy on similar tasks
    3. **Availability**: Currently active and not overloaded
    4. **Diversity**: Minimize overlap with previous annotators on same task

    ```python
    class TaskRouter:
        def __init__(self, labeler_db, task_db, cache):
            self.labeler_db = labeler_db
            self.task_db = task_db
            self.cache = cache

        def get_eligible_labelers(self, task):
            """Find labelers who can work on this task."""
            required_skills = task.required_skills
            required_language = task.language

            # Query labelers with required skills
            eligible_labelers = self.labeler_db.query(
                skills__contains=required_skills,
                languages__contains=required_language,
                status='active',
                banned=False
            )

            # Filter out labelers already assigned to this task
            previous_annotators = self.get_previous_annotators(task.id)
            eligible_labelers = [
                l for l in eligible_labelers
                if l.id not in previous_annotators
            ]

            return eligible_labelers

        def score_labeler_for_task(self, labeler, task):
            """Score how well a labeler matches a task."""
            score = 0.0

            # Skill match (0-40 points)
            skill_overlap = len(
                set(labeler.skills) & set(task.required_skills)
            )
            score += skill_overlap * 10

            # Performance on similar tasks (0-30 points)
            historical_accuracy = self.get_historical_accuracy(
                labeler.id, task.type
            )
            score += historical_accuracy * 30

            # Availability (0-20 points)
            active_tasks = self.cache.get(f"labeler:{labeler.id}:active_tasks")
            if active_tasks < 5:
                score += 20
            elif active_tasks < 10:
                score += 10

            # Recency (0-10 points): prefer labelers not recently assigned
            last_assignment = self.cache.get(f"labeler:{labeler.id}:last_assignment")
            minutes_since_last = (time.time() - last_assignment) / 60
            if minutes_since_last > 30:
                score += 10
            elif minutes_since_last > 15:
                score += 5

            return score

        def route_task(self, task):
            """Select best labeler for a task."""
            eligible_labelers = self.get_eligible_labelers(task)

            if not eligible_labelers:
                return None  # No eligible labelers

            # Score each labeler
            scored_labelers = [
                (labeler, self.score_labeler_for_task(labeler, task))
                for labeler in eligible_labelers
            ]

            # Sort by score descending
            scored_labelers.sort(key=lambda x: x[1], reverse=True)

            # Add randomness to top 5 to avoid always assigning to same person
            top_candidates = scored_labelers[:5]
            weights = [score for _, score in top_candidates]
            selected_labeler = random.choices(
                [l for l, _ in top_candidates],
                weights=weights,
                k=1
            )[0]

            return selected_labeler

        def get_historical_accuracy(self, labeler_id, task_type):
            """Get labeler's accuracy on similar tasks."""
            cache_key = f"labeler:{labeler_id}:accuracy:{task_type}"
            cached_accuracy = self.cache.get(cache_key)

            if cached_accuracy:
                return float(cached_accuracy)

            # Query from DB
            golden_results = self.labeler_db.query_golden_tasks(
                labeler_id=labeler_id,
                task_type=task_type,
                limit=100
            )

            if not golden_results:
                return 0.7  # Default neutral score

            correct = sum(1 for r in golden_results if r.correct)
            accuracy = correct / len(golden_results)

            # Cache for 1 hour
            self.cache.setex(cache_key, 3600, accuracy)

            return accuracy

        def get_previous_annotators(self, task_id):
            """Get labelers who already annotated this task."""
            cache_key = f"task:{task_id}:annotators"
            annotators = self.cache.smembers(cache_key)
            return annotators if annotators else set()
    ```

    ---

    ## Quality Control with Consensus

    ### Consensus Algorithms

    #### 1. Majority Voting (Classification Tasks)

    ```python
    def majority_voting(annotations, labeler_weights=None):
        """
        Simple majority voting for classification tasks.

        Args:
            annotations: List of labels from different annotators
            labeler_weights: Optional weights per labeler (default: equal)

        Returns:
            consensus_label, confidence
        """
        if labeler_weights is None:
            labeler_weights = [1.0] * len(annotations)

        # Count weighted votes
        vote_counts = {}
        total_weight = 0.0

        for label, weight in zip(annotations, labeler_weights):
            vote_counts[label] = vote_counts.get(label, 0) + weight
            total_weight += weight

        # Find majority
        consensus_label = max(vote_counts, key=vote_counts.get)
        confidence = vote_counts[consensus_label] / total_weight

        return consensus_label, confidence

    # Example usage
    annotations = ["cat", "cat", "dog"]
    labeler_weights = [0.95, 0.85, 0.90]  # Based on historical accuracy

    consensus, conf = majority_voting(annotations, labeler_weights)
    # consensus = "cat", conf = 0.67
    ```

    #### 2. Weighted Voting (Performance-Based)

    ```python
    class WeightedVotingConsensus:
        def __init__(self, labeler_db):
            self.labeler_db = labeler_db

        def get_labeler_weight(self, labeler_id, task_type):
            """Calculate weight based on historical accuracy."""
            accuracy = self.labeler_db.get_accuracy(labeler_id, task_type)

            # Use softmax-like transformation
            # High accuracy (0.95) -> weight ~2.5
            # Medium accuracy (0.80) -> weight ~1.0
            # Low accuracy (0.60) -> weight ~0.3
            weight = np.exp(3 * (accuracy - 0.8))
            return weight

        def compute_consensus(self, task_id, annotations):
            """Compute weighted consensus for a task."""
            labeler_ids = [ann.labeler_id for ann in annotations]
            labels = [ann.label for ann in annotations]
            task_type = annotations[0].task_type

            # Get weights
            weights = [
                self.get_labeler_weight(lid, task_type)
                for lid in labeler_ids
            ]

            # Weighted voting
            consensus_label, confidence = majority_voting(labels, weights)

            return {
                "consensus_label": consensus_label,
                "confidence": confidence,
                "method": "weighted_voting",
                "weights": weights,
                "annotations": [
                    {"labeler_id": lid, "label": l, "weight": w}
                    for lid, l, w in zip(labeler_ids, labels, weights)
                ]
            }
    ```

    #### 3. Bounding Box Consensus (Object Detection)

    ```python
    def bbox_iou(box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
        y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def cluster_bounding_boxes(boxes, iou_threshold=0.5):
        """Cluster bounding boxes using IoU similarity."""
        clusters = []

        for box in boxes:
            # Find cluster with IoU > threshold
            matched_cluster = None
            for cluster in clusters:
                # Check IoU with any box in cluster
                if any(bbox_iou(box, b) > iou_threshold for b in cluster):
                    matched_cluster = cluster
                    break

            if matched_cluster:
                matched_cluster.append(box)
            else:
                clusters.append([box])

        return clusters

    def bbox_consensus(annotations, iou_threshold=0.5, min_agreement=2):
        """
        Compute consensus bounding boxes.

        Args:
            annotations: List of bbox annotations from different labelers
            iou_threshold: IoU threshold for clustering
            min_agreement: Minimum annotators for consensus

        Returns:
            List of consensus bounding boxes
        """
        # Flatten all boxes from all annotators
        all_boxes = []
        for ann in annotations:
            for box in ann['boxes']:
                all_boxes.append({
                    **box,
                    'labeler_id': ann['labeler_id']
                })

        # Cluster boxes
        clusters = cluster_bounding_boxes(all_boxes, iou_threshold)

        # Keep only clusters with min_agreement
        consensus_boxes = []
        for cluster in clusters:
            if len(cluster) >= min_agreement:
                # Average box coordinates
                avg_box = {
                    'x': np.mean([b['x'] for b in cluster]),
                    'y': np.mean([b['y'] for b in cluster]),
                    'w': np.mean([b['w'] for b in cluster]),
                    'h': np.mean([b['h'] for b in cluster]),
                    'class': cluster[0]['class'],  # Assume same class in cluster
                    'agreement': len(cluster) / len(annotations),
                    'annotators': [b['labeler_id'] for b in cluster]
                }
                consensus_boxes.append(avg_box)

        return consensus_boxes

    # Example usage
    annotations = [
        {
            'labeler_id': 'lab_1',
            'boxes': [
                {'x': 100, 'y': 150, 'w': 200, 'h': 180, 'class': 'car'},
                {'x': 400, 'y': 200, 'w': 150, 'h': 120, 'class': 'person'}
            ]
        },
        {
            'labeler_id': 'lab_2',
            'boxes': [
                {'x': 105, 'y': 155, 'w': 195, 'h': 175, 'class': 'car'},
                {'x': 405, 'y': 205, 'w': 145, 'h': 115, 'class': 'person'}
            ]
        },
        {
            'labeler_id': 'lab_3',
            'boxes': [
                {'x': 98, 'y': 148, 'w': 205, 'h': 185, 'class': 'car'}
            ]
        }
    ]

    consensus = bbox_consensus(annotations, iou_threshold=0.5, min_agreement=2)
    # Returns 2 boxes (car and person) with averaged coordinates
    ```

    ---

    ## Inter-Annotator Agreement (IAA)

    ### Cohen's Kappa (2 Annotators)

    ```python
    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    class InterAnnotatorAgreement:
        @staticmethod
        def cohens_kappa(annotations1, annotations2):
            """
            Calculate Cohen's kappa for 2 annotators.

            Interpretation:
            - < 0: No agreement
            - 0.01-0.20: None to slight
            - 0.21-0.40: Fair
            - 0.41-0.60: Moderate
            - 0.61-0.80: Substantial
            - 0.81-1.00: Almost perfect
            """
            return cohen_kappa_score(annotations1, annotations2)

        @staticmethod
        def fleiss_kappa(annotations_matrix):
            """
            Calculate Fleiss' kappa for 3+ annotators.

            Args:
                annotations_matrix: n_tasks × n_categories matrix
                                   where each cell is count of annotators
                                   who assigned that category

            Example:
                Task 1: [2, 1, 0] -> 2 annotators said "cat", 1 said "dog"
                Task 2: [0, 3, 0] -> 3 annotators said "dog"
            """
            n_tasks, n_categories = annotations_matrix.shape
            n_annotators = annotations_matrix.sum(axis=1)[0]

            # Proportion of annotators per category per task
            p_matrix = annotations_matrix / n_annotators

            # Observed agreement per task
            P_i = (annotations_matrix ** 2).sum(axis=1) - n_annotators
            P_i = P_i / (n_annotators * (n_annotators - 1))
            P_bar = P_i.mean()  # Average observed agreement

            # Expected agreement
            p_j = annotations_matrix.sum(axis=0) / (n_tasks * n_annotators)
            P_e_bar = (p_j ** 2).sum()

            # Fleiss' kappa
            kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

            return kappa

        @staticmethod
        def krippendorff_alpha(annotations_matrix, metric='nominal'):
            """
            Krippendorff's alpha for any number of annotators and data types.
            More general than Cohen's or Fleiss' kappa.

            Args:
                annotations_matrix: n_annotators × n_tasks matrix
                metric: 'nominal', 'ordinal', 'interval', 'ratio'
            """
            # Implementation omitted for brevity (complex)
            # Use krippendorff package: krippendorff.alpha(annotations_matrix)
            pass

        @staticmethod
        def bbox_agreement_iou(annotations1, annotations2, iou_threshold=0.5):
            """
            Calculate agreement for bounding box annotations.

            Returns:
                - Precision: % of boxes in ann1 that match ann2
                - Recall: % of boxes in ann2 that match ann1
                - F1: Harmonic mean
            """
            boxes1 = annotations1['boxes']
            boxes2 = annotations2['boxes']

            matched1 = 0
            for box1 in boxes1:
                if any(bbox_iou(box1, box2) > iou_threshold for box2 in boxes2):
                    matched1 += 1

            matched2 = 0
            for box2 in boxes2:
                if any(bbox_iou(box2, box1) > iou_threshold for box1 in boxes1):
                    matched2 += 1

            precision = matched1 / len(boxes1) if boxes1 else 0
            recall = matched2 / len(boxes2) if boxes2 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {'precision': precision, 'recall': recall, 'f1': f1}

    # Example usage
    iaa = InterAnnotatorAgreement()

    # Classification task
    ann1 = ["cat", "dog", "cat", "bird", "cat"]
    ann2 = ["cat", "dog", "dog", "bird", "cat"]
    kappa = iaa.cohens_kappa(ann1, ann2)
    print(f"Cohen's kappa: {kappa:.3f}")  # Output: 0.737 (substantial agreement)

    # Multi-annotator task
    matrix = np.array([
        [2, 1, 0],  # Task 1: 2 said "cat", 1 said "dog"
        [3, 0, 0],  # Task 2: 3 said "cat"
        [1, 2, 0],  # Task 3: 1 said "cat", 2 said "dog"
        [0, 0, 3],  # Task 4: 3 said "bird"
    ])
    fleiss = iaa.fleiss_kappa(matrix)
    print(f"Fleiss' kappa: {fleiss:.3f}")
    ```

    ---

    ## Active Learning for Task Selection

    ### Uncertainty Sampling

    ```python
    import numpy as np
    from scipy.stats import entropy

    class ActiveLearningSelector:
        def __init__(self, model, task_queue):
            self.model = model
            self.task_queue = task_queue

        def entropy_sampling(self, tasks, top_k=100):
            """
            Select tasks with highest prediction entropy.

            Entropy = -Σ p(class) * log(p(class))
            High entropy = uncertain prediction
            """
            uncertainties = []

            for task in tasks:
                # Get model predictions
                probs = self.model.predict_proba(task.data)

                # Calculate entropy
                ent = entropy(probs)
                uncertainties.append((task.id, ent))

            # Sort by entropy descending
            uncertainties.sort(key=lambda x: x[1], reverse=True)

            # Return top-k most uncertain
            return [task_id for task_id, _ in uncertainties[:top_k]]

        def margin_sampling(self, tasks, top_k=100):
            """
            Select tasks with smallest margin between top 2 predictions.

            Margin = p(class_1) - p(class_2)
            Small margin = model is uncertain between 2 classes
            """
            uncertainties = []

            for task in tasks:
                probs = self.model.predict_proba(task.data)

                # Sort probabilities descending
                sorted_probs = np.sort(probs)[::-1]

                # Margin between top 2
                margin = sorted_probs[0] - sorted_probs[1]
                uncertainties.append((task.id, margin))

            # Sort by margin ascending (small margin = uncertain)
            uncertainties.sort(key=lambda x: x[1])

            return [task_id for task_id, _ in uncertainties[:top_k]]

        def query_by_committee(self, tasks, committee_models, top_k=100):
            """
            Select tasks with highest disagreement among committee of models.

            Committee: ensemble of models trained on different subsets
            Disagreement: variance in predictions
            """
            uncertainties = []

            for task in tasks:
                # Get predictions from all models
                predictions = [
                    model.predict_proba(task.data)
                    for model in committee_models
                ]

                # Calculate variance across committee
                variance = np.var(predictions, axis=0).mean()
                uncertainties.append((task.id, variance))

            # Sort by variance descending
            uncertainties.sort(key=lambda x: x[1], reverse=True)

            return [task_id for task_id, _ in uncertainties[:top_k]]

        def least_confidence_sampling(self, tasks, top_k=100):
            """
            Select tasks where model has lowest confidence in top prediction.

            Confidence = max(p(class))
            Low confidence = uncertain
            """
            uncertainties = []

            for task in tasks:
                probs = self.model.predict_proba(task.data)
                confidence = np.max(probs)
                uncertainties.append((task.id, confidence))

            # Sort by confidence ascending (low confidence = uncertain)
            uncertainties.sort(key=lambda x: x[1])

            return [task_id for task_id, _ in uncertainties[:top_k]]

        def update_priorities(self, method='entropy', top_k=1000):
            """
            Recalculate task priorities based on active learning.

            Called periodically (e.g., every 10K new labels).
            """
            # Get unlabeled tasks
            unlabeled_tasks = self.task_queue.get_unlabeled_tasks(limit=10000)

            if method == 'entropy':
                priority_tasks = self.entropy_sampling(unlabeled_tasks, top_k)
            elif method == 'margin':
                priority_tasks = self.margin_sampling(unlabeled_tasks, top_k)
            elif method == 'committee':
                committee_models = self.get_committee_models()
                priority_tasks = self.query_by_committee(
                    unlabeled_tasks, committee_models, top_k
                )
            else:
                priority_tasks = self.least_confidence_sampling(unlabeled_tasks, top_k)

            # Update task priorities in queue
            self.task_queue.set_priorities(priority_tasks, priority='high')

            return len(priority_tasks)

    # Example usage
    selector = ActiveLearningSelector(model, task_queue)

    # Run active learning every 10K labels
    if new_labels_count % 10000 == 0:
        num_prioritized = selector.update_priorities(method='entropy', top_k=1000)
        print(f"Prioritized {num_prioritized} high-uncertainty tasks")
    ```

    ---

    ## Dispute Resolution

    ```python
    class DisputeResolver:
        def __init__(self, consensus_engine, expert_pool):
            self.consensus_engine = consensus_engine
            self.expert_pool = expert_pool

        def identify_disputes(self, task_id, annotations):
            """
            Identify if task needs dispute resolution.

            Dispute criteria:
            - Low consensus confidence (< 0.6)
            - Low inter-annotator agreement (kappa < 0.5)
            - Disagreement on critical features
            """
            consensus = self.consensus_engine.compute_consensus(task_id, annotations)

            # Check confidence
            if consensus['confidence'] < 0.6:
                return True, "low_confidence"

            # Check inter-annotator agreement
            if len(annotations) >= 2:
                labels = [ann.label for ann in annotations]
                if len(set(labels)) == len(labels):  # All different
                    return True, "complete_disagreement"

                # Calculate kappa
                iaa = InterAnnotatorAgreement()
                if len(annotations) == 2:
                    kappa = iaa.cohens_kappa(labels[:1], labels[1:2])
                else:
                    # Convert to matrix for Fleiss' kappa
                    matrix = self._labels_to_matrix(labels)
                    kappa = iaa.fleiss_kappa(matrix)

                if kappa < 0.5:
                    return True, "low_agreement"

            return False, None

        def route_to_expert(self, task_id, annotations, dispute_reason):
            """Route disputed task to expert reviewer."""
            # Select expert based on:
            # 1. Domain expertise matching task
            # 2. Availability
            # 3. Load balancing

            task = self.get_task(task_id)
            expert = self.expert_pool.select_expert(
                domain=task.domain,
                available=True
            )

            if not expert:
                # No expert available, queue for later
                self.dispute_queue.enqueue({
                    'task_id': task_id,
                    'annotations': annotations,
                    'reason': dispute_reason,
                    'priority': self._calculate_priority(task)
                })
                return None

            # Create expert review task
            review_task = {
                'task_id': task_id,
                'original_task': task,
                'annotations': annotations,
                'dispute_reason': dispute_reason,
                'expert_id': expert.id,
                'deadline': time.time() + 3600  # 1 hour
            }

            self.expert_queue.enqueue(review_task)

            return expert.id

        def expert_review(self, review_task, expert_annotation):
            """Process expert review and resolve dispute."""
            task_id = review_task['task_id']

            # Store expert decision as final label
            final_annotation = {
                'task_id': task_id,
                'labeler_id': review_task['expert_id'],
                'label': expert_annotation,
                'is_expert': True,
                'resolved_dispute': True,
                'original_annotations': review_task['annotations']
            }

            self.annotation_db.store(final_annotation)

            # Update task status
            self.task_db.update_status(task_id, 'completed')

            # Optionally: provide feedback to original labelers
            self._send_feedback_to_labelers(
                review_task['annotations'],
                expert_annotation
            )

            return final_annotation

        def _calculate_priority(self, task):
            """Calculate dispute priority based on task importance."""
            priority = 0

            # Requester SLA
            if task.requester_tier == 'enterprise':
                priority += 10

            # Deadline urgency
            time_to_deadline = task.deadline - time.time()
            if time_to_deadline < 3600:  # < 1 hour
                priority += 20
            elif time_to_deadline < 86400:  # < 1 day
                priority += 10

            # Task complexity
            if task.complexity == 'high':
                priority += 5

            return priority
    ```

---

=== "⚡ Step 4: Scale & Optimize"

    ## Scaling Strategies

    ### 1. Distributed Task Queue

    **Challenge:** Single task queue becomes bottleneck at scale (100K+ concurrent labelers)

    **Solution:** Shard task queue by task type and region

    ```python
    class ShardedTaskQueue:
        def __init__(self, redis_cluster):
            self.redis = redis_cluster
            self.num_shards = 32  # Based on expected load

        def get_shard_key(self, task_type, region):
            """Determine shard for task based on type and region."""
            # Hash to distribute evenly
            shard_id = hash(f"{task_type}:{region}") % self.num_shards
            return f"task_queue:shard:{shard_id}"

        def enqueue(self, task):
            """Add task to appropriate shard."""
            shard_key = self.get_shard_key(task.type, task.region)

            # Use sorted set with priority as score
            self.redis.zadd(shard_key, {
                task.id: task.priority
            })

        def dequeue_for_labeler(self, labeler):
            """Get next task for labeler from their shard."""
            # Determine labeler's shard based on preferences
            shard_key = self.get_shard_key(
                labeler.preferred_task_type,
                labeler.region
            )

            # Pop highest priority task
            result = self.redis.zpopmax(shard_key)

            if not result:
                # Try other shards if primary is empty
                return self._try_other_shards(labeler)

            task_id, priority = result[0]
            return task_id

        def _try_other_shards(self, labeler):
            """Fallback to other shards if primary is empty."""
            # Try up to 5 random shards
            for _ in range(5):
                shard_id = random.randint(0, self.num_shards - 1)
                shard_key = f"task_queue:shard:{shard_id}"
                result = self.redis.zpopmax(shard_key)
                if result:
                    return result[0][0]

            return None
    ```

    **Benefits:**
    - Parallel queue operations (32x throughput)
    - No single point of contention
    - Geographic distribution (lower latency)

    **Tradeoff:**
    - Complexity in load balancing across shards
    - May need rebalancing if shards become uneven

    ---

    ### 2. Caching Labeling Interfaces

    **Challenge:** Loading labeling UI assets (images, videos) for millions of tasks is expensive

    **Solution:** Multi-tier caching with CDN

    ```python
    class LabelingInterfaceCache:
        def __init__(self, cdn, redis_cache, s3):
            self.cdn = cdn
            self.redis = redis_cache
            self.s3 = s3

        def get_task_data(self, task_id):
            """
            Fetch task data with caching.

            Cache tiers:
            1. Redis (task metadata): 100ms
            2. CDN (media assets): 200ms
            3. S3 (origin): 500ms
            """
            # Try Redis first (metadata)
            cache_key = f"task:{task_id}:data"
            cached_data = self.redis.get(cache_key)

            if cached_data:
                return json.loads(cached_data)

            # Fetch from S3
            task_data = self.s3.get_object(f"tasks/{task_id}.json")

            # Cache in Redis (1 hour TTL)
            self.redis.setex(cache_key, 3600, json.dumps(task_data))

            # Warm CDN for media assets
            if task_data.get('image_url'):
                self._warm_cdn(task_data['image_url'])

            return task_data

        def _warm_cdn(self, asset_url):
            """Proactively cache media in CDN."""
            # CDN pulls from S3 on first request
            # Pre-fetch to CDN edge locations
            self.cdn.prefetch(asset_url, regions=['us-east', 'eu-west', 'ap-south'])

        def batch_prefetch(self, task_ids):
            """Prefetch next tasks for labeler."""
            # Predict next tasks labeler will see
            # Warm cache proactively
            for task_id in task_ids:
                self.get_task_data(task_id)
    ```

    **Benefits:**
    - 5x faster task loading (500ms → 100ms)
    - 90% reduction in S3 costs
    - Better user experience (instant task display)

    ---

    ### 3. Batching for Consensus Calculations

    **Challenge:** Computing consensus for millions of tasks individually is slow

    **Solution:** Batch consensus calculations

    ```python
    class BatchConsensusProcessor:
        def __init__(self, annotation_db, consensus_engine):
            self.annotation_db = annotation_db
            self.consensus_engine = consensus_engine
            self.batch_size = 1000

        def process_pending_consensus(self):
            """
            Process consensus in batches.

            Triggered every 5 minutes or when batch size reached.
            """
            # Find tasks with 3+ annotations but no consensus
            pending_tasks = self.annotation_db.query(
                num_annotations__gte=3,
                has_consensus=False,
                limit=10000
            )

            # Process in batches
            for i in range(0, len(pending_tasks), self.batch_size):
                batch = pending_tasks[i:i+self.batch_size]
                self._process_batch(batch)

        def _process_batch(self, tasks):
            """Process a batch of tasks in parallel."""
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for task in tasks:
                    future = executor.submit(
                        self._compute_consensus_for_task,
                        task
                    )
                    futures.append(future)

                # Wait for all to complete
                for future in futures:
                    future.result()

        def _compute_consensus_for_task(self, task):
            """Compute consensus for single task."""
            annotations = self.annotation_db.get_annotations(task.id)

            consensus = self.consensus_engine.compute_consensus(
                task.id, annotations
            )

            # Store consensus
            self.annotation_db.store_consensus(task.id, consensus)

            # Update task status
            if consensus['confidence'] > 0.8:
                self.annotation_db.update_task_status(task.id, 'completed')
            else:
                # Route to dispute resolution
                self.annotation_db.update_task_status(task.id, 'needs_review')
    ```

    **Benefits:**
    - 10x throughput (100 tasks/sec → 1000 tasks/sec)
    - Efficient use of compute resources
    - Reduced database load (batch writes)

    ---

    ### 4. Fraud Detection

    **Challenge:** Detect fraudulent labelers who submit random/low-quality labels

    **Solution:** Multi-signal fraud detection

    ```python
    class FraudDetector:
        def __init__(self, labeler_db, annotation_db):
            self.labeler_db = labeler_db
            self.annotation_db = annotation_db

        def detect_fraud(self, labeler_id):
            """
            Detect fraudulent behavior using multiple signals.

            Returns fraud score (0-1) and reasons.
            """
            fraud_score = 0.0
            reasons = []

            # Signal 1: Unusually fast submissions
            avg_time = self._get_avg_submission_time(labeler_id)
            if avg_time < 5:  # < 5 seconds per task
                fraud_score += 0.3
                reasons.append("suspiciously_fast")

            # Signal 2: Low accuracy on golden tasks
            golden_accuracy = self._get_golden_accuracy(labeler_id)
            if golden_accuracy < 0.6:
                fraud_score += 0.4
                reasons.append("low_golden_accuracy")

            # Signal 3: Random clicking pattern
            if self._detect_random_pattern(labeler_id):
                fraud_score += 0.3
                reasons.append("random_pattern")

            # Signal 4: Copying from other labelers
            if self._detect_collusion(labeler_id):
                fraud_score += 0.5
                reasons.append("possible_collusion")

            # Signal 5: Consistent disagreement with consensus
            disagreement_rate = self._get_disagreement_rate(labeler_id)
            if disagreement_rate > 0.7:
                fraud_score += 0.2
                reasons.append("high_disagreement")

            fraud_score = min(fraud_score, 1.0)

            return fraud_score, reasons

        def _detect_random_pattern(self, labeler_id):
            """Detect random clicking (e.g., always selecting first option)."""
            recent_annotations = self.annotation_db.get_recent(
                labeler_id, limit=100
            )

            # Check for patterns
            labels = [ann.label for ann in recent_annotations]

            # Always same label?
            if len(set(labels)) == 1:
                return True

            # Alternating pattern? (e.g., A, B, A, B, A, B)
            if self._is_alternating(labels):
                return True

            return False

        def _detect_collusion(self, labeler_id):
            """Detect if labeler is copying from others."""
            # Get tasks labeled by this labeler
            tasks = self.annotation_db.get_tasks_by_labeler(labeler_id, limit=100)

            # For each task, check if another labeler submitted very similar label
            collusion_count = 0
            for task in tasks:
                other_annotations = self.annotation_db.get_annotations(
                    task.id, exclude_labeler=labeler_id
                )

                if not other_annotations:
                    continue

                # Check timestamp proximity (within 10 seconds)
                labeler_ann = self.annotation_db.get_annotation(task.id, labeler_id)
                for other_ann in other_annotations:
                    time_diff = abs(labeler_ann.timestamp - other_ann.timestamp)
                    if time_diff < 10 and labeler_ann.label == other_ann.label:
                        collusion_count += 1
                        break

            # If > 50% of annotations have collusion signals
            return collusion_count / len(tasks) > 0.5

        def take_action(self, labeler_id, fraud_score, reasons):
            """Take action based on fraud score."""
            if fraud_score > 0.8:
                # Ban labeler
                self.labeler_db.update(labeler_id, status='banned')
                # Invalidate all annotations
                self.annotation_db.invalidate_labeler_annotations(labeler_id)
                return "banned"

            elif fraud_score > 0.5:
                # Suspend and require recertification
                self.labeler_db.update(labeler_id, status='suspended')
                # Trigger manual review of annotations
                return "suspended"

            elif fraud_score > 0.3:
                # Warning and increased monitoring
                self.labeler_db.update(labeler_id, warning_count=+1)
                # Increase golden task ratio to 30%
                self.labeler_db.update(labeler_id, golden_task_ratio=0.3)
                return "warning"

            return "no_action"
    ```

    **Benefits:**
    - Protect data quality
    - Reduce wasted labeling costs
    - Maintain platform reputation

    ---

    ### 5. Payment Optimization

    **Challenge:** Fair compensation while controlling costs

    **Solution:** Performance-based payment with bonuses

    ```python
    class PaymentOptimizer:
        def __init__(self, labeler_db, payment_service):
            self.labeler_db = labeler_db
            self.payment_service = payment_service

        def calculate_payment(self, labeler_id, tasks_completed):
            """
            Calculate payment with performance bonuses.

            Base rate: $0.05 per task
            Accuracy bonus: +20% for 95%+ accuracy
            Speed bonus: +10% for > 25 tasks/hour
            Consistency bonus: +10% for high IAA
            """
            base_rate = 0.05
            total_payment = 0.0

            for task in tasks_completed:
                task_payment = base_rate

                # Accuracy bonus
                accuracy = self._get_accuracy_on_task(labeler_id, task.id)
                if accuracy >= 0.95:
                    task_payment *= 1.20
                elif accuracy >= 0.85:
                    task_payment *= 1.10

                # Speed bonus (if high quality maintained)
                avg_speed = self._get_hourly_rate(labeler_id)
                if avg_speed >= 25 and accuracy >= 0.90:
                    task_payment *= 1.10

                # Consistency bonus (high IAA)
                iaa_score = self._get_iaa_contribution(labeler_id, task.id)
                if iaa_score >= 0.80:
                    task_payment *= 1.10

                total_payment += task_payment

            return total_payment

        def optimize_task_pricing(self, task_type, difficulty, deadline):
            """
            Dynamic pricing based on supply/demand.

            Increase price for:
            - Hard tasks (low supply of qualified labelers)
            - Urgent deadlines
            - Peak hours
            """
            base_price = 0.05

            # Difficulty multiplier
            if difficulty == 'hard':
                base_price *= 2.0
            elif difficulty == 'medium':
                base_price *= 1.5

            # Urgency multiplier
            time_to_deadline = deadline - time.time()
            if time_to_deadline < 3600:  # < 1 hour
                base_price *= 2.0
            elif time_to_deadline < 86400:  # < 1 day
                base_price *= 1.5

            # Supply/demand adjustment
            available_labelers = self._count_qualified_labelers(task_type)
            pending_tasks = self._count_pending_tasks(task_type)

            demand_ratio = pending_tasks / max(available_labelers, 1)
            if demand_ratio > 10:  # High demand
                base_price *= 1.3
            elif demand_ratio < 2:  # Low demand
                base_price *= 0.9

            return base_price
    ```

    **Benefits:**
    - Fair compensation attracts high-quality labelers
    - Cost control through performance incentives
    - Dynamic pricing matches supply/demand

    ---

    ## Monitoring & Metrics

    ### Key Metrics to Track

    ```python
    # Task metrics
    - Tasks created per day
    - Tasks completed per day
    - Average time to completion
    - Backlog size
    - Task abandonment rate

    # Quality metrics
    - Average inter-annotator agreement (Cohen's kappa, Fleiss' kappa)
    - Consensus confidence distribution
    - Golden task accuracy by labeler
    - Dispute rate (% tasks needing expert review)
    - Label quality score (validated by downstream ML model)

    # Labeler metrics
    - Active labelers per hour
    - Average tasks per labeler per hour
    - Labeler retention rate (7-day, 30-day)
    - Earnings per labeler per day
    - Fraud detection rate

    # Performance metrics
    - Task assignment latency (p50, p95, p99)
    - Label submission latency
    - Consensus calculation latency
    - API response time

    # Cost metrics
    - Cost per label
    - Compute cost per task
    - Storage cost per task
    - Payment processing fees
    ```

    ### Alerting Thresholds

    ```python
    # Critical alerts
    - Inter-annotator agreement drops below 0.6 (data quality issue)
    - Fraud detection rate > 5% (platform security issue)
    - Task backlog > 1M tasks (capacity issue)
    - API latency p95 > 500ms (performance issue)

    # Warning alerts
    - Average time to completion > 2 hours (labeler engagement issue)
    - Labeler retention rate < 50% (platform UX issue)
    - Cost per label > $0.15 (cost efficiency issue)
    ```

---

=== "📚 References & Real-World Implementation"

    ## Industry Examples

    ### Labelbox Architecture

    **Scale:** 10K+ organizations, millions of tasks daily

    **Key Features:**
    - Unified labeling platform for images, text, video, audio
    - Automated quality control with consensus
    - Integrated model-assisted labeling (active learning)
    - Workflow automation and review pipelines
    - Enterprise SSO and data governance

    **Technical Highlights:**
    - **Task routing:** Skill-based assignment with performance tracking
    - **Quality control:** Configurable consensus (2-5 annotators)
    - **Active learning:** Model-in-the-loop for uncertainty sampling
    - **APIs:** GraphQL API for programmatic access
    - **Integrations:** Connect to AWS, GCP, Azure storage

    ---

    ### Scale AI Architecture

    **Scale:** Largest data labeling company, billions of annotations

    **Key Features:**
    - Managed workforce of 300K+ labelers
    - Specialized teams for autonomous vehicles, NLP, document AI
    - Rapid API for real-time labeling
    - Studio for custom labeling workflows
    - 99%+ accuracy guarantee with multi-layer QA

    **Technical Highlights:**
    - **Task routing:** AI-powered assignment to specialized labelers
    - **Quality control:** 5-stage QA (initial label → peer review → expert review → automated QA → customer acceptance)
    - **Consensus:** Weighted voting based on labeler expertise scores
    - **Active learning:** Prioritize edge cases and low-confidence predictions
    - **Fraud detection:** Real-time anomaly detection and manual audits

    **Architecture Pattern:**
    ```
    Client API
      ↓
    Task Ingestion (API Gateway + Kafka)
      ↓
    Task Router (Assignment Service + Redis)
      ↓
    Labeling Platform (React + WebSockets)
      ↓
    Quality Control (Consensus Engine + Expert Review)
      ↓
    Client Delivery (API + Webhooks)
    ```

    ---

    ### Amazon SageMaker Ground Truth

    **Scale:** AWS-managed service, integrated with ML workflows

    **Key Features:**
    - Built-in labeling workflows (image classification, object detection, NER)
    - Private workforce (internal teams) or public (Mechanical Turk)
    - Active learning to reduce labeling costs by 70%
    - Automated data labeling with human verification
    - Integrated with SageMaker training pipelines

    **Technical Highlights:**
    - **Active learning:** Pre-label with models, humans verify uncertain predictions
    - **Consensus:** Configurable (1-9 annotators per task)
    - **Quality control:** Automated and manual verification
    - **Cost optimization:** Use models to label easy tasks, humans for hard cases

    ---

    ## Design Tradeoffs

    ### Quality vs Speed

    **Scenario:** High-quality labels with 3+ annotators vs fast turnaround with 1 annotator

    **Tradeoff:**
    - **High quality (3+ annotators):**
      - Pros: Better consensus, catches errors, higher accuracy
      - Cons: 3x cost, 3x slower, requires more labelers
    - **Fast (1 annotator):**
      - Pros: Faster, cheaper, simpler workflow
      - Cons: No validation, single point of failure, lower accuracy

    **Solution:** Adaptive quality control
    - Use 1 annotator for easy tasks (high model confidence)
    - Use 3+ annotators for hard tasks (low model confidence)
    - Insert golden tasks for continuous validation

    ---

    ### Cost vs Accuracy

    **Scenario:** Expert labelers ($5/hour) vs crowd labelers ($3/hour)

    **Tradeoff:**
    - **Expert labelers:**
      - Pros: Higher accuracy (95%+), faster (domain knowledge), fewer errors
      - Cons: Higher cost, limited availability, scaling challenges
    - **Crowd labelers:**
      - Pros: Lower cost, unlimited scale, 24/7 availability
      - Cons: Lower accuracy (80-85%), requires more QA, training overhead

    **Solution:** Hybrid approach
    - Use crowd for simple tasks (image classification)
    - Use experts for complex tasks (medical imaging, legal documents)
    - Use experts for dispute resolution and calibration

    ---

    ### Consensus Algorithm Selection

    | Algorithm | Use Case | Pros | Cons |
    |-----------|----------|------|------|
    | **Majority Voting** | Classification tasks | Simple, fast, easy to explain | Equal weight to all labelers |
    | **Weighted Voting** | All tasks | Rewards quality, discourages fraud | Requires tracking performance |
    | **DAWID-SKENE** | Multi-class with noise | Handles varying labeler quality | Computationally expensive |
    | **Bounding Box IoU** | Object detection | Precise spatial agreement | Doesn't handle missing/extra boxes well |

    **Recommendation:** Use weighted voting for most cases, DAWID-SKENE for high-noise environments

    ---

    ## Additional Resources

    ### Research Papers

    1. **"Learning from Crowds" (JMLR 2010)** - Foundational paper on consensus mechanisms
    2. **"Active Learning for Deep Object Detection" (BMVC 2018)** - Active learning for vision tasks
    3. **"Cohen's Kappa: Measure of Agreement" (1960)** - Inter-annotator agreement metric
    4. **"DAWID-SKENE Model" (1979)** - EM algorithm for multi-annotator consensus

    ### Open Source Tools

    - **Label Studio:** Open-source labeling platform with ML-assisted labeling
    - **CVAT:** Computer Vision Annotation Tool for images and videos
    - **Doccano:** Text annotation for NLP (classification, NER, seq2seq)
    - **Prodigy:** Scriptable annotation tool with active learning

    ### Best Practices

    1. **Golden tasks:** Insert 10-20% pre-labeled validation tasks
    2. **Training:** Require certification tests before production labeling
    3. **Calibration:** Regular re-certification (monthly) for labelers
    4. **Feedback:** Provide real-time feedback on golden task performance
    5. **Instructions:** Clear, visual instructions with examples
    6. **Dispute resolution:** Expert review for low-consensus tasks
    7. **Fraud detection:** Multi-signal detection with human review
    8. **Payment:** Fair compensation with performance bonuses
    9. **Active learning:** Integrate with ML models to prioritize tasks
    10. **Monitoring:** Track quality metrics (IAA, accuracy) continuously

---

=== "❓ Common Interview Questions"

    ## Question 1: How would you handle a task that receives 3 completely different labels?

    **Answer:**

    This is a dispute scenario requiring multi-stage resolution:

    1. **Check inter-annotator agreement:**
       - Calculate Fleiss' kappa for the 3 annotations
       - If kappa < 0.5, flag as low agreement

    2. **Route to expert review:**
       - Select domain expert based on task type
       - Provide side-by-side comparison of 3 annotations
       - Expert provides final adjudicated label

    3. **Provide feedback to labelers:**
       - Show expert decision to original 3 labelers
       - Use as training example for similar tasks
       - Update labeler performance scores

    4. **Check task instructions:**
       - If many tasks have this issue, instructions may be ambiguous
       - Revise instructions and examples
       - Consider breaking task into subtasks

    **Code example:**
    ```python
    if consensus_confidence < 0.5:
        expert_id = route_to_expert(task_id, annotations)
        expert_label = await_expert_review(task_id)
        store_consensus(task_id, expert_label, is_expert=True)
        send_feedback_to_labelers(annotations, expert_label)
    ```

    ---

    ## Question 2: How do you prevent labelers from colluding to submit same labels?

    **Answer:**

    Multi-layered fraud detection approach:

    1. **Temporal analysis:**
       - Track submission timestamps
       - Flag if multiple labelers submit identical labels within 10 seconds
       - Use statistical tests (chi-square) to detect abnormal patterns

    2. **Task isolation:**
       - Don't show labeler IDs or counts to other labelers
       - Randomize task assignment to prevent same group working together
       - Use asynchronous assignment (labeler can't see who else got task)

    3. **IP and device fingerprinting:**
       - Track IP addresses and device IDs
       - Flag if multiple accounts from same IP
       - Use browser fingerprinting to detect VPN/proxy usage

    4. **Golden task validation:**
       - Insert pre-labeled validation tasks
       - If multiple labelers fail same golden task identically, investigate
       - Use honeypot tasks with obvious wrong answers

    5. **Action:**
       - Suspend accounts and require identity verification
       - Invalidate all annotations from colluding labelers
       - Ban if fraud confirmed

    ---

    ## Question 3: How would you optimize labeling cost while maintaining quality?

    **Answer:**

    Cost optimization strategies:

    1. **Active learning:**
       - Pre-label with ML model
       - Only send uncertain predictions to humans
       - Can reduce labeling costs by 50-70%

    2. **Adaptive consensus:**
       - Use 1 annotator for easy tasks (model confidence > 90%)
       - Use 3 annotators for hard tasks (model confidence < 70%)
       - Saves 2x cost on easy tasks

    3. **Tiered workforce:**
       - Crowd labelers ($3/hour) for simple tasks
       - Expert labelers ($10/hour) for complex tasks
       - Experts only for dispute resolution

    4. **Batching:**
       - Batch similar tasks together for same labeler
       - Reduces context switching and training time
       - Can increase throughput by 30%

    5. **Performance incentives:**
       - Base rate + accuracy bonuses
       - Encourages quality over quantity
       - Can reduce rework costs by 40%

    **Example cost calculation:**
    ```
    Without optimization: 10M tasks × $0.15 (3 annotators) = $1.5M

    With optimization:
    - Easy tasks (50%): 5M × $0.05 (1 annotator) = $250K
    - Medium tasks (30%): 3M × $0.10 (2 annotators) = $300K
    - Hard tasks (20%): 2M × $0.20 (3 annotators + expert) = $400K
    Total: $950K (37% savings)
    ```

    ---

    ## Question 4: How do you handle video annotation at scale?

    **Answer:**

    Video annotation has unique challenges (temporal data, large files):

    1. **Frame sampling:**
       - Don't annotate every frame (30fps = too expensive)
       - Use keyframe detection or sample every Nth frame
       - Interpolate annotations between keyframes

    2. **Temporal consistency:**
       - Track objects across frames
       - Use object IDs to maintain consistency
       - Validate no sudden appearance/disappearance

    3. **Consensus for video:**
       - Calculate IoU per frame, then average across video
       - Check temporal agreement (same object tracked consistently)
       - Use 3D bounding boxes for spatial-temporal consensus

    4. **Storage optimization:**
       - Don't store raw video in annotation DB
       - Store references to video frames in object storage
       - Use CDN for fast frame delivery

    5. **Specialized tools:**
       - Provide timeline scrubber for easy navigation
       - Support playback speed control
       - Allow copy/paste annotations across frames

    **Architecture:**
    ```
    Video Upload → Frame Extraction Service → CDN
                                              ↓
    Labeler → Load Frames from CDN → Annotate Keyframes
                                              ↓
    Interpolation Service → Fill Gaps → Consensus Engine
    ```

    ---

    ## Question 5: How do you measure and improve inter-annotator agreement?

    **Answer:**

    Measuring and improving IAA:

    **Measurement:**

    1. **Choose appropriate metric:**
       - Cohen's kappa for 2 annotators
       - Fleiss' kappa for 3+ annotators
       - Krippendorff's alpha for general cases
       - IoU for bounding boxes

    2. **Calculate regularly:**
       - Compute IAA on batches of 1000 tasks
       - Track trend over time
       - Alert if drops below threshold (0.75)

    3. **Segment by task type:**
       - Different thresholds for different complexity
       - Easy tasks: expect kappa > 0.9
       - Hard tasks: kappa > 0.6 acceptable

    **Improvement:**

    1. **Improve instructions:**
       - If low IAA, instructions likely ambiguous
       - Add more examples (positive and negative)
       - Create decision tree for edge cases

    2. **Training and calibration:**
       - Require certification test (kappa > 0.8 with gold standard)
       - Regular calibration sessions (monthly)
       - Show examples of good vs bad annotations

    3. **Simplified taxonomy:**
       - Fewer classes = higher agreement
       - Consider hierarchical labels (coarse → fine)
       - Remove ambiguous categories

    4. **Collaborative annotation:**
       - Allow labelers to discuss difficult cases
       - Create FAQ from common questions
       - Expert review sessions

    **Example improvement:**
    ```
    Initial IAA: kappa = 0.58 (moderate agreement)

    Improvements:
    1. Revised instructions with 10 examples → kappa = 0.68
    2. Added decision tree for edge cases → kappa = 0.76
    3. Recertification for low performers → kappa = 0.82

    Result: Substantial agreement achieved
    ```

