# System Design Problems

Master system design interviews with 50 real-world problems asked by top tech companies. Each problem follows the 4-step framework with architectural diagrams, capacity planning, and optimization strategies.

**Status:** ✅ 92 Complete | 🚧 0 In Progress | 📋 0 Planned

---

## Quick Start

| Your Goal | Start Here | Time Needed |
|-----------|------------|-------------|
| 🎯 **Interview next week** | Browse by Company → Pick top 10 | 7-10 days |
| 📚 **Learn system design** | Learning Path → Week 1 | 8 weeks |
| 🔍 **Practice specific concept** | Browse by Concept → Pick category | 1-2 weeks |
| ⚡ **Quick review** | Browse by Difficulty → Start Easy | 3-5 days |

---

=== "📁 By Category"

    ## Storage & Content Systems

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**URL Shortener**](url-shortener.md) | 🟡 Medium | ⭐⭐⭐⭐⭐ | Amazon, Google, Meta, Microsoft, Uber | Short code generation, caching, analytics |
    | [**Pastebin**](pastebin.md) | 🟡 Medium | ⭐⭐⭐⭐ | Amazon, Microsoft, Twitter | Text storage, expiration, syntax highlighting |
    | [**File Upload Service**](file-upload-service.md) | 🟡 Medium | ⭐⭐⭐⭐ | Dropbox, Google, Microsoft | Chunking, resumable uploads, deduplication |
    | [**Image Hosting**](image-hosting.md) | 🟡 Medium | ⭐⭐⭐⭐ | Instagram, Pinterest, Imgur | Image processing, CDN, thumbnails |
    | [**Cloud Storage (Dropbox)**](dropbox.md) | 🔴 Hard | ⭐⭐⭐⭐ | Dropbox, Google Drive, OneDrive | Sync, conflict resolution, versioning |

    ## Social & Communication

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Twitter Feed**](twitter.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Twitter, Meta, LinkedIn | Fan-out, timelines, real-time updates |
    | [**Instagram**](instagram.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Meta, Instagram, TikTok | Photo storage, feeds, followers graph |
    | [**LinkedIn**](linkedin.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | LinkedIn, Meta, Microsoft | Social graph, job matching, news feed |
    | [**WhatsApp/Chat**](whatsapp.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Meta, WhatsApp, Slack, Discord | Real-time messaging, presence, group chat |
    | [**Slack**](slack.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Slack, Microsoft Teams, Discord | Team messaging, channels, WebSocket |
    | [**Notification System**](notification-system.md) | 🟡 Medium | ⭐⭐⭐⭐ | All companies | Push notifications, delivery guarantees |
    | [**News Feed**](news-feed.md) | 🔴 Hard | ⭐⭐⭐⭐ | Facebook, LinkedIn, Reddit | Ranking, personalization, real-time |

    ## Media & Entertainment

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Video Streaming (Netflix)**](netflix.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Netflix, YouTube, Amazon | Adaptive bitrate, CDN, encoding |
    | [**Music Streaming (Spotify)**](spotify.md) | 🔴 Hard | ⭐⭐⭐⭐ | Spotify, Apple Music | Audio delivery, playlists, recommendations |
    | [**Live Streaming (Twitch)**](live-streaming.md) | 🔴 Hard | ⭐⭐⭐⭐ | Twitch, YouTube Live | Low latency, chat, transcoding |
    | [**Video Conferencing (Zoom)**](video-conferencing.md) | 🔴 Hard | ⭐⭐⭐⭐ | Zoom, Google Meet, Teams | WebRTC, signaling, mixing |

    ## Search & Discovery

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Search Engine (Google)**](search-engine.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Google, Bing | Crawling, indexing, ranking |
    | [**Autocomplete/Typeahead**](autocomplete.md) | 🟡 Medium | ⭐⭐⭐⭐⭐ | Google, Amazon, Netflix | Trie, caching, prefix matching |
    | [**Recommendation System**](recommendation-system.md) | 🔴 Hard | ⭐⭐⭐⭐ | Netflix, Amazon, YouTube | Collaborative filtering, ML models |
    | [**Web Crawler**](web-crawler.md) | 🟡 Medium | ⭐⭐⭐⭐ | Google, Bing, Archive.org | Queue, deduplication, politeness |

    ## E-Commerce & Payments

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**E-Commerce Platform (Amazon)**](ecommerce.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Amazon, eBay, Shopify | Inventory, orders, payments, catalog |
    | [**Payment System (Stripe)**](payment-system.md) | 🔴 Hard | ⭐⭐⭐⭐ | Stripe, PayPal, Square | Transactions, idempotency, ledger |
    | [**Ticket Booking (BookMyShow)**](ticket-booking.md) | 🔴 Hard | ⭐⭐⭐⭐ | BookMyShow, Ticketmaster | Concurrency, seat locking, inventory |
    | [**Airbnb**](airbnb.md) | 🔴 Hard | ⭐⭐⭐⭐ | Airbnb, Booking.com, Vrbo | Geospatial search, booking system, calendar |
    | [**Food Delivery (UberEats)**](food-delivery.md) | 🔴 Hard | ⭐⭐⭐⭐ | UberEats, DoorDash, GrubHub | Matching, routing, real-time tracking |

    ## Location-Based Services

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Ride Sharing (Uber)**](uber.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Uber, Lyft | Geohashing, matching, ETA, surge pricing |
    | [**Yelp/Nearby Places**](yelp.md) | 🟡 Medium | ⭐⭐⭐⭐ | Yelp, Google Maps | Geospatial indexing, quadtree |
    | [**Google Maps**](google-maps.md) | 🔴 Hard | ⭐⭐⭐⭐ | Google, Apple | Routing, traffic, graph algorithms |
    | [**Location Tracking**](location-tracking.md) | 🟡 Medium | ⭐⭐⭐ | Uber, DoorDash, Find My | GPS data, geofencing, privacy |

    ## Infrastructure & Developer Tools

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Key-Value Store**](key-value-store.md) | 🟢 Easy | ⭐⭐⭐ | All companies | Hash map, LRU cache, TTL, eviction |
    | [**URL Validator**](url-validator.md) | 🟢 Easy | ⭐⭐⭐ | All companies | Multi-layer validation, DNS lookup, security |
    | [**Simple Cache**](simple-cache.md) | 🟢 Easy | ⭐⭐⭐⭐ | All companies | Cache-aside, write-through, cache warming |
    | [**Health Checker**](health-checker.md) | 🟢 Easy | ⭐⭐⭐ | All companies | Health checks, alerting, SLA tracking |
    | [**Rate Limiter**](rate-limiter.md) | 🟡 Medium | ⭐⭐⭐⭐⭐ | All companies | Token bucket, sliding window, distributed |
    | [**API Gateway**](api-gateway.md) | 🟡 Medium | ⭐⭐⭐⭐ | Kong, AWS, Google | Routing, authentication, rate limiting |
    | [**Distributed Cache (Redis)**](distributed-cache.md) | 🔴 Hard | ⭐⭐⭐⭐ | Redis, Memcached | Consistent hashing, replication, eviction |
    | [**Message Queue (Kafka)**](message-queue.md) | 🔴 Hard | ⭐⭐⭐⭐ | Kafka, RabbitMQ, SQS | Partitioning, ordering, delivery guarantees |
    | [**Load Balancer**](load-balancer.md) | 🟡 Medium | ⭐⭐⭐⭐ | All companies | Algorithms, health checks, sticky sessions |

    ## Collaboration & Productivity

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Google Docs**](google-docs.md) | 🔴 Hard | ⭐⭐⭐⭐ | Google, Microsoft, Notion | CRDT, OT, real-time collaboration |
    | [**GitHub**](github.md) | 🔴 Hard | ⭐⭐⭐⭐ | GitHub, GitLab, Bitbucket | Git protocol, code search, CI/CD, webhooks |
    | [**Calendar System**](calendar-system.md) | 🟡 Medium | ⭐⭐⭐ | Google, Microsoft, Apple | Availability, conflicts, recurring events |
    | [**Task Management**](task-management.md) | 🟡 Medium | ⭐⭐⭐ | Asana, Jira, Trello | Projects, workflows, notifications |

    ## Analytics & Monitoring

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Analytics Platform**](analytics-platform.md) | 🔴 Hard | ⭐⭐⭐⭐ | Google Analytics, Mixpanel | Event tracking, aggregation, dashboards |
    | [**Metrics Monitoring**](metrics-monitoring.md) | 🟡 Medium | ⭐⭐⭐⭐ | Datadog, Prometheus, Grafana, New Relic | Time-series DB, alerting, downsampling, aggregation |
    | [**Log Aggregation**](log-aggregation.md) | 🟡 Medium | ⭐⭐⭐ | Splunk, ELK Stack | Collection, indexing, search |
    | [**Distributed Tracing**](distributed-tracing.md) | 🔴 Hard | ⭐⭐⭐ | Jaeger, Zipkin | Trace IDs, spans, correlation |

    ## Internet of Things (IoT)

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Smart Home Hub**](smart-home-hub.md) | 🔴 Hard | ⭐⭐⭐⭐ | Amazon Alexa, Google Home, Apple HomeKit | Device registry, MQTT, command routing, voice processing |
    | [**Smart Lock System**](smart-lock.md) | 🟡 Medium | ⭐⭐⭐ | Amazon Key, August, Yale | Bluetooth/WiFi, access control, audit logs, battery optimization |
    | [**Smart Thermostat**](smart-thermostat.md) | 🟡 Medium | ⭐⭐⭐ | Nest, Ecobee, Honeywell | Temperature control, learning algorithms, energy optimization |
    | [**Smart Meter (Electricity)**](smart-meter.md) | 🟡 Medium | ⭐⭐⭐⭐ | Utility companies, Sense | Real-time consumption, time-series data, billing, anomaly detection |
    | [**Smart Doorbell**](smart-doorbell.md) | 🟡 Medium | ⭐⭐⭐ | Ring, Nest Hello | Video streaming, motion detection, cloud recording, notifications |
    | [**Connected Car Platform**](connected-car.md) | 🔴 Hard | ⭐⭐⭐⭐ | Tesla, GM OnStar, BMW ConnectedDrive | OTA updates, telemetry, remote control, fleet management |
    | [**Fitness Tracker System**](fitness-tracker.md) | 🟡 Medium | ⭐⭐⭐ | Fitbit, Apple Watch, Garmin | Activity tracking, heart rate monitoring, sync, battery life |
    | [**IoT Device Management**](iot-device-management.md) | 🔴 Hard | ⭐⭐⭐⭐ | AWS IoT, Azure IoT Hub, Google Cloud IoT | Device provisioning, shadow state, OTA, fleet monitoring |
    | [**Smart City Traffic**](smart-city-traffic.md) | 🔴 Hard | ⭐⭐⭐ | City governments, Siemens | Traffic sensors, signal optimization, congestion prediction |
    | [**Industrial IoT Monitor**](industrial-iot.md) | 🔴 Hard | ⭐⭐⭐ | GE Predix, Siemens MindSphere | Predictive maintenance, sensor data, edge computing |

    ## Data Engineering

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**Data Lake**](data-lake.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | AWS S3, Delta Lake, Apache Iceberg | Object storage, partitioning, metadata, ACID transactions |
    | [**Data Warehouse**](data-warehouse.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Snowflake, BigQuery, Redshift | Columnar storage, MPP, query optimization, materialized views |
    | [**ETL Pipeline**](etl-pipeline.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Airflow, dbt, Fivetran | Orchestration, data transformation, scheduling, dependencies |
    | [**Real-time Data Pipeline**](realtime-data-pipeline.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Kafka, Flink, Spark Streaming | Stream processing, exactly-once, windowing, state management |
    | [**Change Data Capture (CDC)**](change-data-capture.md) | 🔴 Hard | ⭐⭐⭐⭐ | Debezium, AWS DMS, Airbyte | Database logs, replication, event streaming, consistency |
    | [**Data Quality Platform**](data-quality-platform.md) | 🟡 Medium | ⭐⭐⭐⭐ | Great Expectations, Monte Carlo | Data validation, anomaly detection, SLAs, lineage |
    | [**Data Catalog**](data-catalog.md) | 🟡 Medium | ⭐⭐⭐⭐ | DataHub, Amundsen, Collibra | Metadata management, search, discovery, governance |
    | [**Batch Processing System**](batch-processing.md) | 🔴 Hard | ⭐⭐⭐⭐ | Apache Spark, Hadoop | Distributed computing, partitioning, shuffle, fault tolerance |
    | [**Data Lineage Tracker**](data-lineage.md) | 🟡 Medium | ⭐⭐⭐ | OpenLineage, Marquez | Graph database, impact analysis, compliance, audit |
    | [**Data Mesh Platform**](data-mesh.md) | 🔴 Hard | ⭐⭐⭐ | Modern data teams | Domain ownership, federated governance, self-serve |

    ## Machine Learning Systems

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**ML Training Pipeline**](ml-training-pipeline.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Google Vertex AI, AWS SageMaker | Distributed training, hyperparameter tuning, checkpointing |
    | [**Feature Store**](feature-store.md) | 🔴 Hard | ⭐⭐⭐⭐ | Tecton, Feast, AWS Feature Store | Feature engineering, online/offline store, versioning, serving |
    | [**Model Serving Platform**](model-serving.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | TensorFlow Serving, Seldon, KServe | Model deployment, autoscaling, A/B testing, canary |
    | [**ML Experiment Tracking**](ml-experiment-tracking.md) | 🟡 Medium | ⭐⭐⭐⭐ | MLflow, Weights & Biases, Neptune | Metrics logging, artifact storage, comparison, reproducibility |
    | [**AutoML Platform**](automl-platform.md) | 🔴 Hard | ⭐⭐⭐⭐ | Google AutoML, H2O.ai | Neural architecture search, automated feature engineering |
    | [**A/B Testing Framework**](ab-testing-framework.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Optimizely, Google Optimize | Statistical significance, variant assignment, metrics calculation |
    | [**Model Monitoring**](model-monitoring.md) | 🟡 Medium | ⭐⭐⭐⭐ | Arize, WhyLabs, Evidently | Drift detection, performance monitoring, bias detection |
    | [**Real-time Prediction**](realtime-prediction.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Amazon Personalize, Netflix | Low-latency inference, caching, feature computation |
    | [**ML Model Registry**](ml-model-registry.md) | 🟡 Medium | ⭐⭐⭐⭐ | MLflow Registry, Neptune | Model versioning, metadata, approval workflow, deployment |
    | [**ML Labeling Platform**](ml-labeling.md) | 🟡 Medium | ⭐⭐⭐ | Labelbox, Scale AI | Data annotation, quality control, workforce management |

    ## Generative AI & LLM Systems

    | Problem | Difficulty | Frequency | Companies | Key Concepts |
    |---------|-----------|-----------|-----------|--------------|
    | [**ChatGPT-like System**](chatgpt-system.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | OpenAI, Anthropic | LLM serving, conversation state, streaming, rate limiting |
    | [**RAG System**](rag-system.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | OpenAI, Anthropic, Enterprise AI | Vector search, embeddings, retrieval, context injection |
    | [**AI Agent Platform**](ai-agent-platform.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | LangChain, AutoGPT | Tool calling, memory, planning, multi-agent orchestration |
    | [**AI Code Assistant**](ai-code-assistant.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | GitHub Copilot, Cursor, Replit | Code completion, context window, latency, caching |
    | [**Text-to-Image Generator**](text-to-image.md) | 🔴 Hard | ⭐⭐⭐⭐ | Midjourney, DALL-E, Stable Diffusion | Diffusion models, GPU queue, image storage, prompt engineering |
    | [**Vector Database**](vector-database.md) | 🔴 Hard | ⭐⭐⭐⭐⭐ | Pinecone, Weaviate, Qdrant | HNSW/IVF, similarity search, sharding, hybrid search |
    | [**Document Q&A System**](document-qa.md) | 🟡 Medium | ⭐⭐⭐⭐ | Enterprise AI | PDF parsing, chunking, embeddings, citation |
    | [**AI Voice Assistant**](ai-voice-assistant.md) | 🔴 Hard | ⭐⭐⭐⭐ | Siri, Google Assistant | Speech-to-text, NLU, TTS, wake word detection |
    | [**Prompt Management System**](prompt-management.md) | 🟡 Medium | ⭐⭐⭐⭐ | PromptLayer, Helicone | Prompt versioning, A/B testing, caching, analytics |
    | [**AI Content Moderation**](ai-content-moderation.md) | 🟡 Medium | ⭐⭐⭐⭐ | OpenAI Moderation, Perspective API | Classification, toxicity detection, human-in-loop, appeals |
    | [**LLM Fine-tuning Platform**](llm-finetuning.md) | 🔴 Hard | ⭐⭐⭐⭐ | OpenAI, Anthropic, Hugging Face | LoRA/QLoRA, dataset management, evaluation, deployment |
    | [**Multi-modal AI System**](multimodal-ai.md) | 🔴 Hard | ⭐⭐⭐⭐ | GPT-4V, Gemini | Vision+Language, audio processing, unified embeddings |

=== "🎚️ By Difficulty"

    ## 🟢 Easy Problems (5 problems)

    **Perfect for:** Beginners, first interview prep, understanding basics

    **Time per problem:** 30-40 minutes

    | Problem | Category | Frequency | Key Learning |
    |---------|----------|-----------|--------------|
    | [**Key-Value Store**](key-value-store.md) | Infrastructure | ⭐⭐⭐ | Hash map, LRU eviction, TTL |
    | [**URL Validator**](url-validator.md) | Tools | ⭐⭐⭐ | Validation pipeline, DNS, security |
    | [**Simple Cache**](simple-cache.md) | Infrastructure | ⭐⭐⭐⭐ | Cache-aside, eviction, invalidation |
    | [**Unique ID Generator**](unique-id-generator.md) | Infrastructure | ⭐⭐⭐⭐ | Distributed ID generation |
    | [**Health Checker**](health-checker.md) | Monitoring | ⭐⭐⭐ | Health checks, alerting, SLA |

    ---

    ## 🟡 Medium Problems (38 problems)

    **Perfect for:** Intermediate prep, common interview questions, building fundamentals

    **Time per problem:** 45-60 minutes

    | Problem | Category | Frequency | Key Learning |
    |---------|----------|-----------|--------------|
    | [**URL Shortener**](url-shortener.md) | Storage | ⭐⭐⭐⭐⭐ | Code generation, caching, analytics |
    | [**Pastebin**](pastebin.md) | Storage | ⭐⭐⭐⭐ | Text storage, expiration handling |
    | [**Rate Limiter**](rate-limiter.md) | Infrastructure | ⭐⭐⭐⭐⭐ | Token bucket, sliding window |
    | [**Autocomplete**](autocomplete.md) | Search | ⭐⭐⭐⭐⭐ | Trie, prefix matching, caching |
    | [**File Upload Service**](file-upload-service.md) | Storage | ⭐⭐⭐⭐ | Chunking, resumable uploads, deduplication |
    | [**Image Hosting**](image-hosting.md) | Storage | ⭐⭐⭐⭐ | CDN, image processing, thumbnails |
    | [**Notification System**](notification-system.md) | Social | ⭐⭐⭐⭐ | Fan-out, delivery guarantees |
    | [**Web Crawler**](web-crawler.md) | Search | ⭐⭐⭐⭐ | Queue, deduplication |
    | [**Yelp/Nearby**](yelp.md) | Location | ⭐⭐⭐⭐ | Geospatial indexing |
    | [**Calendar System**](calendar-system.md) | Collaboration | ⭐⭐⭐ | Recurring events, conflict detection |
    | [**Task Management**](task-management.md) | Collaboration | ⭐⭐⭐ | Workflows, dependencies, notifications |
    | [**API Gateway**](api-gateway.md) | Infrastructure | ⭐⭐⭐⭐ | Routing, auth, rate limiting |
    | [**Load Balancer**](load-balancer.md) | Infrastructure | ⭐⭐⭐⭐ | Algorithms, health checks |
    | [**Metrics Monitoring**](metrics-monitoring.md) | Monitoring | ⭐⭐⭐⭐ | Time-series DB, downsampling, alerting |
    | [**Log Aggregation**](log-aggregation.md) | Monitoring | ⭐⭐⭐ | Collection, indexing, search |
    | [**Location Tracking**](location-tracking.md) | Location | ⭐⭐⭐ | GPS, geofencing, privacy |
    | [**Smart Lock System**](smart-lock.md) | IoT | ⭐⭐⭐ | Bluetooth/WiFi, access control, battery optimization |
    | [**Smart Meter**](smart-meter.md) | IoT | ⭐⭐⭐⭐ | Time-series data, billing, anomaly detection |
    | [**Smart Thermostat**](smart-thermostat.md) | IoT | ⭐⭐⭐ | Temperature control, learning algorithms, energy optimization |
    | [**Smart Doorbell**](smart-doorbell.md) | IoT | ⭐⭐⭐ | Video streaming, motion detection, cloud recording |
    | [**Fitness Tracker System**](fitness-tracker.md) | IoT | ⭐⭐⭐ | Activity tracking, HR monitoring, sync, battery life |
    | [**Data Quality Platform**](data-quality-platform.md) | Data Eng | ⭐⭐⭐⭐ | Validation, anomaly detection, profiling |
    | [**Data Catalog**](data-catalog.md) | Data Eng | ⭐⭐⭐⭐ | Metadata management, search, lineage, tagging |
    | [**ML Experiment Tracking**](ml-experiment-tracking.md) | ML | ⭐⭐⭐⭐ | Metrics logging, artifact storage, comparison |
    | [**Model Monitoring**](model-monitoring.md) | ML | ⭐⭐⭐⭐ | Drift detection, performance monitoring, bias |
    | [**Document Q&A System**](document-qa.md) | GenAI | ⭐⭐⭐⭐ | PDF parsing, chunking, embeddings, RAG |
    | [**Prompt Management System**](prompt-management.md) | GenAI | ⭐⭐⭐⭐ | Versioning, A/B testing, caching, analytics |
    | [**AI Content Moderation**](ai-content-moderation.md) | GenAI | ⭐⭐⭐⭐ | Multi-modal classification, toxicity, NSFW, human review |
    | [**Data Lineage Tracker**](data-lineage.md) | Data Eng | ⭐⭐⭐ | Column-level lineage, impact analysis, compliance |
    | [**ML Model Registry**](ml-model-registry.md) | ML | ⭐⭐⭐⭐ | Versioning, approval workflows, staging, deployment tracking |
    | [**ML Labeling Platform**](ml-labeling.md) | ML | ⭐⭐⭐ | Consensus, quality control, active learning, IAA |

    ---

    ## 🔴 Hard Problems (49 problems)

    **Perfect for:** Advanced prep, FAANG interviews, senior roles

    **Time per problem:** 60-75 minutes

    | Problem | Category | Frequency | Key Learning |
    |---------|----------|-----------|--------------|
    | [**Twitter Feed**](twitter.md) | Social | ⭐⭐⭐⭐⭐ | Fan-out, timeline generation |
    | [**Instagram**](instagram.md) | Social | ⭐⭐⭐⭐⭐ | Photo storage, social graph |
    | [**LinkedIn**](linkedin.md) | Social | ⭐⭐⭐⭐⭐ | Social graph, job matching, Neo4j |
    | [**WhatsApp/Chat**](whatsapp.md) | Social | ⭐⭐⭐⭐⭐ | Real-time messaging, presence |
    | [**Slack**](slack.md) | Social | ⭐⭐⭐⭐⭐ | WebSocket, channels, message ordering |
    | [**Video Streaming**](netflix.md) | Media | ⭐⭐⭐⭐⭐ | CDN, encoding, adaptive bitrate |
    | [**Search Engine**](search-engine.md) | Search | ⭐⭐⭐⭐⭐ | Crawling, indexing, ranking |
    | [**E-Commerce Platform**](ecommerce.md) | E-Commerce | ⭐⭐⭐⭐⭐ | Inventory, transactions, catalog |
    | [**Ride Sharing**](uber.md) | Location | ⭐⭐⭐⭐⭐ | Geohashing, matching, routing |
    | [**News Feed**](news-feed.md) | Social | ⭐⭐⭐⭐ | Ranking, personalization |
    | [**Music Streaming**](spotify.md) | Media | ⭐⭐⭐⭐ | Audio delivery, recommendations |
    | [**Live Streaming**](live-streaming.md) | Media | ⭐⭐⭐⭐ | Low latency, transcoding |
    | [**Video Conferencing**](video-conferencing.md) | Media | ⭐⭐⭐⭐ | WebRTC, signaling |
    | [**Recommendation System**](recommendation-system.md) | Search | ⭐⭐⭐⭐ | Collaborative filtering, ML |
    | [**Cloud Storage**](dropbox.md) | Storage | ⭐⭐⭐⭐ | Sync, conflict resolution |
    | [**Payment System**](payment-system.md) | E-Commerce | ⭐⭐⭐⭐ | Transactions, idempotency |
    | [**Ticket Booking**](ticket-booking.md) | E-Commerce | ⭐⭐⭐⭐ | Concurrency, locking |
    | [**Airbnb**](airbnb.md) | E-Commerce | ⭐⭐⭐⭐ | Geospatial search, booking consistency |
    | [**Food Delivery**](food-delivery.md) | E-Commerce | ⭐⭐⭐⭐ | Matching, real-time tracking |
    | [**Google Maps**](google-maps.md) | Location | ⭐⭐⭐⭐ | Routing, traffic algorithms |
    | [**Google Docs**](google-docs.md) | Collaboration | ⭐⭐⭐⭐ | CRDT, real-time collaboration |
    | [**GitHub**](github.md) | Collaboration | ⭐⭐⭐⭐ | Git storage, code search, CI/CD |
    | [**Distributed Cache**](distributed-cache.md) | Infrastructure | ⭐⭐⭐⭐ | Consistent hashing, replication |
    | [**Message Queue**](message-queue.md) | Infrastructure | ⭐⭐⭐⭐ | Partitioning, ordering |
    | [**Analytics Platform**](analytics-platform.md) | Monitoring | ⭐⭐⭐⭐ | Event tracking, aggregation |
    | [**Distributed Tracing**](distributed-tracing.md) | Monitoring | ⭐⭐⭐ | Trace IDs, spans, correlation |
    | [**ChatGPT-like System**](chatgpt-system.md) | GenAI | ⭐⭐⭐⭐⭐ | LLM serving, streaming, conversation state |
    | [**RAG System**](rag-system.md) | GenAI | ⭐⭐⭐⭐⭐ | Vector search, retrieval, context injection |
    | [**Vector Database**](vector-database.md) | GenAI | ⭐⭐⭐⭐⭐ | HNSW, similarity search, sharding |
    | [**Data Lake**](data-lake.md) | Data Eng | ⭐⭐⭐⭐⭐ | ACID transactions, partitioning, schema evolution |
    | [**Data Warehouse**](data-warehouse.md) | Data Eng | ⭐⭐⭐⭐⭐ | MPP, columnar storage, query optimization |
    | [**ETL Pipeline**](etl-pipeline.md) | Data Eng | ⭐⭐⭐⭐⭐ | DAG orchestration, incremental loading, lineage |
    | [**Feature Store**](feature-store.md) | ML | ⭐⭐⭐⭐ | Online/offline serving, point-in-time joins |
    | [**Model Serving Platform**](model-serving.md) | ML | ⭐⭐⭐⭐⭐ | Dynamic batching, A/B testing, canary deployment |
    | [**AI Code Assistant**](ai-code-assistant.md) | GenAI | ⭐⭐⭐⭐⭐ | Code completion, context extraction, caching |
    | [**Real-time Data Pipeline**](realtime-data-pipeline.md) | Data Eng | ⭐⭐⭐⭐⭐ | Stream processing, exactly-once, windowing |
    | [**ML Training Pipeline**](ml-training-pipeline.md) | ML | ⭐⭐⭐⭐⭐ | Distributed training, hyperparameter tuning, checkpointing |
    | [**A/B Testing Framework**](ab-testing-framework.md) | ML | ⭐⭐⭐⭐⭐ | Statistical testing, variant assignment, multi-armed bandits |
    | [**AI Agent Platform**](ai-agent-platform.md) | GenAI | ⭐⭐⭐⭐⭐ | ReAct prompting, tool calling, multi-agent orchestration |
    | [**Smart Home Hub**](smart-home-hub.md) | IoT | ⭐⭐⭐⭐ | Device registry, MQTT, command routing, voice processing |
    | [**Change Data Capture**](change-data-capture.md) | Data Eng | ⭐⭐⭐⭐ | Database logs, replication, event streaming |
    | [**Text-to-Image Generator**](text-to-image.md) | GenAI | ⭐⭐⭐⭐ | Diffusion models, GPU queue, image storage |
    | [**Connected Car Platform**](connected-car.md) | IoT | ⭐⭐⭐⭐ | OTA updates, telemetry, remote control, fleet management |
    | [**AutoML Platform**](automl-platform.md) | ML | ⭐⭐⭐⭐ | Neural architecture search, hyperparameter optimization |
    | [**AI Voice Assistant**](ai-voice-assistant.md) | GenAI | ⭐⭐⭐⭐ | Wake word, STT, NLU, TTS, multi-turn dialogue |
    | [**IoT Device Management**](iot-device-management.md) | IoT | ⭐⭐⭐⭐ | Device provisioning, shadow state, OTA updates |
    | [**Batch Processing System**](batch-processing.md) | Data Eng | ⭐⭐⭐⭐ | Spark/Hadoop, distributed computing, shuffle |
    | [**Real-time Prediction**](realtime-prediction.md) | ML | ⭐⭐⭐⭐⭐ | Low-latency inference, online features, caching |
    | [**LLM Fine-tuning Platform**](llm-finetuning.md) | GenAI | ⭐⭐⭐⭐ | LoRA/QLoRA, FSDP, instruction tuning, RLHF |
    | [**Smart City Traffic**](smart-city-traffic.md) | IoT | ⭐⭐⭐ | Signal optimization, congestion prediction, RL |
    | [**Industrial IoT Monitor**](industrial-iot.md) | IoT | ⭐⭐⭐ | Predictive maintenance, OPC UA, edge computing, RUL |
    | [**Data Mesh Platform**](data-mesh.md) | Data Eng | ⭐⭐⭐ | Domain ownership, data products, federated governance |
    | [**Multi-modal AI System**](multimodal-ai.md) | GenAI | ⭐⭐⭐⭐ | Vision-language, unified embeddings, cross-modal attention |

=== "🏢 By Company"

    ## FAANG Companies

    ### Meta (Facebook)

    **Focus:** Social graphs, real-time systems, massive scale

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | [**Instagram**](instagram.md) | 🔴 Hard | Core product, photo storage, feeds |
    | [**WhatsApp**](whatsapp.md) | 🔴 Hard | Messaging at scale, real-time |
    | [**News Feed**](news-feed.md) | 🔴 Hard | Timeline generation, ranking |
    | [**Twitter Feed**](twitter.md) | 🔴 Hard | Fan-out, social graph |
    | [**Notification System**](notification-system.md) | 🟡 Medium | Cross-platform notifications |
    | [**Live Streaming**](live-streaming.md) | 🔴 Hard | Facebook Live, Instagram Live |
    | [**Chat System**](whatsapp.md) | 🔴 Hard | Messenger architecture |

    ### Amazon

    **Focus:** E-commerce, consistency, API design, transactions

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | [**E-Commerce Platform**](ecommerce.md) | 🔴 Hard | Core business, inventory management |
    | [**Payment System**](payment-system.md) | 🔴 Hard | Transactions, consistency |
    | [**URL Shortener**](url-shortener.md) | 🟡 Medium | API design fundamentals |
    | [**Rate Limiter**](rate-limiter.md) | 🟡 Medium | API protection, throttling |
    | [**Recommendation System**](recommendation-system.md) | 🔴 Hard | Product recommendations |
    | [**Distributed Cache**](distributed-cache.md) | 🔴 Hard | Performance optimization |
    | [**Search Engine**](search-engine.md) | 🔴 Hard | Product search |

    ### Apple

    **Focus:** Mobile-first, sync, privacy, user experience

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | [**iMessage**](whatsapp.md) | 🔴 Hard | Real-time messaging, encryption |
    | [**iCloud Storage**](dropbox.md) | 🔴 Hard | Sync across devices |
    | **Calendar System** | 🟡 Medium | Sync, conflict resolution |
    | [**Music Streaming**](spotify.md) | 🔴 Hard | Apple Music architecture |
    | **Location Tracking** | 🟡 Medium | Find My, privacy |
    | [**Notification System**](notification-system.md) | 🟡 Medium | APNs architecture |

    ### Netflix

    **Focus:** Video streaming, CDN, recommendations, global scale

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | [**Video Streaming**](netflix.md) | 🔴 Hard | Core product, CDN strategy |
    | [**Recommendation System**](recommendation-system.md) | 🔴 Hard | Content personalization |
    | [**Analytics Platform**](analytics-platform.md) | 🔴 Hard | User behavior tracking |
    | [**Distributed Cache**](distributed-cache.md) | 🔴 Hard | Content caching |
    | [**API Gateway**](api-gateway.md) | 🟡 Medium | Microservices gateway |
    | [**Rate Limiter**](rate-limiter.md) | 🟡 Medium | API protection |

    ### Google

    **Focus:** Search, scale, distributed systems, ML

    | Problem | Difficulty | Why They Ask |
    |---------|-----------|--------------|
    | [**Search Engine**](search-engine.md) | 🔴 Hard | Core product, indexing, ranking |
    | [**Google Maps**](google-maps.md) | 🔴 Hard | Routing, traffic, geospatial |
    | [**YouTube**](youtube.md) | 🔴 Hard | Video streaming, recommendations |
    | [**Google Docs**](google-docs.md) | 🔴 Hard | Real-time collaboration |
    | [**Google Drive**](dropbox.md) | 🔴 Hard | Cloud storage, sync |
    | [**Autocomplete**](autocomplete.md) | 🟡 Medium | Search suggestions |
    | [**Web Crawler**](web-crawler.md) | 🟡 Medium | Indexing the web |
    | [**Distributed Cache**](distributed-cache.md) | 🔴 Hard | Memcached, performance |

    ---

    ## Other Tech Giants

    ### Microsoft

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | [**Teams/Slack**](slack.md) | 🔴 Hard | Chat, team collaboration |
    | [**OneDrive**](dropbox.md) | 🔴 Hard | Cloud storage, sync |
    | [**GitHub**](github.md) | 🔴 Hard | Code hosting, CI/CD |
    | **Calendar System** | 🟡 Medium | Outlook calendar |
    | [**Video Conferencing**](video-conferencing.md) | 🔴 Hard | Teams meetings |
    | [**LinkedIn**](linkedin.md) | 🔴 Hard | Professional network |

    ### Uber

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | [**Ride Sharing**](uber.md) | 🔴 Hard | Core product, matching |
    | [**Food Delivery**](food-delivery.md) | 🔴 Hard | UberEats, routing |
    | [**Google Maps**](google-maps.md) | 🔴 Hard | Navigation, ETA |
    | **Location Tracking** | 🟡 Medium | Real-time tracking |
    | [**Notification System**](notification-system.md) | 🟡 Medium | Driver/rider notifications |
    | [**Payment System**](payment-system.md) | 🔴 Hard | Payment processing |

    ### Airbnb

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | [**Airbnb**](airbnb.md) | 🔴 Hard | Core product, booking system |
    | [**Ticket Booking**](ticket-booking.md) | 🔴 Hard | Reservation system |
    | [**Search Engine**](search-engine.md) | 🔴 Hard | Property search |
    | [**Payment System**](payment-system.md) | 🔴 Hard | Booking payments |
    | **Calendar System** | 🟡 Medium | Availability calendar |
    | [**Recommendation System**](recommendation-system.md) | 🔴 Hard | Property recommendations |

    ### LinkedIn

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | [**LinkedIn**](linkedin.md) | 🔴 Hard | Core product, social graph, job matching |
    | [**News Feed**](news-feed.md) | 🔴 Hard | Professional feed |
    | [**Twitter Feed**](twitter.md) | 🔴 Hard | Timeline generation |
    | [**Notification System**](notification-system.md) | 🟡 Medium | Job alerts, messages |
    | [**Search Engine**](search-engine.md) | 🔴 Hard | Job/people search |
    | [**Recommendation System**](recommendation-system.md) | 🔴 Hard | Job/connection recommendations |

    ### Twitter

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | [**Twitter Feed**](twitter.md) | 🔴 Hard | Core product, timeline |
    | [**Notification System**](notification-system.md) | 🟡 Medium | Real-time notifications |
    | [**Search Engine**](search-engine.md) | 🔴 Hard | Tweet search |
    | [**URL Shortener**](url-shortener.md) | 🟡 Medium | t.co shortener |
    | [**Live Streaming**](live-streaming.md) | 🔴 Hard | Twitter Spaces |

    ### Slack

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | [**Slack**](slack.md) | 🔴 Hard | Core product, team messaging |
    | [**WhatsApp**](whatsapp.md) | 🔴 Hard | Real-time messaging |
    | [**Notification System**](notification-system.md) | 🟡 Medium | Channel notifications |
    | [**Search Engine**](search-engine.md) | 🔴 Hard | Message search |
    | [**File Upload Service**](whatsapp.md) | 🟡 Medium | File sharing |

    ### GitHub

    | Problem | Difficulty | Focus Area |
    |---------|-----------|------------|
    | [**GitHub**](github.md) | 🔴 Hard | Core product, code hosting |
    | [**Distributed Cache**](distributed-cache.md) | 🔴 Hard | Git object caching |
    | [**Search Engine**](search-engine.md) | 🔴 Hard | Code search |
    | [**Notification System**](notification-system.md) | 🟡 Medium | Pull request notifications |
    | [**Message Queue**](message-queue.md) | 🔴 Hard | Webhook delivery |

=== "🧩 By Concept"

    **Learn specific system design concepts through relevant problems**

    ## Caching (Most Important!)

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Simple Cache**](simple-cache.md) | Cache-aside, write-through, TTL, eviction | 🟢 Easy |
    | [**URL Shortener**](url-shortener.md) | Multi-layer caching, cache invalidation | 🟡 Medium |
    | [**Rate Limiter**](rate-limiter.md) | Distributed cache, sliding window | 🟡 Medium |
    | [**Autocomplete**](autocomplete.md) | Cache warming, prefix caching | 🟡 Medium |
    | [**News Feed**](news-feed.md) | Cache strategy for timelines | 🔴 Hard |
    | [**Distributed Cache**](distributed-cache.md) | Consistent hashing, replication | 🔴 Hard |
    | [**Video Streaming**](netflix.md) | CDN caching, edge caching | 🔴 Hard |

    ## Database Sharding & Partitioning

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Twitter Feed**](twitter.md) | Shard by user_id, fan-out strategy | 🔴 Hard |
    | [**Instagram**](instagram.md) | Photo metadata sharding | 🔴 Hard |
    | [**E-Commerce**](ecommerce.md) | Product catalog sharding | 🔴 Hard |
    | [**URL Shortener**](url-shortener.md) | Shard by short_code prefix | 🟡 Medium |
    | [**WhatsApp**](whatsapp.md) | Message sharding, chat_id routing | 🔴 Hard |

    ## Consistent Hashing

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Distributed Cache**](distributed-cache.md) | Hash ring, virtual nodes | 🔴 Hard |
    | [**Load Balancer**](load-balancer.md) | Server selection, rebalancing | 🟡 Medium |
    | [**URL Shortener**](url-shortener.md) | Shard routing | 🟡 Medium |
    | [**Message Queue**](message-queue.md) | Partition assignment | 🔴 Hard |

    ## Fan-out Pattern

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Twitter Feed**](twitter.md) | Push vs pull, hybrid fan-out | 🔴 Hard |
    | [**Instagram**](instagram.md) | Photo upload fan-out | 🔴 Hard |
    | [**Notification System**](notification-system.md) | Multi-channel fan-out | 🟡 Medium |
    | [**News Feed**](news-feed.md) | Timeline generation | 🔴 Hard |

    ## Real-time Systems

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**WhatsApp**](whatsapp.md) | WebSocket, message delivery | 🔴 Hard |
    | [**Live Streaming**](live-streaming.md) | Low latency, buffering | 🔴 Hard |
    | [**Video Conferencing**](video-conferencing.md) | WebRTC, peer connections | 🔴 Hard |
    | [**Google Docs**](google-docs.md) | OT/CRDT, conflict resolution | 🔴 Hard |
    | **Location Tracking** | GPS updates, real-time | 🟡 Medium |

    ## Geospatial Systems

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Ride Sharing**](uber.md) | Geohashing, driver matching | 🔴 Hard |
    | [**Yelp/Nearby**](yelp.md) | Quadtree, geospatial queries | 🟡 Medium |
    | [**Google Maps**](google-maps.md) | Graph algorithms, routing | 🔴 Hard |
    | [**Food Delivery**](food-delivery.md) | Route optimization | 🔴 Hard |

    ## CDN & Content Delivery

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Video Streaming**](netflix.md) | CDN strategy, edge servers | 🔴 Hard |
    | **Image Hosting** | Image CDN, transformations | 🟡 Medium |
    | [**Music Streaming**](spotify.md) | Audio delivery | 🔴 Hard |
    | [**Cloud Storage**](dropbox.md) | File distribution | 🔴 Hard |

    ## Message Queues & Async Processing

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Message Queue**](message-queue.md) | Kafka, partitioning, ordering | 🔴 Hard |
    | [**Notification System**](notification-system.md) | Async delivery, retry logic | 🟡 Medium |
    | [**Analytics Platform**](analytics-platform.md) | Event streaming, processing | 🔴 Hard |
    | [**Web Crawler**](web-crawler.md) | Queue management, priority | 🟡 Medium |

    ## Consistency & Transactions

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Payment System**](payment-system.md) | ACID, idempotency, ledger | 🔴 Hard |
    | [**Ticket Booking**](ticket-booking.md) | Optimistic/pessimistic locking | 🔴 Hard |
    | [**E-Commerce**](ecommerce.md) | Inventory consistency | 🔴 Hard |
    | [**Cloud Storage**](dropbox.md) | Sync, conflict resolution | 🔴 Hard |

    ## Search & Ranking

    | Problem | What You'll Learn | Difficulty |
    |---------|-------------------|------------|
    | [**Search Engine**](search-engine.md) | Inverted index, TF-IDF, PageRank | 🔴 Hard |
    | [**Autocomplete**](autocomplete.md) | Trie, prefix search | 🟡 Medium |
    | [**Recommendation System**](recommendation-system.md) | Collaborative filtering, ML | 🔴 Hard |
    | [**E-Commerce**](ecommerce.md) | Product ranking | 🔴 Hard |

=== "📅 Learning Path"

    **8-week structured program from beginner to advanced**

    ## Week 1-2: Foundation

    **Goal:** Build fundamentals with easy/medium problems

    **Time commitment:** 2-3 problems per week, 2-3 hours per problem

    | Problem | Day | Key Learning | Time |
    |---------|-----|--------------|------|
    | [**URL Shortener**](url-shortener.md) | 1-2 | Code generation, caching basics, capacity estimation | 3h |
    | [**Rate Limiter**](rate-limiter.md) | 3-4 | Token bucket, sliding window, distributed systems | 2h |
    | [**Pastebin**](pastebin.md) | 5-6 | Text storage, expiration, similar to URL shortener | 2h |
    | [**Key-Value Store**](key-value-store.md) | 7 | Hash map, LRU cache, basic CRUD operations | 1h |

    **✅ Checkpoint:** Can you explain caching strategies and do capacity estimation?

    ---

    ## Week 3-4: Scale & Distribution

    **Goal:** Learn distribution, partitioning, and scaling patterns

    | Problem | Day | Key Learning | Time |
    |---------|-----|--------------|------|
    | [**Autocomplete**](autocomplete.md) | 8-9 | Trie data structure, prefix caching | 2h |
    | [**Notification System**](notification-system.md) | 10-11 | Fan-out pattern, multi-channel delivery | 3h |
    | [**Web Crawler**](web-crawler.md) | 12-13 | Queue management, distributed coordination | 3h |
    | [**Yelp/Nearby**](yelp.md) | 14 | Geospatial indexing, quadtree | 2h |

    **✅ Checkpoint:** Can you explain fan-out patterns and sharding strategies?

    ---

    ## Week 5-6: Complex Systems (Hard Problems)

    **Goal:** Tackle FAANG-level problems with multiple components

    | Problem | Day | Key Learning | Time |
    |---------|-----|--------------|------|
    | [**Twitter Feed**](twitter.md) | 15-17 | Timeline generation, hybrid fan-out, massive scale | 4h |
    | [**Video Streaming**](netflix.md) | 18-20 | CDN, encoding, adaptive bitrate, global scale | 4h |
    | [**Ride Sharing**](uber.md) | 21-23 | Geohashing, matching algorithms, real-time | 4h |
    | [**Metrics Monitoring**](metrics-monitoring.md) | 24-25 | Time-series DB, downsampling, alerting | 2h |

    **✅ Checkpoint:** Can you design systems with 100M+ users?

    ---

    ## Week 7-8: Specialization & Practice

    **Goal:** Deep dive into your target company's domain

    ### Choose Your Track:

    === "Social Media Track"

        | Problem | Focus |
        |---------|-------|
        | [**Instagram**](instagram.md) | Photo storage, social graph |
        | [**WhatsApp**](whatsapp.md) | Real-time messaging |
        | [**News Feed**](news-feed.md) | Ranking, personalization |
        | [**Live Streaming**](live-streaming.md) | Low latency video |

    === "E-Commerce Track"

        | Problem | Focus |
        |---------|-------|
        | [**E-Commerce Platform**](ecommerce.md) | Inventory, transactions |
        | [**Payment System**](payment-system.md) | Consistency, ledger |
        | [**Ticket Booking**](ticket-booking.md) | Concurrency, locking |
        | [**Search Engine**](search-engine.md) | Product search |

    === "Media Track"

        | Problem | Focus |
        |---------|-------|
        | [**Video Streaming**](netflix.md) | CDN, encoding |
        | [**Music Streaming**](spotify.md) | Recommendations |
        | [**Video Conferencing**](video-conferencing.md) | WebRTC |
        | [**Analytics Platform**](analytics-platform.md) | Event tracking |

    === "Infrastructure Track"

        | Problem | Focus |
        |---------|-------|
        | [**Distributed Cache**](distributed-cache.md) | Consistent hashing |
        | [**Message Queue**](message-queue.md) | Partitioning, ordering |
        | [**API Gateway**](api-gateway.md) | Routing, auth |
        | [**Load Balancer**](load-balancer.md) | Algorithms |

    **✅ Final Checkpoint:** Ready for interviews! Practice explaining designs out loud.

    ---

    ## Interview Week Prep

    **Goal:** Quick review and polish

    | Day | Activity | Time |
    |-----|----------|------|
    | **-7 days** | Review top 10 most frequent problems | 2h |
    | **-6 days** | Mock interview #1 (with friend/peer) | 1h |
    | **-5 days** | Practice calculations and estimations | 1h |
    | **-4 days** | Review trade-offs and bottlenecks | 2h |
    | **-3 days** | Mock interview #2 | 1h |
    | **-2 days** | Review company-specific problems | 2h |
    | **-1 day** | Light review, rest well | 1h |

---

## How to Practice Each Problem

**Follow this systematic approach:**

### 1. First Attempt (30-45 min)
- Read problem statement only
- Try to design it yourself
- Don't look at the solution
- Draw diagrams on paper/whiteboard
- Calculate capacity estimates

### 2. Study Solution (30-45 min)
- Read through complete solution
- Understand each component's purpose
- Note what you missed
- Study the diagrams

### 3. Identify Gaps (15 min)
- What did you miss?
- Which concepts were new?
- What would you do differently?

### 4. Explain Out Loud (20 min)
- Pretend you're in an interview
- Explain the design from scratch
- Focus on trade-offs

### 5. Review (3 days later)
- Can you still explain it?
- Revisit weak areas
- Practice with variations

---

## Interview Preparation Checklist

### Before Your Interview

**Technical Readiness:**
- [ ] Completed 10+ problems (5 easy/medium, 5 hard)
- [ ] Can do capacity calculations in < 5 minutes
- [ ] Understand caching strategies deeply
- [ ] Know database sharding approaches
- [ ] Can explain CAP theorem with examples
- [ ] Comfortable with trade-off discussions

**Communication Readiness:**
- [ ] Practiced 2+ mock interviews
- [ ] Can think out loud naturally
- [ ] Ask clarifying questions first
- [ ] Draw clear diagrams quickly
- [ ] Handle "what if" questions confidently

**Company-Specific:**
- [ ] Completed 5+ problems from target company list
- [ ] Understand their tech stack
- [ ] Know their scale (users, requests, data)
- [ ] Researched their engineering blog

---

## Study Resources

| Resource | Use For |
|----------|---------|
| [Interview Framework](../interviews/framework.md) | 4-step process, structure |
| [Calculations Guide](../interviews/calculations.md) | Quick capacity estimation |
| [Communication Tips](../interviews/communication.md) | Interview techniques |
| [Common Mistakes](../interviews/common-mistakes.md) | What to avoid |

---

## Quick Reference

### Each Problem Contains:

| Section | Time to Spend | What's Included |
|---------|---------------|-----------------|
| **Step 1: Requirements** | 10-15 min | Functional/non-functional requirements, capacity estimation with calculations |
| **Step 2: High-Level Design** | 15-20 min | Architecture diagrams, API design, database schema, data flow |
| **Step 3: Deep Dive** | 15-20 min | Algorithms with code, caching strategies, optimization techniques |
| **Step 4: Scale & Optimize** | 5-10 min | Bottlenecks, trade-offs, monitoring, reliability patterns |

### Difficulty Legend:
- 🟢 **Easy:** Single-server, simple CRUD, basic caching (5 problems)
- 🟡 **Medium:** Distribution, sharding, moderate scale (25 problems)
- 🔴 **Hard:** Massive scale, complex trade-offs, real-time (20 problems)

### Frequency Legend:
- ⭐⭐⭐⭐⭐ Very High (asked by 5+ companies regularly)
- ⭐⭐⭐⭐ High (asked by 3-4 companies)
- ⭐⭐⭐ Medium (asked by 2 companies)

---

## Getting Started

**Choose your starting point:**

- 🎯 **Interview this week** → "By Company" tab → Pick your company's top 10
- 📚 **Learning from scratch** → "Learning Path" tab → Start Week 1
- 🔍 **Master specific concept** → "By Concept" tab → Pick your topic
- ⚡ **Quick review** → "By Difficulty" tab → Practice easy → medium → hard

**Most popular starting problem:** [URL Shortener](url-shortener.md) ⭐⭐⭐⭐⭐

---

**Ready to begin? Pick your approach above and start practicing!** 🚀
