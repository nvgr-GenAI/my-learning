# Design Log Aggregation System

A distributed, scalable log aggregation platform that collects, parses, indexes, and searches logs from thousands of sources in real-time, providing full-text search, filtering, alerting, and visualization capabilities for operational insights and debugging.

**Difficulty:** 🟡 Medium | **Frequency:** ⭐⭐⭐ Medium | **Time:** 45-60 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 10 TB/day logs, 100K logs/sec, <1s ingestion latency, <200ms search latency, 99.9% availability |
| **Key Challenges** | Log parsing (Grok patterns), full-text indexing, hot-warm-cold storage, real-time search, cardinality management, compression |
| **Core Concepts** | Log shippers (Filebeat), message queue buffering, inverted index, Lucene segments, log retention, structured logging |
| **Companies** | Splunk, Elasticsearch (ELK Stack), Datadog, Sumo Logic, Loggly, Papertrail, Grafana Loki |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Log Collection** | Collect logs from servers, containers, apps (agents/sidecar) | P0 (Must have) |
    | **Log Parsing** | Parse unstructured logs into structured fields (Grok patterns) | P0 (Must have) |
    | **Full-Text Search** | Search logs with keywords, regex, boolean queries | P0 (Must have) |
    | **Filtering** | Filter by timestamp, log level, service, host, fields | P0 (Must have) |
    | **Real-Time Streaming** | View live log tail (WebSocket streaming) | P0 (Must have) |
    | **Indexing** | Fast indexing with inverted index (Elasticsearch-style) | P0 (Must have) |
    | **Aggregation** | Count, group by field, top values, histogram | P1 (Should have) |
    | **Alerting** | Alert on log patterns (error rate spikes) | P1 (Should have) |
    | **Dashboard** | Visualize log trends, error rates, latency | P1 (Should have) |
    | **Log Retention** | Archive old logs (hot-warm-cold) | P1 (Should have) |

    **Explicitly Out of Scope** (mention in interview):

    - Distributed tracing (use Jaeger/Zipkin)
    - Metrics collection (use Prometheus/Datadog)
    - Log anonymization/redaction (PII masking)
    - Machine learning anomaly detection
    - SIEM capabilities (security event correlation)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Ingestion Rate** | 100K logs/sec | Support large-scale distributed systems |
    | **Ingestion Latency** | < 1 second | Near real-time log availability |
    | **Search Latency** | < 200ms p95 | Fast debugging and troubleshooting |
    | **Availability** | 99.9% uptime | Critical for production debugging |
    | **Retention** | 90 days (hot: 7 days, warm: 30 days, cold: 90 days) | Balance cost and compliance |
    | **Compression** | 10:1 ratio | Reduce storage costs (logs are text) |
    | **Data Durability** | 99.99% | No log loss (message queue buffering) |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Infrastructure scale:
    - Total servers: 5,000 servers
    - Containers/pods: 25,000 containers
    - Applications: 500 microservices
    - Logs per server: 100 logs/sec (system, kernel, daemon logs)
    - Logs per container: 50 logs/sec (app logs)
    - Logs per application: 200 logs/sec (business logic)

    Total log ingestion:
    - Server logs: 5,000 × 100 = 500K logs/sec
    - Container logs: 25,000 × 50 = 1.25M logs/sec
    - Application logs: 500 × 200 = 100K logs/sec
    - Total: ~1.85M logs/sec (peak: 5M during incidents)
    - Design for: 100K logs/sec average (conservative estimate)

    Log volume:
    - Average log size: 500 bytes (single line)
    - Average multi-line log: 2 KB (stack traces)
    - Effective average: 1 KB per log entry

    Daily volume:
    - 100K logs/sec × 86,400 sec/day = 8.64B logs/day
    - 8.64B logs × 1 KB = 8.64 TB/day raw
    - With compression (10:1): ~864 GB/day

    Query load:
    - Active developers: 1,000 developers
    - Queries per developer: 50 queries/day
    - Total queries: 50,000 queries/day ≈ 1 query/sec
    - Peak (incident): 100 queries/sec

    Log streaming:
    - Active live tail sessions: 100 concurrent
    - Logs per stream: 10 logs/sec per session
    - Total streaming: 1K logs/sec = 1 MB/sec
    ```

    ### Storage Estimates

    ```
    Per log entry:
    - Timestamp: 8 bytes (Unix timestamp)
    - Log level: 10 bytes (INFO, ERROR, WARN)
    - Service name: 50 bytes
    - Host/container ID: 64 bytes (hostname + pod name)
    - Message: 500 bytes (avg)
    - Structured fields: 200 bytes (JSON fields)
    - Metadata: 100 bytes (labels, tags)
    - Total per log: ~932 bytes ≈ 1 KB

    Raw storage (no compression):
    - 100K logs/sec × 1 KB = 100 MB/sec
    - Daily: 100 MB/sec × 86,400 = 8.64 TB/day
    - Monthly: 8.64 TB/day × 30 = 259 TB/month

    With compression (10:1):
    - Daily: 864 GB/day
    - Monthly: 25.9 TB/month

    Retention tiers (hot-warm-cold):

    Tier 1: Hot storage (SSD) - Last 7 days
    - Storage: 864 GB/day × 7 = 6 TB
    - Use: Real-time search, active debugging
    - Search latency: < 100ms

    Tier 2: Warm storage (SSD) - 8-30 days
    - Storage: 864 GB/day × 23 = 19.9 TB
    - Use: Recent investigations, weekly analysis
    - Search latency: < 500ms

    Tier 3: Cold storage (S3/GCS) - 31-90 days
    - Storage: 864 GB/day × 60 = 51.8 TB
    - Use: Compliance, historical analysis
    - Search latency: 2-5 seconds (rehydration)

    Total storage: 6 + 19.9 + 51.8 = 77.7 TB

    With replication (3x for hot/warm, 1x for cold):
    - Hot: 6 TB × 3 = 18 TB
    - Warm: 19.9 TB × 3 = 59.7 TB
    - Cold: 51.8 TB × 1 = 51.8 TB
    - Total: 129.5 TB

    Index storage (inverted index):
    - Index size: ~30% of raw data
    - Hot index: 6 TB × 0.3 = 1.8 TB
    - Warm index: 19.9 TB × 0.3 = 6 TB
    - Total index: 7.8 TB (in-memory + disk)
    ```

    ### Bandwidth Estimates

    ```
    Ingress (log collection):
    - 100K logs/sec × 1 KB = 100 MB/sec ≈ 800 Mbps
    - Compressed: 80 Mbps
    - Peak (5x): 400 Mbps

    Egress (search queries):
    - 1 query/sec × 100 logs × 1 KB = 100 KB/sec ≈ 1 Mbps
    - Peak (incidents): 100 queries/sec = 100 Mbps
    - Log streaming: 1 MB/sec = 8 Mbps

    Total bandwidth: 80 Mbps (ingress) + 100 Mbps (egress) = 180 Mbps
    Peak: 400 Mbps (ingress) + 100 Mbps (egress) = 500 Mbps
    ```

    ### Server Estimates

    ```
    Log shippers (agents):
    - 5,000 servers + 25,000 containers = 30,000 agents
    - Each agent: Filebeat/Fluentd (50 MB RAM, 0.1 CPU)

    Ingestion layer (Logstash/Fluentd):
    - 100K logs/sec / 10K logs per node = 10 ingestion nodes
    - CPU: 8 cores per node (parsing, enrichment)
    - Memory: 16 GB per node (buffering)

    Message queue (Kafka):
    - 100 MB/sec ingestion / 50 MB per broker = 2-3 brokers
    - Replication: 3x = 9 Kafka brokers
    - Storage: 24 hours buffer = 8.64 TB × 3 = 26 TB

    Indexing layer (Elasticsearch):
    - 77.7 TB / 2 TB per node = 39 data nodes
    - CPU: 16 cores per node (indexing)
    - Memory: 64 GB per node (heap: 31 GB, file cache: 33 GB)
    - Disk: 2 TB SSD per node

    Query layer (Elasticsearch):
    - Coordinating nodes: 5 nodes (query routing)
    - CPU: 8 cores per node
    - Memory: 32 GB per node

    Total servers:
    - Log shippers: 30,000 agents (embedded)
    - Ingestion: 10 nodes
    - Kafka: 9 brokers
    - Elasticsearch: 39 data + 5 coordinating = 44 nodes
    - Total: ~63 dedicated nodes
    ```

    ---

    ## Key Assumptions

    1. Average log size: 1 KB (500 bytes text + 500 bytes metadata)
    2. 100K logs/sec sustained (5M peak during incidents)
    3. 10:1 compression ratio (text logs compress well)
    4. 7-day hot storage for 95% of queries
    5. 3x replication for durability (hot/warm), 1x for cold
    6. 30% index overhead (inverted index size)

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Agent-based collection:** Lightweight agents on each host (Filebeat)
    2. **Message queue buffering:** Kafka for durability and backpressure
    3. **Pipeline processing:** Parse, enrich, transform logs (Logstash)
    4. **Inverted indexing:** Fast full-text search (Elasticsearch)
    5. **Multi-tier storage:** Hot-warm-cold architecture for cost optimization
    6. **Real-time streaming:** WebSocket for live log tail

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Log Sources"
            Server1[Server 1<br/>App logs<br/>System logs]
            Server2[Server 2<br/>Nginx logs<br/>App logs]
            K8s[Kubernetes Pods<br/>Container logs<br/>stdout/stderr]
            App1[Application 1<br/>Structured logs<br/>JSON]
            App2[Application 2<br/>Unstructured logs<br/>Plain text]
        end

        subgraph "Log Collection Layer"
            Filebeat1[Filebeat Agent 1<br/>File tailing<br/>Backpressure]
            Filebeat2[Filebeat Agent 2<br/>File tailing]
            FluentBit[Fluent Bit<br/>Kubernetes logs<br/>Sidecar container]
            DirectAPI[Direct API<br/>HTTP POST<br/>Batch ingestion]
        end

        subgraph "Message Queue Buffer"
            Kafka1[Kafka Broker 1<br/>Topic: logs-raw]
            Kafka2[Kafka Broker 2<br/>Topic: logs-raw]
            Kafka3[Kafka Broker 3<br/>Topic: logs-raw]
            ZK[ZooKeeper<br/>Coordination]
        end

        subgraph "Log Processing Pipeline"
            Logstash1[Logstash 1<br/>Parse<br/>Grok patterns]
            Logstash2[Logstash 2<br/>Enrich<br/>Add metadata]
            Logstash3[Logstash N<br/>Transform<br/>Filter]
        end

        subgraph "Indexing & Search (Elasticsearch Cluster)"
            subgraph "Master Nodes"
                Master1[Master 1<br/>Cluster state]
                Master2[Master 2<br/>Cluster state]
                Master3[Master 3<br/>Cluster state]
            end

            subgraph "Coordinating Nodes"
                Coord1[Coordinator 1<br/>Query routing]
                Coord2[Coordinator 2<br/>Query routing]
            end

            subgraph "Data Nodes (Hot)"
                Hot1[Hot Node 1<br/>SSD<br/>Last 7 days]
                Hot2[Hot Node 2<br/>SSD]
                Hot3[Hot Node N<br/>SSD]
            end

            subgraph "Data Nodes (Warm)"
                Warm1[Warm Node 1<br/>SSD<br/>8-30 days]
                Warm2[Warm Node 2<br/>SSD]
            end

            subgraph "Data Nodes (Cold)"
                Cold[(Cold Storage<br/>S3/GCS<br/>31-90 days<br/>Snapshots)]
            end
        end

        subgraph "Alerting & Monitoring"
            AlertEngine[Alert Engine<br/>Pattern matching<br/>Threshold checks]
            Watcher[Elasticsearch Watcher<br/>Rule evaluation]
            Notifications[Notifications<br/>Slack/PagerDuty<br/>Email]
        end

        subgraph "Visualization Layer"
            Kibana[Kibana<br/>Dashboards<br/>Log viewer<br/>Discover]
            API[REST API<br/>Search queries<br/>Aggregations]
            WebSocket[WebSocket<br/>Live log tail]
        end

        subgraph "Storage Management"
            ILM[Index Lifecycle<br/>Management<br/>Hot→Warm→Cold]
            Curator[Curator<br/>Snapshot<br/>Delete old indices]
        end

        Server1 --> Filebeat1
        Server2 --> Filebeat2
        K8s --> FluentBit
        App1 --> DirectAPI
        App2 --> DirectAPI

        Filebeat1 --> Kafka1
        Filebeat2 --> Kafka2
        FluentBit --> Kafka3
        DirectAPI --> Kafka1

        Kafka1 --> ZK
        Kafka2 --> ZK
        Kafka3 --> ZK

        Kafka1 --> Logstash1
        Kafka2 --> Logstash2
        Kafka3 --> Logstash3

        Logstash1 --> Coord1
        Logstash2 --> Coord2
        Logstash3 --> Coord1

        Coord1 --> Master1
        Coord2 --> Master2

        Coord1 --> Hot1
        Coord1 --> Hot2
        Coord1 --> Hot3

        Coord2 --> Warm1
        Coord2 --> Warm2

        Hot1 --> ILM
        Hot2 --> ILM
        Hot3 --> ILM

        ILM --> Warm1
        ILM --> Warm2
        Warm1 --> ILM
        ILM --> Cold

        Warm1 --> Curator
        Curator --> Cold

        Kibana --> API
        API --> Coord1
        API --> Coord2

        WebSocket --> Coord1

        Watcher --> Coord1
        Watcher --> AlertEngine
        AlertEngine --> Notifications

        Master1 --> Master2
        Master2 --> Master3

        style Kafka1 fill:#fff4e1
        style Kafka2 fill:#fff4e1
        style Kafka3 fill:#fff4e1
        style Hot1 fill:#ffe1e1
        style Hot2 fill:#ffe1e1
        style Hot3 fill:#ffe1e1
        style Warm1 fill:#fff4e1
        style Warm2 fill:#fff4e1
        style Cold fill:#f0f0f0
        style Kibana fill:#e8f5e9
    ```

    ---

    ## Component Rationale

    | Component | Why We Need It | Alternative Considered |
    |-----------|----------------|----------------------|
    | **Filebeat (Log Shipper)** | Lightweight (50 MB), backpressure handling, guaranteed delivery | Fluentd (heavier, 200 MB), custom agents (complex) |
    | **Kafka (Message Queue)** | Durability (no log loss), buffering (handle spikes), replay capability | Direct to Elasticsearch (no buffer, data loss), RabbitMQ (no replay) |
    | **Logstash (Pipeline)** | Grok parsing (regex for unstructured logs), enrichment (GeoIP, DNS) | Ingest nodes only (less flexible), custom parsers (complex) |
    | **Elasticsearch (Index)** | Inverted index (fast full-text search), distributed, sharding, replication | Splunk (expensive), Loki (labels only, no full-text), ClickHouse (no full-text) |
    | **Hot-Warm-Cold Storage** | 90% cost savings (S3 vs. SSD), query performance (hot = fast) | All-SSD (10x cost), all-S3 (slow queries) |
    | **Index Lifecycle Management** | Automated tier transitions, deletion, cost optimization | Manual scripts (error-prone), no automation (expensive) |

    **Key Trade-off:** We chose **availability over consistency**. During network partitions, log ingestion continues (Kafka buffering), but search results may be slightly stale (eventual consistency).

    ---

    ## API Design

    ### 1. Log Ingestion (Direct API)

    **Request:**
    ```bash
    POST /api/v1/logs
    Content-Type: application/json
    Authorization: Bearer <api_token>

    {
      "logs": [
        {
          "timestamp": "2026-02-05T10:30:15.123Z",
          "level": "ERROR",
          "service": "payment-service",
          "host": "payment-01.prod.us-east-1",
          "message": "Payment processing failed: insufficient funds",
          "fields": {
            "user_id": "user_12345",
            "transaction_id": "txn_abc789",
            "amount": 99.99,
            "currency": "USD",
            "error_code": "INSUFFICIENT_FUNDS"
          },
          "tags": ["payment", "error", "critical"]
        },
        {
          "timestamp": "2026-02-05T10:30:16.456Z",
          "level": "INFO",
          "service": "payment-service",
          "host": "payment-01.prod.us-east-1",
          "message": "Payment retry initiated",
          "fields": {
            "user_id": "user_12345",
            "transaction_id": "txn_abc789",
            "retry_count": 1
          }
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "status": "success",
      "accepted": 2,
      "rejected": 0,
      "errors": []
    }
    ```

    **Design Notes:**

    - Batch multiple logs in single request (reduce overhead)
    - Structured fields for fast filtering (JSON nested fields)
    - Tags for categorization and dashboard grouping
    - ISO 8601 timestamp format (timezone-aware)

    ---

    ### 2. Search Logs (Full-Text Query)

    **Request:**
    ```bash
    POST /api/v1/logs/search
    Content-Type: application/json
    Authorization: Bearer <token>

    {
      "query": {
        "bool": {
          "must": [
            {
              "match": {
                "message": "payment failed"
              }
            },
            {
              "term": {
                "level": "ERROR"
              }
            },
            {
              "range": {
                "timestamp": {
                  "gte": "2026-02-05T00:00:00Z",
                  "lte": "2026-02-05T23:59:59Z"
                }
              }
            }
          ],
          "filter": [
            {
              "term": {
                "service": "payment-service"
              }
            }
          ]
        }
      },
      "size": 100,
      "sort": [
        {
          "timestamp": {
            "order": "desc"
          }
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "total": 1543,
      "hits": [
        {
          "timestamp": "2026-02-05T10:30:15.123Z",
          "level": "ERROR",
          "service": "payment-service",
          "host": "payment-01.prod.us-east-1",
          "message": "Payment processing failed: insufficient funds",
          "fields": {
            "user_id": "user_12345",
            "transaction_id": "txn_abc789",
            "error_code": "INSUFFICIENT_FUNDS"
          },
          "highlights": {
            "message": ["<em>Payment</em> processing <em>failed</em>: insufficient funds"]
          }
        }
      ],
      "took_ms": 87,
      "timed_out": false
    }
    ```

    **Query Types:**

    - `match` - Full-text search (tokenized, scored)
    - `term` - Exact match (keyword fields)
    - `range` - Timestamp, numeric range queries
    - `wildcard` - Pattern matching (e.g., "error-*")
    - `regex` - Regular expression queries
    - `exists` - Check if field exists

    ---

    ### 3. Aggregation Query (Error Rate)

    **Request:**
    ```bash
    POST /api/v1/logs/aggregate
    Content-Type: application/json

    {
      "query": {
        "range": {
          "timestamp": {
            "gte": "now-1h"
          }
        }
      },
      "aggs": {
        "error_rate": {
          "date_histogram": {
            "field": "timestamp",
            "interval": "1m"
          },
          "aggs": {
            "errors": {
              "filter": {
                "term": {
                  "level": "ERROR"
                }
              }
            }
          }
        },
        "top_services": {
          "terms": {
            "field": "service",
            "size": 10
          }
        },
        "error_types": {
          "terms": {
            "field": "fields.error_code",
            "size": 20
          }
        }
      }
    }
    ```

    **Response:**
    ```json
    {
      "aggregations": {
        "error_rate": {
          "buckets": [
            {
              "key": "2026-02-05T10:00:00Z",
              "doc_count": 12500,
              "errors": {
                "doc_count": 325
              }
            },
            {
              "key": "2026-02-05T10:01:00Z",
              "doc_count": 12800,
              "errors": {
                "doc_count": 450
              }
            }
          ]
        },
        "top_services": {
          "buckets": [
            {
              "key": "payment-service",
              "doc_count": 45000
            },
            {
              "key": "user-service",
              "doc_count": 38000
            }
          ]
        },
        "error_types": {
          "buckets": [
            {
              "key": "INSUFFICIENT_FUNDS",
              "doc_count": 1200
            },
            {
              "key": "NETWORK_TIMEOUT",
              "doc_count": 850
            }
          ]
        }
      }
    }
    ```

    ---

    ### 4. Live Log Tail (WebSocket)

    **WebSocket Connection:**
    ```javascript
    const ws = new WebSocket('wss://logs.example.com/api/v1/logs/tail');

    ws.onopen = () => {
      // Subscribe to specific service logs
      ws.send(JSON.stringify({
        action: 'subscribe',
        filters: {
          service: 'payment-service',
          level: ['ERROR', 'WARN']
        }
      }));
    };

    ws.onmessage = (event) => {
      const log = JSON.parse(event.data);
      console.log(`[${log.timestamp}] ${log.level}: ${log.message}`);
    };
    ```

    **Server Push:**
    ```json
    {
      "timestamp": "2026-02-05T10:30:20.789Z",
      "level": "ERROR",
      "service": "payment-service",
      "message": "Database connection timeout after 30s"
    }
    ```

    ---

    ### 5. Create Alert Rule

    **Request:**
    ```bash
    POST /api/v1/alerts/rules
    Content-Type: application/json

    {
      "name": "HighErrorRate",
      "description": "Alert when error rate exceeds 5% in 5 minutes",
      "query": {
        "bool": {
          "must": [
            {
              "term": {
                "level": "ERROR"
              }
            },
            {
              "range": {
                "timestamp": {
                  "gte": "now-5m"
                }
              }
            }
          ]
        }
      },
      "condition": {
        "type": "threshold",
        "operator": "gt",
        "value": 1000
      },
      "schedule": {
        "interval": "1m"
      },
      "actions": [
        {
          "type": "slack",
          "channel": "#alerts-production",
          "message": "🚨 High error rate: {{count}} errors in last 5 minutes"
        },
        {
          "type": "pagerduty",
          "severity": "critical"
        }
      ]
    }
    ```

    **Response:**
    ```json
    {
      "rule_id": "alert_xyz123",
      "status": "active",
      "created_at": "2026-02-05T10:30:00Z"
    }
    ```

    ---

    ## Database Schema

    ### Elasticsearch Index Mapping

    **Index Template:**

    ```json
    {
      "index_patterns": ["logs-*"],
      "settings": {
        "number_of_shards": 5,
        "number_of_replicas": 2,
        "refresh_interval": "5s",
        "codec": "best_compression",
        "index.lifecycle.name": "logs-policy"
      },
      "mappings": {
        "properties": {
          "timestamp": {
            "type": "date",
            "format": "strict_date_optional_time||epoch_millis"
          },
          "level": {
            "type": "keyword"
          },
          "service": {
            "type": "keyword",
            "fields": {
              "text": {
                "type": "text"
              }
            }
          },
          "host": {
            "type": "keyword"
          },
          "message": {
            "type": "text",
            "analyzer": "standard",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "fields": {
            "type": "object",
            "dynamic": true
          },
          "tags": {
            "type": "keyword"
          },
          "source_ip": {
            "type": "ip"
          },
          "duration_ms": {
            "type": "long"
          },
          "geo": {
            "type": "geo_point"
          }
        }
      }
    }
    ```

    **Inverted Index Structure (Conceptual):**

    ```
    Term Dictionary (message field):
    ┌─────────────────────────────────────────┐
    │ Term: "payment"                         │
    │   Document IDs: [1, 5, 8, 12, 15, ...]  │
    │   Positions: [[0], [3], [1], ...]       │
    │                                         │
    │ Term: "failed"                          │
    │   Document IDs: [1, 8, 15, 23, ...]     │
    │   Positions: [[2], [4], [3], ...]       │
    │                                         │
    │ Term: "error"                           │
    │   Document IDs: [1, 3, 8, 9, 12, ...]   │
    │   Positions: [[5], [0], [2], ...]       │
    └─────────────────────────────────────────┘

    Document Store:
    ┌─────────────────────────────────────────┐
    │ Doc ID: 1                               │
    │ {                                       │
    │   "timestamp": "2026-02-05T10:30:15Z",  │
    │   "level": "ERROR",                     │
    │   "message": "payment processing failed"│
    │ }                                       │
    └─────────────────────────────────────────┘

    Field Data (for aggregations):
    ┌─────────────────────────────────────────┐
    │ Field: "level"                          │
    │   ERROR: [1, 3, 8, 9, 12, ...]          │
    │   WARN: [2, 4, 6, 10, ...]              │
    │   INFO: [5, 7, 11, 13, ...]             │
    └─────────────────────────────────────────┘
    ```

    ---

    ### PostgreSQL (Metadata Store)

    **Alert Rules:**

    ```sql
    CREATE TABLE alert_rules (
        rule_id UUID PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        query JSONB NOT NULL,                 -- Elasticsearch DSL query
        condition JSONB NOT NULL,             -- Threshold, comparison
        schedule JSONB NOT NULL,              -- Interval, cron
        actions JSONB NOT NULL,               -- Slack, PagerDuty, email
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        last_evaluated TIMESTAMP
    );

    CREATE INDEX idx_active_rules ON alert_rules(is_active) WHERE is_active = true;
    ```

    **Alert History:**

    ```sql
    CREATE TABLE alert_history (
        alert_id UUID PRIMARY KEY,
        rule_id UUID REFERENCES alert_rules(rule_id),
        state VARCHAR(20) NOT NULL,           -- pending, firing, resolved
        match_count BIGINT,
        query_result JSONB,
        fired_at TIMESTAMP,
        resolved_at TIMESTAMP,
        notified BOOLEAN DEFAULT false
    );

    CREATE INDEX idx_firing_alerts ON alert_history(state) WHERE state = 'firing';
    CREATE INDEX idx_rule_history ON alert_history(rule_id, fired_at);
    ```

    **User Dashboards:**

    ```sql
    CREATE TABLE dashboards (
        dashboard_id UUID PRIMARY KEY,
        user_id VARCHAR(255),
        name VARCHAR(255) NOT NULL,
        description TEXT,
        panels JSONB NOT NULL,                -- Array of panel configs
        filters JSONB,                        -- Default filters
        refresh_interval INT DEFAULT 30,      -- Seconds
        is_public BOOLEAN DEFAULT false,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    ```

    ---

    ## Data Flow Diagrams

    ### Write Path (Log Ingestion)

    ```mermaid
    sequenceDiagram
        participant App as Application<br/>(Java/Python)
        participant Filebeat as Filebeat Agent
        participant Kafka
        participant Logstash
        participant ES as Elasticsearch

        App->>App: Write log to file<br/>/var/log/app.log

        Filebeat->>Filebeat: Tail log file<br/>Detect new lines
        Filebeat->>Filebeat: Add metadata<br/>(host, labels)

        Filebeat->>Kafka: Publish to topic<br/>"logs-raw"<br/>(batch: 1000 logs)

        Note over Filebeat,Kafka: Backpressure:<br/>If Kafka slow,<br/>Filebeat buffers

        Kafka-->>Filebeat: Ack (committed)

        Logstash->>Kafka: Consume logs<br/>(consumer group)

        Logstash->>Logstash: Parse log with Grok:<br/>"%{TIMESTAMP} %{LOGLEVEL} %{MESSAGE}"

        Logstash->>Logstash: Enrich:<br/>- GeoIP lookup<br/>- DNS resolution<br/>- Add tags

        Logstash->>Logstash: Transform:<br/>- Normalize fields<br/>- Drop sensitive data

        Logstash->>ES: Bulk index request<br/>(batch: 5000 docs)

        ES->>ES: Index documents:<br/>- Build inverted index<br/>- Store doc in segment

        ES-->>Logstash: Success

        Logstash->>Kafka: Commit offset

        Note over ES: Refresh interval: 5s<br/>Logs searchable after 5s
    ```

    ---

    ### Query Path (Log Search)

    ```mermaid
    sequenceDiagram
        participant User as Developer<br/>(Kibana)
        participant API as Search API
        participant Coord as Coordinator Node
        participant Hot as Hot Data Node
        participant Warm as Warm Data Node

        User->>API: Search query:<br/>level:ERROR AND message:"timeout"<br/>Last 24 hours

        API->>Coord: Execute query<br/>(Elasticsearch DSL)

        Coord->>Coord: Parse query<br/>Optimize<br/>Identify shards

        Note over Coord: Query: last 24h<br/>Target: hot indices only

        par Query hot shards
            Coord->>Hot: Shard 0: Execute query
            Coord->>Hot: Shard 1: Execute query
            Coord->>Hot: Shard 2: Execute query
        end

        Hot->>Hot: Use inverted index:<br/>1. Lookup "error" term<br/>2. Lookup "timeout" term<br/>3. Intersect doc IDs

        Hot->>Hot: Score documents<br/>BM25 ranking

        Hot-->>Coord: Top 100 docs per shard

        Coord->>Coord: Merge results<br/>Global sort<br/>Top 100 overall

        Coord->>Hot: Fetch document fields<br/>(doc IDs)

        Hot-->>Coord: Full documents

        Coord-->>API: Search results<br/>(100 docs, 87ms)

        API-->>User: Display logs<br/>with highlights
    ```

    ---

    ### Alert Evaluation Path

    ```mermaid
    sequenceDiagram
        participant Scheduler as Cron Scheduler<br/>(1 min interval)
        participant Engine as Alert Engine
        participant ES as Elasticsearch
        participant DB as PostgreSQL
        participant Notifier as Slack/PagerDuty

        Scheduler->>Engine: Trigger evaluation

        Engine->>DB: Fetch active alert rules

        loop For each rule
            Engine->>ES: Execute rule query<br/>count(level:ERROR) last 5m

            ES->>ES: Run aggregation query
            ES-->>Engine: Result: 1543 errors

            Engine->>Engine: Evaluate condition:<br/>1543 > 1000 ✓

            alt Condition met
                Engine->>DB: Check alert state

                alt Not firing yet
                    Engine->>DB: Create alert:<br/>state = firing

                    Engine->>Notifier: Send notification:<br/>"🚨 High error rate: 1543 errors"

                    Notifier-->>Engine: Notification sent

                else Already firing
                    Engine->>DB: Update last_seen

                    Note over Engine: Don't re-notify<br/>(debouncing)
                end

            else Condition not met
                Engine->>DB: Check if was firing

                alt Was firing
                    Engine->>DB: Update state: resolved

                    Engine->>Notifier: Send resolved notification

                else Not firing
                    Note over Engine: No action needed
                end
            end
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## 3.1 Log Parsing with Grok Patterns

    ### Grok Pattern Matching

    ```python
    class GrokParser:
        """
        Parse unstructured logs into structured fields using Grok patterns

        Grok = regex on steroids (named capture groups with built-in patterns)

        Example:
        Raw log: "2026-02-05 10:30:15 ERROR [payment-service] Payment failed for user 12345"

        Grok pattern:
        %{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} \[%{DATA:service}\] %{GREEDYDATA:message}

        Result:
        {
          "timestamp": "2026-02-05 10:30:15",
          "level": "ERROR",
          "service": "payment-service",
          "message": "Payment failed for user 12345"
        }
        """

        def __init__(self):
            # Built-in Grok patterns
            self.patterns = {
                'TIMESTAMP_ISO8601': r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?',
                'LOGLEVEL': r'(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)',
                'DATA': r'.*?',
                'GREEDYDATA': r'.*',
                'NUMBER': r'(?:\d+\.?\d*)',
                'IP': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
                'WORD': r'\b\w+\b',
                'QUOTEDSTRING': r'"(?:[^"\\]|\\.)*"',
                'UUID': r'[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}',
                'PATH': r'(?:/[^/\s]+)+',
            }

            # Application-specific patterns
            self.custom_patterns = {
                'NGINX_ACCESS': (
                    r'%{IP:client_ip} - %{DATA:user} \[%{TIMESTAMP_ISO8601:timestamp}\] '
                    r'"%{WORD:method} %{PATH:path} HTTP/%{NUMBER:http_version}" '
                    r'%{NUMBER:status_code} %{NUMBER:bytes_sent}'
                ),
                'JAVA_EXCEPTION': (
                    r'%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} '
                    r'\[%{DATA:thread}\] %{DATA:class} - %{GREEDYDATA:message}'
                ),
                'SYSLOG': (
                    r'%{TIMESTAMP_ISO8601:timestamp} %{WORD:hostname} '
                    r'%{WORD:program}\[%{NUMBER:pid}\]: %{GREEDYDATA:message}'
                ),
            }

        def parse(self, log_line, pattern_name='GENERIC'):
            """
            Parse a log line using specified Grok pattern
            """
            # Get pattern
            if pattern_name in self.custom_patterns:
                pattern = self.custom_patterns[pattern_name]
            else:
                pattern = pattern_name

            # Expand Grok patterns to regex
            regex_pattern = self._expand_grok_pattern(pattern)

            # Match log line
            match = re.match(regex_pattern, log_line)

            if match:
                # Extract named groups
                fields = match.groupdict()

                # Post-process fields
                fields = self._post_process_fields(fields)

                return fields
            else:
                # Parsing failed - return raw log
                return {
                    'message': log_line,
                    'parse_error': True
                }

        def _expand_grok_pattern(self, pattern):
            """
            Expand Grok patterns like %{LOGLEVEL:level} to named regex groups
            """
            # Find all Grok patterns: %{PATTERN:field_name}
            grok_regex = r'%{(\w+):(\w+)}'

            def replacer(match):
                pattern_name = match.group(1)
                field_name = match.group(2)

                if pattern_name in self.patterns:
                    # Replace with named capture group
                    return f'(?P<{field_name}>{self.patterns[pattern_name]})'
                else:
                    return match.group(0)

            # Replace all Grok patterns
            regex = re.sub(grok_regex, replacer, pattern)

            return regex

        def _post_process_fields(self, fields):
            """
            Post-process extracted fields
            """
            processed = {}

            for key, value in fields.items():
                if value is None:
                    continue

                # Convert timestamps
                if key == 'timestamp':
                    processed[key] = self._parse_timestamp(value)

                # Convert numbers
                elif key in ['status_code', 'bytes_sent', 'pid', 'duration_ms']:
                    try:
                        processed[key] = int(value) if '.' not in value else float(value)
                    except ValueError:
                        processed[key] = value

                # Normalize log level
                elif key == 'level':
                    processed[key] = value.upper()

                else:
                    processed[key] = value

            return processed

        def _parse_timestamp(self, timestamp_str):
            """
            Parse timestamp string to ISO 8601 format
            """
            # Try multiple formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%d/%b/%Y:%H:%M:%S %z',  # Nginx format
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue

            # Fallback: return as-is
            return timestamp_str


    # Example usage
    parser = GrokParser()

    # Parse Nginx access log
    nginx_log = '192.168.1.100 - - [2026-02-05T10:30:15Z] "GET /api/users/12345 HTTP/1.1" 200 1543'
    fields = parser.parse(nginx_log, 'NGINX_ACCESS')
    print(fields)
    # Output:
    # {
    #   'client_ip': '192.168.1.100',
    #   'timestamp': '2026-02-05T10:30:15Z',
    #   'method': 'GET',
    #   'path': '/api/users/12345',
    #   'http_version': '1.1',
    #   'status_code': 200,
    #   'bytes_sent': 1543
    # }

    # Parse Java exception log
    java_log = '2026-02-05 10:30:15 ERROR [http-nio-8080-exec-1] c.e.PaymentService - Payment processing failed'
    fields = parser.parse(java_log, 'JAVA_EXCEPTION')
    print(fields)
    # Output:
    # {
    #   'timestamp': '2026-02-05T10:30:15',
    #   'level': 'ERROR',
    #   'thread': 'http-nio-8080-exec-1',
    #   'class': 'c.e.PaymentService',
    #   'message': 'Payment processing failed'
    # }
    ```

    ---

    ## 3.2 Inverted Index for Full-Text Search

    ### Building Inverted Index

    ```python
    class InvertedIndex:
        """
        Build inverted index for fast full-text search

        Inverted index maps terms to document IDs

        Example:
        Doc 1: "payment processing failed"
        Doc 2: "payment successful"
        Doc 3: "order processing started"

        Inverted index:
        "payment" -> [1, 2]
        "processing" -> [1, 3]
        "failed" -> [1]
        "successful" -> [2]
        "order" -> [3]
        "started" -> [3]
        """

        def __init__(self):
            self.index = {}  # term -> [(doc_id, positions), ...]
            self.documents = {}  # doc_id -> document
            self.doc_count = 0
            self.analyzer = self._create_analyzer()

        def _create_analyzer(self):
            """
            Create text analyzer (tokenization + normalization)

            Steps:
            1. Lowercase
            2. Remove punctuation
            3. Tokenize (split on whitespace)
            4. Remove stop words (the, a, is, etc.)
            5. Stemming (running -> run)
            """
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer

            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()

            def analyze(text):
                # Lowercase
                text = text.lower()

                # Remove punctuation
                text = re.sub(r'[^\w\s]', ' ', text)

                # Tokenize
                tokens = text.split()

                # Remove stop words and stem
                tokens = [
                    stemmer.stem(token)
                    for token in tokens
                    if token not in stop_words and len(token) > 2
                ]

                return tokens

            return analyze

        def index_document(self, doc_id, document):
            """
            Index a document
            """
            self.documents[doc_id] = document
            self.doc_count += 1

            # Analyze document text
            text = document.get('message', '')
            tokens = self.analyzer(text)

            # Build positional inverted index
            for position, term in enumerate(tokens):
                if term not in self.index:
                    self.index[term] = []

                # Store document ID and term position
                self.index[term].append((doc_id, position))

        def search(self, query_text):
            """
            Search documents using boolean query

            Example: "payment failed"
            Returns: Documents containing both "payment" AND "failed"
            """
            # Analyze query
            query_terms = self.analyzer(query_text)

            if not query_terms:
                return []

            # Get posting lists for each term
            posting_lists = []
            for term in query_terms:
                if term in self.index:
                    # Get unique document IDs for this term
                    doc_ids = set(doc_id for doc_id, _ in self.index[term])
                    posting_lists.append(doc_ids)
                else:
                    # Term not found - no results
                    return []

            # Intersect posting lists (AND operation)
            matching_docs = posting_lists[0]
            for posting_list in posting_lists[1:]:
                matching_docs = matching_docs.intersection(posting_list)

            # Retrieve documents and score
            results = []
            for doc_id in matching_docs:
                document = self.documents[doc_id]
                score = self._calculate_bm25_score(doc_id, query_terms)
                results.append((document, score))

            # Sort by score (descending)
            results.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, score in results]

        def _calculate_bm25_score(self, doc_id, query_terms):
            """
            Calculate BM25 relevance score

            BM25 = sum over query terms:
              IDF(term) * (TF(term) * (k1 + 1)) / (TF(term) + k1 * (1 - b + b * (doc_len / avg_doc_len)))

            Where:
            - IDF = inverse document frequency (rare terms score higher)
            - TF = term frequency in document
            - k1 = 1.2 (term frequency saturation)
            - b = 0.75 (length normalization)
            """
            k1 = 1.2
            b = 0.75

            # Calculate average document length
            avg_doc_len = sum(
                len(self.analyzer(doc.get('message', '')))
                for doc in self.documents.values()
            ) / self.doc_count

            # Get document
            document = self.documents[doc_id]
            doc_text = document.get('message', '')
            doc_tokens = self.analyzer(doc_text)
            doc_len = len(doc_tokens)

            # Calculate BM25 score
            score = 0.0

            for term in query_terms:
                if term not in self.index:
                    continue

                # Term frequency (TF)
                tf = sum(1 for t in doc_tokens if t == term)

                # Inverse document frequency (IDF)
                doc_freq = len(set(doc_id for doc_id, _ in self.index[term]))
                idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

                # Length normalization
                norm = k1 * (1 - b + b * (doc_len / avg_doc_len))

                # BM25 formula
                term_score = idf * (tf * (k1 + 1)) / (tf + norm)
                score += term_score

            return score


    # Example usage
    index = InvertedIndex()

    # Index documents
    index.index_document(1, {
        'timestamp': '2026-02-05T10:30:15Z',
        'level': 'ERROR',
        'message': 'Payment processing failed due to network timeout'
    })

    index.index_document(2, {
        'timestamp': '2026-02-05T10:30:20Z',
        'level': 'INFO',
        'message': 'Payment processed successfully'
    })

    index.index_document(3, {
        'timestamp': '2026-02-05T10:30:25Z',
        'level': 'ERROR',
        'message': 'Order processing failed validation'
    })

    # Search
    results = index.search('payment failed')
    print(results)
    # Output: [document 1] (contains both "payment" and "failed")
    ```

    ---

    ## 3.3 Hot-Warm-Cold Architecture

    ### Index Lifecycle Management

    ```python
    class IndexLifecycleManager:
        """
        Manage index lifecycle: hot -> warm -> cold -> delete

        Phases:
        1. Hot (0-7 days): Fast indexing, fast search (SSD, active writing)
        2. Warm (8-30 days): Read-only, search-optimized (SSD, merged segments)
        3. Cold (31-90 days): Archived, slow search (S3, rehydrate on demand)
        4. Delete (>90 days): Deleted or moved to glacier

        Benefits:
        - 90% cost reduction (vs. all-hot)
        - Fast queries for recent data
        - Long retention for compliance
        """

        def __init__(self, es_client):
            self.es = es_client
            self.policy = {
                'hot': {
                    'max_age': '7d',
                    'max_size': '50gb',
                    'actions': {
                        'rollover': {},  # Create new index when conditions met
                        'set_priority': {'priority': 100}  # High priority for recovery
                    }
                },
                'warm': {
                    'min_age': '7d',
                    'actions': {
                        'readonly': {},  # Make index read-only
                        'forcemerge': {'max_num_segments': 1},  # Optimize for search
                        'shrink': {'number_of_shards': 1},  # Reduce shards
                        'allocate': {
                            'require': {
                                'data': 'warm'  # Move to warm tier nodes
                            }
                        },
                        'set_priority': {'priority': 50}
                    }
                },
                'cold': {
                    'min_age': '30d',
                    'actions': {
                        'searchable_snapshot': {
                            'snapshot_repository': 's3-snapshots'  # Move to S3
                        },
                        'allocate': {
                            'require': {
                                'data': 'cold'
                            }
                        },
                        'set_priority': {'priority': 0}
                    }
                },
                'delete': {
                    'min_age': '90d',
                    'actions': {
                        'delete': {}  # Delete index
                    }
                }
            }

        def create_policy(self):
            """
            Create ILM policy in Elasticsearch
            """
            policy_name = 'logs-policy'

            self.es.ilm.put_lifecycle(
                policy=policy_name,
                body={
                    'policy': {
                        'phases': self.policy
                    }
                }
            )

            print(f"Created ILM policy: {policy_name}")

        def apply_policy_to_index(self, index_name):
            """
            Apply ILM policy to index
            """
            self.es.indices.put_settings(
                index=index_name,
                body={
                    'index.lifecycle.name': 'logs-policy',
                    'index.lifecycle.rollover_alias': 'logs'
                }
            )

        def check_index_phase(self, index_name):
            """
            Check current lifecycle phase of index
            """
            response = self.es.ilm.explain_lifecycle(index=index_name)

            if index_name in response['indices']:
                index_info = response['indices'][index_name]
                phase = index_info.get('phase', 'unknown')
                age = index_info.get('age', 'unknown')

                return {
                    'index': index_name,
                    'phase': phase,
                    'age': age,
                    'step': index_info.get('step', 'unknown')
                }

            return None

        def manual_phase_transition(self, index_name, target_phase):
            """
            Manually transition index to target phase
            """
            if target_phase == 'warm':
                # Set read-only
                self.es.indices.put_settings(
                    index=index_name,
                    body={'index.blocks.write': True}
                )

                # Force merge to 1 segment
                self.es.indices.forcemerge(
                    index=index_name,
                    max_num_segments=1
                )

                # Move to warm nodes
                self.es.indices.put_settings(
                    index=index_name,
                    body={
                        'index.routing.allocation.require.data': 'warm'
                    }
                )

            elif target_phase == 'cold':
                # Create snapshot
                snapshot_name = f"{index_name}-snapshot"
                self.es.snapshot.create(
                    repository='s3-snapshots',
                    snapshot=snapshot_name,
                    body={
                        'indices': index_name,
                        'include_global_state': False
                    }
                )

                # Delete local index (keep snapshot only)
                self.es.indices.delete(index=index_name)

            elif target_phase == 'delete':
                # Delete index
                self.es.indices.delete(index=index_name)


    # Example usage
    ilm = IndexLifecycleManager(es_client)

    # Create ILM policy
    ilm.create_policy()

    # Apply to index
    ilm.apply_policy_to_index('logs-2026.02.05')

    # Check phase
    phase_info = ilm.check_index_phase('logs-2026.02.05')
    print(phase_info)
    # Output: {'index': 'logs-2026.02.05', 'phase': 'hot', 'age': '2d'}
    ```

    ---

    ## 3.4 Real-Time Log Streaming

    ### WebSocket Log Tail

    ```python
    class LogTailServer:
        """
        Real-time log streaming via WebSocket

        Use case: Live log tail (like "tail -f" but distributed)

        Architecture:
        1. Client opens WebSocket connection
        2. Subscribe to specific filters (service, level)
        3. Server streams matching logs in real-time
        4. Client displays logs as they arrive
        """

        def __init__(self, es_client, kafka_consumer):
            self.es = es_client
            self.kafka = kafka_consumer
            self.active_connections = {}  # websocket_id -> (connection, filters)

        async def handle_connection(self, websocket, connection_id):
            """
            Handle WebSocket connection
            """
            try:
                # Register connection
                self.active_connections[connection_id] = {
                    'websocket': websocket,
                    'filters': None,
                    'buffer': []
                }

                # Wait for subscription message
                async for message in websocket:
                    data = json.loads(message)

                    if data['action'] == 'subscribe':
                        # Store filters
                        filters = data.get('filters', {})
                        self.active_connections[connection_id]['filters'] = filters

                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            'type': 'subscribed',
                            'filters': filters
                        }))

                        # Start streaming logs
                        await self.stream_logs(connection_id)

                    elif data['action'] == 'unsubscribe':
                        # Stop streaming
                        break

            finally:
                # Cleanup connection
                del self.active_connections[connection_id]

        async def stream_logs(self, connection_id):
            """
            Stream logs matching filters to client
            """
            connection = self.active_connections[connection_id]
            websocket = connection['websocket']
            filters = connection['filters']

            # Consume from Kafka (real-time logs)
            for message in self.kafka.consume():
                log = json.loads(message.value)

                # Apply filters
                if not self._matches_filters(log, filters):
                    continue

                # Send to client
                try:
                    await websocket.send(json.dumps({
                        'type': 'log',
                        'data': log
                    }))
                except:
                    # Client disconnected
                    break

        def _matches_filters(self, log, filters):
            """
            Check if log matches filters
            """
            if not filters:
                return True

            # Service filter
            if 'service' in filters:
                if log.get('service') != filters['service']:
                    return False

            # Log level filter
            if 'level' in filters:
                allowed_levels = filters['level'] if isinstance(filters['level'], list) else [filters['level']]
                if log.get('level') not in allowed_levels:
                    return False

            # Host filter
            if 'host' in filters:
                if filters['host'] not in log.get('host', ''):
                    return False

            # Text filter (substring match)
            if 'text' in filters:
                if filters['text'].lower() not in log.get('message', '').lower():
                    return False

            return True


    # Client-side example (JavaScript)
    """
    const ws = new WebSocket('wss://logs.example.com/api/v1/logs/tail');

    ws.onopen = () => {
        // Subscribe to payment service errors
        ws.send(JSON.stringify({
            action: 'subscribe',
            filters: {
                service: 'payment-service',
                level: ['ERROR', 'WARN']
            }
        }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'subscribed') {
            console.log('Subscribed with filters:', msg.filters);
        } else if (msg.type === 'log') {
            const log = msg.data;
            console.log(`[${log.timestamp}] ${log.level}: ${log.message}`);
        }
    };
    """
    ```

    ---

    ## 3.5 Log Compression

    ### Compression Strategies

    ```python
    class LogCompression:
        """
        Compress logs for storage efficiency

        Compression techniques:
        1. Dictionary encoding (repeated strings)
        2. Timestamp delta encoding
        3. Gzip/LZ4 compression
        4. Columnar storage (group by field type)

        Typical compression ratio: 10:1 for text logs
        """

        def compress_log_batch(self, logs):
            """
            Compress a batch of logs using dictionary encoding
            """
            # Extract repeated strings (service names, hosts, etc.)
            dictionary = self._build_dictionary(logs)

            # Encode logs using dictionary
            compressed_logs = []

            for log in logs:
                compressed_log = {}

                # Replace strings with dictionary IDs
                for field in ['service', 'host', 'level']:
                    if field in log:
                        value = log[field]
                        if value in dictionary:
                            compressed_log[f'{field}_id'] = dictionary[value]
                        else:
                            compressed_log[field] = value

                # Delta-encode timestamps
                if 'timestamp' in log:
                    compressed_log['timestamp_delta'] = self._delta_encode_timestamp(
                        log['timestamp'],
                        logs[0]['timestamp']  # Reference timestamp
                    )

                # Keep message as-is (will be gzipped)
                compressed_log['message'] = log['message']

                compressed_logs.append(compressed_log)

            # Serialize and compress
            serialized = json.dumps({
                'dictionary': {v: k for k, v in dictionary.items()},  # Reverse mapping
                'logs': compressed_logs
            })

            compressed = gzip.compress(serialized.encode('utf-8'))

            # Calculate compression ratio
            original_size = len(json.dumps(logs).encode('utf-8'))
            compressed_size = len(compressed)
            ratio = original_size / compressed_size

            return compressed, ratio

        def _build_dictionary(self, logs):
            """
            Build dictionary of repeated strings
            """
            string_counts = {}

            # Count string occurrences
            for log in logs:
                for field in ['service', 'host', 'level']:
                    if field in log:
                        value = log[field]
                        string_counts[value] = string_counts.get(value, 0) + 1

            # Create dictionary for strings appearing > 3 times
            dictionary = {}
            dict_id = 0

            for string, count in string_counts.items():
                if count > 3:
                    dictionary[string] = dict_id
                    dict_id += 1

            return dictionary

        def _delta_encode_timestamp(self, timestamp, base_timestamp):
            """
            Encode timestamp as delta from base

            Example:
            Base: 1675598400 (2026-02-05 10:00:00)
            Current: 1675598415 (2026-02-05 10:00:15)
            Delta: 15 seconds (2 bytes instead of 8 bytes)
            """
            base_ts = int(datetime.fromisoformat(base_timestamp).timestamp())
            current_ts = int(datetime.fromisoformat(timestamp).timestamp())

            return current_ts - base_ts


    # Compression results example:
    """
    Original logs (1000 logs):
    - Total size: 1,000 KB (1 KB per log)

    After compression:
    - Dictionary encoding: 800 KB (20% reduction)
    - Delta encoding: 750 KB (25% reduction)
    - Gzip: 100 KB (90% reduction)

    Overall compression ratio: 10:1
    """
    ```

=== "⚖️ Step 4: Trade-offs & Scale"

    ## Scalability Strategies

    ### Horizontal Scaling

    ```
    Log Collection (Agents):
    - Filebeat on every host (30K agents)
    - Automatic backpressure handling
    - No centralized coordination needed

    Ingestion Pipeline (Logstash):
    - Scale based on Kafka lag
    - Each node: 10K logs/sec
    - 100K logs/sec = 10 nodes
    - Auto-scale based on consumer lag

    Message Queue (Kafka):
    - Partition logs by service or timestamp
    - Replication factor: 3x
    - Scale brokers for throughput
    - Retention: 24 hours buffer

    Indexing (Elasticsearch):
    - Shard by time (daily indices)
    - Each shard: 20-50 GB
    - Hot nodes: SSD, fast indexing
    - Warm nodes: SSD, read-only
    - Cold nodes: S3 snapshots
    - Scale by adding data nodes

    Search (Elasticsearch):
    - Coordinating nodes route queries
    - Parallel query execution across shards
    - Scale coordinators for query load
    - Query cache for hot queries
    ```

    ---

    ## Performance Optimization

    | Optimization | Improvement | Trade-off |
    |-------------|-------------|-----------|
    | **Kafka buffering** | 100% durability, handle spikes | 1-5s ingestion latency |
    | **Batch indexing** | 10x throughput (5000 docs/bulk) | Increased latency |
    | **Hot-warm-cold** | 90% cost reduction | Query latency for cold data |
    | **Compression (10:1)** | 90% storage savings | CPU overhead (10-20%) |
    | **Index refresh (5s)** | Balance freshness vs. indexing speed | 5s delay for searchability |
    | **Force merge (warm)** | 50% faster queries (1 segment) | CPU-intensive operation |
    | **Query cache** | 80% hit rate, 5x faster | Stale results (30s TTL) |
    | **Field data cache** | Fast aggregations | High memory usage |

    ---

    ## Cost Optimization

    ```
    Monthly Cost (100K logs/sec, 90-day retention):

    Compute:
    - Filebeat agents: $0 (runs on existing hosts)
    - Ingestion (10 nodes): 10 × $200 = $2,000
    - Kafka (9 brokers): 9 × $300 = $2,700
    - Elasticsearch hot (15 nodes): 15 × $500 = $7,500
    - Elasticsearch warm (20 nodes): 20 × $400 = $8,000
    - Elasticsearch cold (S3): N/A (serverless)
    - Total compute: $20,200/month

    Storage:
    - Hot tier (18 TB SSD): 18 × $100 = $1,800
    - Warm tier (60 TB SSD): 60 × $100 = $6,000
    - Cold tier (52 TB S3): 52 × $23 = $1,196
    - Kafka buffer (26 TB): 26 × $100 = $2,600
    - Total storage: $11,596/month

    Network:
    - Ingress: 80 Mbps × 330 TB/month × $0.08/GB = $26,400
    - Egress: 100 Mbps × 330 TB/month × $0.08/GB = $26,400
    - Total network: $52,800/month (high due to distributed architecture)

    Total: ~$85K/month ≈ $1M/year

    Optimizations:
    1. Aggressive compression (10:1 → 15:1): -$2K storage
    2. Reduce cold retention (90d → 60d): -$400 storage
    3. Reserved instances (30% discount): -$6K compute
    4. Private network (reduce egress): -$20K network
    5. Smaller batch size (reduce Kafka buffer): -$1K storage

    Optimized Total: ~$55K/month ≈ $660K/year

    Cost per log: $0.000006 per log (660K / 100K logs/sec / 86400s / 30 days)
    ```

    ---

    ## Availability & Disaster Recovery

    ```
    High Availability:
    - Elasticsearch: 3 master nodes (quorum), 3 replicas per shard
    - Kafka: 3 brokers, replication factor 3
    - Logstash: Stateless, auto-scaling
    - Multi-AZ deployment

    Data Durability:
    - Kafka: 99.99% (replicated, persistent)
    - Elasticsearch: 99.99% (3 replicas)
    - S3 snapshots: 99.999999999% (eleven 9s)

    Failure Scenarios:

    1. Elasticsearch node failure:
       - Impact: None (replicas serve traffic)
       - Recovery: Automatic shard rebalancing

    2. Kafka broker failure:
       - Impact: None (replicated partitions)
       - Recovery: Leader election (< 1s)

    3. Logstash node failure:
       - Impact: Reduced throughput
       - Recovery: Kafka buffering, auto-scale replacement

    4. Network partition:
       - Impact: Logs buffered in Kafka (24h)
       - Recovery: Replay when partition heals

    5. Full Elasticsearch cluster failure:
       - Impact: Search unavailable
       - Recovery: Restore from S3 snapshots (1-2 hours)

    RTO (Recovery Time Objective): < 2 hours
    RPO (Recovery Point Objective): < 1 minute (Kafka buffer)
    ```

---

## Interview Tips

**Common Follow-up Questions:**

1. **"Why use Kafka between log shippers and Elasticsearch?"**
   - **Durability:** No log loss if Elasticsearch is down/slow
   - **Buffering:** Handle traffic spikes (5x peak)
   - **Backpressure:** Protect Elasticsearch from overload
   - **Replay:** Reprocess logs if parsing logic changes
   - **Decoupling:** Elasticsearch maintenance doesn't block ingestion

2. **"How do you parse unstructured logs?"**
   - **Grok patterns:** Regex-based extraction (Logstash)
   - **Example:** `%{TIMESTAMP} %{LOGLEVEL} %{GREEDYDATA:message}`
   - **Built-in patterns:** Nginx, Apache, syslog, Java exceptions
   - **Custom patterns:** Application-specific log formats
   - **Fallback:** If parsing fails, store as raw message

3. **"How does full-text search work?"**
   - **Inverted index:** Map terms to document IDs
   - **Tokenization:** Split text into terms (lowercase, stem)
   - **Boolean queries:** AND, OR, NOT operations
   - **Scoring:** BM25 algorithm (relevance ranking)
   - **Query optimization:** Push filters down, use cached results

4. **"What's the difference between hot, warm, and cold storage?"**
   - **Hot (0-7d):** SSD, active indexing, fast search (<100ms), highest cost
   - **Warm (8-30d):** SSD, read-only, merged segments, medium cost
   - **Cold (31-90d):** S3 snapshots, slow search (2-5s), lowest cost
   - **Benefit:** 90% cost reduction vs. all-hot, fast for recent queries

5. **"How do you handle log spikes during incidents?"**
   - **Kafka buffering:** 24-hour buffer capacity
   - **Auto-scaling:** Scale Logstash nodes based on consumer lag
   - **Rate limiting:** Drop low-priority logs if needed
   - **Sampling:** Sample verbose debug logs (keep errors)
   - **Alerting:** Alert on high ingestion lag

6. **"How do you alert on log patterns?"**
   - **Query-based alerts:** Run Elasticsearch query periodically (1 min)
   - **Threshold:** Count > N in time window (e.g., >1000 errors in 5 min)
   - **Pattern matching:** Regex or term match (e.g., "OutOfMemoryError")
   - **Aggregation:** Group by field (error rate per service)
   - **Deduplication:** Don't re-alert if already firing

7. **"What if Elasticsearch becomes slow?"**
   - **Add data nodes:** Rebalance shards across more nodes
   - **Optimize indices:** Force merge to 1 segment per shard
   - **Increase heap:** Up to 31 GB per node (JVM max)
   - **Use query cache:** Cache hot queries (80% hit rate)
   - **Reduce retention:** Move old data to cold tier sooner

**Key Points to Mention:**

- Agent-based collection (Filebeat) with backpressure handling
- Kafka buffering for durability and spike handling
- Grok parsing for unstructured logs
- Inverted index for fast full-text search
- Hot-warm-cold architecture (90% cost savings)
- Index lifecycle management (automated transitions)
- Real-time log streaming (WebSocket)
- 10:1 compression ratio (text logs compress well)
- Query optimization (cache, push-down filters)
- Scalability through sharding and horizontal scaling

---

## Real-World Examples

**ELK Stack (Elasticsearch, Logstash, Kibana):**
- Most popular open-source log aggregation
- Elasticsearch: Distributed search and analytics
- Logstash: Log processing pipeline
- Kibana: Visualization and dashboards
- Beats: Lightweight log shippers
- Used by: Netflix, LinkedIn, Adobe

**Splunk:**
- Commercial log aggregation platform
- Proprietary indexing (faster than Elasticsearch)
- Machine learning for anomaly detection
- 90-day free tier, expensive for scale
- Used by: Domino's, Nasdaq, Mercedes-Benz

**Grafana Loki:**
- Log aggregation focused on labels (not full-text)
- 10x cheaper than Elasticsearch (less indexing)
- Integrates with Prometheus and Grafana
- Good for Kubernetes (pod labels)
- Trade-off: No full-text search

**Datadog Logs:**
- SaaS log management platform
- Automatic parsing (no Grok config)
- 15-month retention
- Integration with metrics and traces
- Used by: Airbnb, Peloton, Samsung

---

## Summary

**System Characteristics:**

- **Ingestion:** 100K logs/sec (5M peak)
- **Storage:** 130 TB total (10:1 compression)
- **Search Latency:** <200ms p95
- **Availability:** 99.9% uptime
- **Retention:** 90 days (hot: 7d, warm: 30d, cold: 90d)

**Core Components:**

1. **Filebeat:** Lightweight log shipper on every host
2. **Kafka:** Message queue for buffering and durability
3. **Logstash:** Log parsing and enrichment pipeline
4. **Elasticsearch:** Distributed search and indexing
5. **Kibana:** Visualization and log viewer
6. **ILM:** Automated hot-warm-cold transitions

**Key Design Decisions:**

- Agent-based collection (Filebeat) with automatic backpressure
- Kafka buffering for durability (24-hour buffer)
- Grok parsing for unstructured logs (regex patterns)
- Inverted index for fast full-text search (Lucene)
- Hot-warm-cold storage (90% cost savings)
- Index lifecycle management (automated)
- 10:1 compression ratio (dictionary + gzip)
- Real-time streaming (WebSocket for live tail)
- Eventual consistency (availability over consistency)

This design provides a scalable, cost-effective log aggregation system capable of handling billions of log entries per day with fast search capabilities and long retention periods.
