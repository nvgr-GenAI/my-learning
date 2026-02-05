# Design a Health Checker

Design a service health monitoring system that continuously checks the health of multiple services and alerts when problems are detected.

**Difficulty:** 🟢 Easy | **Frequency:** ⭐⭐⭐ Medium | **Time:** 25-35 minutes

---

## Quick Overview

| Aspect | Details |
|--------|---------|
| **Scale** | 1000 services, 10K health checks/min, multi-region monitoring |
| **Key Challenges** | False positives, check frequency, alerting, service dependencies |
| **Core Concepts** | Heartbeat monitoring, HTTP probes, status aggregation, SLA tracking |
| **Companies** | All companies with microservices |

---

=== "📋 Step 1: Requirements"

    ## Functional Requirements

    **Core Features:**

    | Feature | Description | Priority |
    |---------|-------------|----------|
    | **Service Registration** | Register services with health check endpoints | P0 (Must have) |
    | **Health Checks** | Periodically check service health (HTTP, TCP, etc.) | P0 (Must have) |
    | **Status Dashboard** | View current health status of all services | P0 (Must have) |
    | **Alerting** | Notify when services become unhealthy | P0 (Must have) |
    | **Historical Data** | Track uptime and incident history | P1 (Should have) |
    | **Dependency Tracking** | Understand service dependencies | P2 (Nice to have) |

    **Explicitly Out of Scope:**

    - Auto-remediation (self-healing)
    - Distributed tracing
    - Log aggregation
    - Application performance monitoring (APM)
    - Infrastructure monitoring (CPU, memory, disk)

    ---

    ## Non-Functional Requirements

    | Requirement | Target | Reasoning |
    |-------------|--------|-----------|
    | **Availability** | 99.99% | Monitoring system must be highly available |
    | **Latency** | < 1s check duration | Fast failure detection |
    | **Scalability** | 10K checks/min | Support many services |
    | **Reliability** | < 0.1% false positives | Avoid alert fatigue |
    | **Multi-region** | Global monitoring | Detect regional outages |

    ---

    ## Capacity Estimation

    ### Traffic Estimates

    ```
    Services monitored: 1,000 services
    Check interval: 30 seconds
    Checks per service: 2 checks/minute

    Total health checks:
    - 1,000 services × 2 checks/min = 2,000 checks/min
    - Peak: 3x average = 6,000 checks/min

    With multi-region (3 regions):
    - 3 regions × 2,000 checks/min = 6,000 checks/min
    - Total: 100 checks/second
    ```

    ### Storage Estimates

    ```
    Health check result size: 500 bytes
    - Service ID: 50 bytes
    - Status: 10 bytes
    - Response time: 10 bytes
    - Timestamp: 20 bytes
    - Error message: 200 bytes
    - Metadata: 210 bytes

    Daily storage:
    - 2,000 checks/min × 60 min × 24 hours = 2.88M checks/day
    - 2.88M × 500 bytes = 1.44 GB/day

    Retention (30 days): 1.44 GB × 30 = ~43 GB
    Retention (1 year): 1.44 GB × 365 = ~525 GB
    ```

    ### Alert Volume

    ```
    Assuming 99.9% uptime target:
    - Downtime: 0.1% of time
    - Services down at any time: 1,000 × 0.001 = 1 service
    - Incidents per day: ~5-10 incidents

    Alert volume:
    - 5-10 incidents/day
    - ~1-2 alerts/hour during business hours
    ```

    ---

    ## Key Assumptions

    1. HTTP/HTTPS health check endpoints
    2. Services provide `/health` or `/ping` endpoints
    3. Simple up/down status (not degraded states initially)
    4. Alert via email, Slack, PagerDuty
    5. 30-second check interval is acceptable

=== "🏗️ Step 2: High-Level Design"

    ## System Architecture Overview

    **Core Design Principles:**

    1. **Distributed checkers** - Multiple checker instances for reliability
    2. **Configurable checks** - HTTP, TCP, custom protocols
    3. **Smart alerting** - Debouncing and escalation
    4. **Historical tracking** - Store check results for SLA reporting
    5. **Dependency awareness** - Understand service relationships

    ---

    ## Architecture Diagram

    ```mermaid
    graph TB
        subgraph "Monitored Services"
            S1[Service A<br/>/health]
            S2[Service B<br/>/health]
            S3[Service N<br/>/health]
        end

        subgraph "Health Checker System"
            API[API Server<br/>REST API]
            Scheduler[Check Scheduler<br/>Job Queue]
            Worker1[Health Check Worker 1]
            Worker2[Health Check Worker 2]
            WorkerN[Health Check Worker N]
            Alerting[Alert Manager<br/>Deduplication]
            Dashboard[Web Dashboard<br/>Real-time Status]
        end

        subgraph "Data Layer"
            DB[(PostgreSQL<br/>Config + Results)]
            Cache[(Redis<br/>Current Status)]
            TSStore[(InfluxDB<br/>Time Series Data)]
        end

        subgraph "Alert Channels"
            Email[Email]
            Slack[Slack]
            PagerDuty[PagerDuty]
        end

        API --> DB
        API --> Dashboard

        Scheduler --> DB
        Scheduler --> Worker1
        Scheduler --> Worker2
        Scheduler --> WorkerN

        Worker1 --> S1
        Worker2 --> S2
        WorkerN --> S3

        Worker1 --> Cache
        Worker2 --> Cache
        WorkerN --> Cache

        Worker1 --> TSStore
        Worker2 --> TSStore
        WorkerN --> TSStore

        Cache --> Alerting
        Alerting --> Email
        Alerting --> Slack
        Alerting --> PagerDuty

        Cache --> Dashboard

        style Cache fill:#e1f5ff
        style DB fill:#fff4e1
        style Alerting fill:#ffe1e1
    ```

    ---

    ## API Design

    ### 1. Register Service

    **Request:**
    ```http
    POST /api/v1/services
    Content-Type: application/json

    {
      "name": "user-service",
      "endpoint": "https://api.example.com/health",
      "check_type": "http",
      "check_interval": 30,
      "timeout": 5,
      "expected_status": 200,
      "expected_body": "OK",
      "alert_channels": ["email", "slack"],
      "metadata": {
        "team": "backend",
        "environment": "production",
        "region": "us-east-1"
      }
    }
    ```

    **Response:**
    ```http
    HTTP/1.1 201 Created
    Content-Type: application/json

    {
      "service_id": "svc_abc123",
      "name": "user-service",
      "status": "pending",
      "next_check": "2024-01-15T10:30:00Z"
    }
    ```

    ---

    ### 2. Get Service Status

    **Request:**
    ```http
    GET /api/v1/services/svc_abc123
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "service_id": "svc_abc123",
      "name": "user-service",
      "status": "healthy",
      "last_check": "2024-01-15T10:29:45Z",
      "response_time_ms": 145,
      "uptime_percentage": 99.95,
      "checks_total": 2880,
      "checks_failed": 2
    }
    ```

    ---

    ### 3. Get All Services Status

    **Request:**
    ```http
    GET /api/v1/services?status=unhealthy
    ```

    **Response:**
    ```http
    HTTP/1.1 200 OK
    Content-Type: application/json

    {
      "services": [
        {
          "service_id": "svc_xyz789",
          "name": "payment-service",
          "status": "unhealthy",
          "last_check": "2024-01-15T10:29:30Z",
          "error": "Connection timeout",
          "down_since": "2024-01-15T10:25:00Z"
        }
      ],
      "total": 1
    }
    ```

    ---

    ## Health Check Types

    ### 1. HTTP/HTTPS Check

    ```python
    import requests
    from typing import Dict, Optional

    class HTTPHealthCheck:
        def __init__(self, endpoint: str, timeout: int = 5,
                     expected_status: int = 200,
                     expected_body: Optional[str] = None):
            self.endpoint = endpoint
            self.timeout = timeout
            self.expected_status = expected_status
            self.expected_body = expected_body

        def check(self) -> Dict:
            """Perform HTTP health check"""
            start_time = time.time()

            try:
                response = requests.get(
                    self.endpoint,
                    timeout=self.timeout,
                    verify=True  # Verify SSL certificates
                )

                duration_ms = (time.time() - start_time) * 1000

                # Check status code
                if response.status_code != self.expected_status:
                    return {
                        "status": "unhealthy",
                        "error": f"Expected status {self.expected_status}, got {response.status_code}",
                        "response_time_ms": duration_ms
                    }

                # Check body if specified
                if self.expected_body and self.expected_body not in response.text:
                    return {
                        "status": "unhealthy",
                        "error": f"Expected body contains '{self.expected_body}'",
                        "response_time_ms": duration_ms
                    }

                return {
                    "status": "healthy",
                    "response_time_ms": duration_ms
                }

            except requests.Timeout:
                return {
                    "status": "unhealthy",
                    "error": "Request timeout",
                    "response_time_ms": self.timeout * 1000
                }
            except requests.ConnectionError as e:
                return {
                    "status": "unhealthy",
                    "error": f"Connection error: {str(e)}",
                    "response_time_ms": (time.time() - start_time) * 1000
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": f"Check failed: {str(e)}",
                    "response_time_ms": (time.time() - start_time) * 1000
                }
    ```

    ---

    ### 2. TCP Check

    ```python
    import socket

    class TCPHealthCheck:
        def __init__(self, host: str, port: int, timeout: int = 5):
            self.host = host
            self.port = port
            self.timeout = timeout

        def check(self) -> Dict:
            """Check if TCP port is open"""
            start_time = time.time()

            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                result = sock.connect_ex((self.host, self.port))
                sock.close()

                duration_ms = (time.time() - start_time) * 1000

                if result == 0:
                    return {
                        "status": "healthy",
                        "response_time_ms": duration_ms
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"Port {self.port} is closed",
                        "response_time_ms": duration_ms
                    }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "response_time_ms": (time.time() - start_time) * 1000
                }
    ```

    ---

    ## Data Flow

    ### Health Check Execution Flow

    ```mermaid
    sequenceDiagram
        participant Scheduler
        participant Queue
        participant Worker
        participant Service
        participant Redis
        participant Alerting
        participant InfluxDB

        Scheduler->>Queue: Enqueue health check job
        Worker->>Queue: Dequeue next job
        Worker->>Service: HTTP GET /health

        alt Service Healthy
            Service-->>Worker: 200 OK
            Worker->>Redis: Update status: healthy
            Worker->>InfluxDB: Store check result
        else Service Unhealthy
            Service-->>Worker: Error/Timeout
            Worker->>Redis: Update status: unhealthy
            Worker->>InfluxDB: Store check result
            Worker->>Alerting: Trigger alert
            Alerting->>Alerting: Check alert rules
            alt Should Alert
                Alerting-->>Alerting: Send to channels
            end
        end
    ```

=== "🔍 Step 3: Deep Dive"

    ## Key Topics

    ### 1. Alert Debouncing

    **Problem:** Avoid alert fatigue from transient failures

    **Solution: Require N consecutive failures**

    ```python
    class AlertManager:
        def __init__(self, failure_threshold: int = 3):
            self.failure_threshold = failure_threshold
            self.failure_counts = {}  # service_id -> count

        def should_alert(self, service_id: str, status: str) -> bool:
            """Determine if alert should be sent"""

            if status == "healthy":
                # Reset failure count on success
                self.failure_counts[service_id] = 0
                return False

            # Increment failure count
            count = self.failure_counts.get(service_id, 0) + 1
            self.failure_counts[service_id] = count

            # Alert only after threshold consecutive failures
            if count == self.failure_threshold:
                return True

            return False
    ```

    **Benefits:**
    - Filters out transient network blips
    - Reduces false positive alerts
    - Still detects real outages quickly

    ---

    ### 2. Check Scheduling

    **Approaches:**

    **Simple: Fixed Interval**
    ```python
    # Check every 30 seconds
    schedule.every(30).seconds.do(check_service)
    ```

    **Better: Distributed Job Queue**
    ```python
    from celery import Celery

    app = Celery('health_checker')

    @app.task
    def check_service_health(service_id: str):
        """Background job to check service health"""
        service = db.get_service(service_id)
        result = perform_health_check(service)
        store_result(result)

        # Schedule next check
        check_service_health.apply_async(
            args=[service_id],
            countdown=service.check_interval
        )
    ```

    **Benefits:**
    - Distributed across multiple workers
    - Automatic retries on worker failure
    - Load balancing

    ---

    ### 3. Status Aggregation

    **Service health states:**

    ```python
    from enum import Enum

    class HealthStatus(Enum):
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
        UNKNOWN = "unknown"

    def aggregate_status(check_results: List[Dict]) -> HealthStatus:
        """Aggregate multiple check results into overall status"""

        if not check_results:
            return HealthStatus.UNKNOWN

        recent_results = check_results[-5:]  # Last 5 checks

        healthy_count = sum(1 for r in recent_results if r["status"] == "healthy")
        total_count = len(recent_results)

        # 100% healthy -> HEALTHY
        if healthy_count == total_count:
            return HealthStatus.HEALTHY

        # 80%+ healthy -> DEGRADED
        elif healthy_count / total_count >= 0.8:
            return HealthStatus.DEGRADED

        # <80% healthy -> UNHEALTHY
        else:
            return HealthStatus.UNHEALTHY
    ```

    ---

    ### 4. Multi-Region Monitoring

    **Challenge:** Avoid false positives from regional network issues

    **Solution: Check from multiple regions**

    ```python
    class MultiRegionChecker:
        def __init__(self, regions: List[str]):
            self.regions = regions  # ["us-east-1", "eu-west-1", "ap-southeast-1"]

        def check_service(self, service: Service) -> Dict:
            """Check service from multiple regions"""

            results = []
            for region in self.regions:
                result = self.check_from_region(service, region)
                results.append(result)

            # Service is healthy if majority of regions report healthy
            healthy_count = sum(1 for r in results if r["status"] == "healthy")

            if healthy_count >= len(self.regions) / 2:
                return {"status": "healthy", "regions": results}
            else:
                return {"status": "unhealthy", "regions": results}
    ```

    **Benefits:**
    - Detect regional outages
    - Reduce false positives from network issues
    - Global view of service health

    ---

    ### 5. SLA Tracking

    ```python
    class SLATracker:
        def calculate_uptime(self, service_id: str, time_range: str) -> float:
            """Calculate uptime percentage for time range"""

            # Get all check results in time range
            results = db.query("""
                SELECT status, checked_at
                FROM health_check_results
                WHERE service_id = ? AND checked_at >= ?
                ORDER BY checked_at
            """, service_id, time_range)

            if not results:
                return 0.0

            # Calculate total uptime
            total_duration = 0
            uptime_duration = 0

            for i in range(len(results) - 1):
                duration = results[i+1].checked_at - results[i].checked_at

                total_duration += duration
                if results[i].status == "healthy":
                    uptime_duration += duration

            return (uptime_duration / total_duration) * 100

        def get_sla_status(self, uptime_percentage: float) -> str:
            """Determine SLA status"""
            if uptime_percentage >= 99.9:
                return "MEETING_SLA"  # 99.9% = 43.2 min downtime/month
            elif uptime_percentage >= 99.0:
                return "WARNING"
            else:
                return "VIOLATION"
    ```

=== "⚡ Step 4: Scale & Optimize"

    ## Scalability Improvements

    ### 1. Worker Auto-Scaling

    ```python
    class WorkerScaler:
        def scale_workers(self, queue_depth: int, current_workers: int) -> int:
            """Determine optimal number of workers"""

            # Target: Process queue in 60 seconds
            target_processing_time = 60
            checks_per_worker_per_minute = 120

            required_workers = math.ceil(
                queue_depth / (checks_per_worker_per_minute * target_processing_time / 60)
            )

            # Min 2 workers, max 20 workers
            return max(2, min(20, required_workers))
    ```

    ---

    ### 2. Caching Current Status

    ```python
    import redis

    class StatusCache:
        def __init__(self):
            self.redis = redis.Redis()

        def get_status(self, service_id: str) -> Optional[Dict]:
            """Get cached service status"""
            data = self.redis.get(f"status:{service_id}")
            if data:
                return json.loads(data)
            return None

        def set_status(self, service_id: str, status: Dict):
            """Cache service status with 5-minute expiry"""
            self.redis.setex(
                f"status:{service_id}",
                300,  # 5 minutes
                json.dumps(status)
            )
    ```

    **Benefits:**
    - Dashboard queries don't hit database
    - Sub-millisecond status lookups
    - Reduces database load

    ---

    ### 3. Monitoring Metrics

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Check Success Rate** | > 99% | < 95% |
    | **Check Duration (p99)** | < 1s | > 5s |
    | **Alert Response Time** | < 30s | > 60s |
    | **Worker Queue Depth** | < 100 | > 500 |
    | **False Positive Rate** | < 0.1% | > 1% |

    ---

    ### 4. Dashboard Optimization

    **Use WebSockets for real-time updates:**

    ```python
    from flask_socketio import SocketIO, emit

    socketio = SocketIO(app)

    @socketio.on('subscribe')
    def handle_subscribe(service_ids: List[str]):
        """Subscribe to service status updates"""
        for service_id in service_ids:
            join_room(f"service:{service_id}")

    def broadcast_status_change(service_id: str, status: Dict):
        """Broadcast status change to subscribed clients"""
        socketio.emit(
            'status_update',
            status,
            room=f"service:{service_id}"
        )
    ```

=== "📝 Summary & Tips"

    ## Key Design Decisions

    1. **Distributed workers** - Multiple checker instances for scalability
    2. **Job queue for scheduling** - Celery/RQ for distributed task execution
    3. **Alert debouncing** - Require N consecutive failures
    4. **Multi-region checks** - Avoid false positives
    5. **Redis for current state** - Fast status lookups
    6. **InfluxDB for time series** - Historical data and SLA tracking

    ## Interview Tips

    ✅ **Start with health check types** - HTTP, TCP, custom protocols
    ✅ **Discuss scheduling** - Fixed interval vs. job queue
    ✅ **Address false positives** - Debouncing and multi-region checks
    ✅ **Design alert system** - Channels, escalation, deduplication
    ✅ **Consider dependencies** - How one service failure affects others

    ## Common Follow-up Questions

    | Question | Key Points |
    |----------|------------|
    | **"How to avoid false positives?"** | Multi-region checks, require N consecutive failures, adjust check frequency |
    | **"What if health checker goes down?"** | Deploy in multiple regions, have fallback monitoring |
    | **"How to handle service dependencies?"** | Build dependency graph, cascade status (if DB down, API is unhealthy) |
    | **"How to scale to millions of checks?"** | Shard by service ID, use distributed workers, cache aggressively |
    | **"How to prevent alert fatigue?"** | Smart debouncing, escalation policies, alert grouping |

    ## Real-World Examples

    - **Pingdom**: Third-party uptime monitoring
    - **DataDog**: Infrastructure and application monitoring
    - **PagerDuty**: Incident management with on-call scheduling
    - **Prometheus + Alertmanager**: Self-hosted monitoring stack
    - **AWS CloudWatch**: Cloud-native monitoring

---

**Difficulty:** 🟢 Easy | **Interview Time:** 25-35 minutes | **Companies:** Companies with microservices

---

*This problem demonstrates distributed systems monitoring, alerting strategies, and SLA tracking - essential skills for maintaining production services.*
