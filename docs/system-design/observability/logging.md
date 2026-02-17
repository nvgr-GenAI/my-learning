# Logging

Metrics tell you *that* something is wrong. Logs tell you *why*. A monitoring dashboard shows the error rate spiked to 5% — but it's the logs that reveal "payment service returning 503 because Stripe API key expired." Logs capture discrete events with full context, making them essential for debugging, auditing, and understanding system behavior.

The challenge with logging isn't generating logs — every system generates plenty. The challenge is making them **useful**: structured so they're searchable, centralized so you can find them, and correlated so you can trace a problem across services.

---

=== "Structured Logging"

    ## Structured vs Unstructured Logging

    The difference between debugging in minutes and debugging in hours:

    ```
    Unstructured (hard to search, hard to parse):
      2024-01-15 10:30:45 ERROR Payment failed for user 12345, order 67890, amount $99.99

    Structured (machine-parseable, searchable, filterable):
      {
        "timestamp": "2024-01-15T10:30:45.123Z",
        "level": "ERROR",
        "message": "Payment failed",
        "service": "order-service",
        "userId": "12345",
        "orderId": "67890",
        "amount": 99.99,
        "error": "Stripe API key expired",
        "requestId": "abc-123-def"
      }
    ```

    With structured logs, you can query: "show all ERROR logs where service=order-service and amount>50 in the last hour." With unstructured logs, you're grep-ing through text files hoping your regex matches.

    **Every production system should use structured (JSON) logging.** All modern logging libraries support it: Winston (Node.js), Logrus/Zap (Go), structlog (Python), Serilog (.NET).

    ---

    ## Log Levels

    Log levels indicate severity. Getting them right prevents noise (too many INFO logs) and blind spots (missing ERROR logs).

    ```
    FATAL    System is unusable. Database gone. Out of memory. Application can't start.
      │      Action: Page on-call immediately.
      │
    ERROR    Something failed that shouldn't have. Payment declined. API returned 500.
      │      Action: Investigate. May need immediate fix.
      │
    WARN     Something concerning but not broken. Slow query. Retry needed. Deprecation.
      │      Action: Monitor. Fix in normal development cycle.
      │
    INFO     Normal business events. User logged in. Order created. Service started.
      │      Action: None. Reference during debugging.
      │
    DEBUG    Detailed diagnostic info. Variable values. Function entry/exit.
      │      Action: None. Typically disabled in production.
      │
    TRACE    Very detailed. Full request/response bodies. SQL with parameters.
             Action: None. Never enable in production (too verbose).
    ```

    **Production should run at INFO level.** Drop to DEBUG temporarily for specific services when investigating issues. ERROR and WARN should always be logged — if they're too noisy, fix the underlying issue rather than silencing the logs.

    ---

    ## What to Log (and What Not To)

    ```
    DO Log:                                     DON'T Log:
    ────────                                    ──────────
    ✓ HTTP requests (method, URL, status,       ✗ Passwords or password hashes
      duration, request ID)                     ✗ Credit card numbers
    ✓ Authentication events (login,             ✗ Social Security Numbers
      logout, failed attempts)                  ✗ API keys or tokens
    ✓ Business events (order created,           ✗ Personal health information
      payment processed, user registered)       ✗ Encryption keys
    ✓ Errors with full context and              ✗ Every database query (too verbose)
      stack traces                              ✗ User-generated content
    ✓ External API calls (URL, duration,          without sanitization
      status code)
    ✓ Background job start/completion
    ✓ Configuration changes
    ```

    The "don't log" list isn't optional — logging PII or credentials violates regulations (GDPR, HIPAA, PCI-DSS) and creates security vulnerabilities. If an attacker accesses your log storage, leaked credentials become a breach.

=== "Centralized Logging"

    ## Centralized Logging

    In a distributed system, logs from dozens of services are useless if they're scattered across individual servers. Centralized logging collects all logs into a single, searchable system.

    ```
    ┌──────────────┐
    │ Service A     │──→ stdout (JSON)
    └──────────────┘          │
    ┌──────────────┐          ▼
    │ Service B     │──→ ┌──────────┐     ┌────────────────┐     ┌──────────┐
    └──────────────┘    │ Log      │──→  │ Log Processing │──→  │ Search   │
    ┌──────────────┐    │ Shipper  │     │ (parse, enrich,│     │ & Store  │
    │ Service C     │──→ │ (Filebeat│     │  transform)    │     │ (Elastic │
    └──────────────┘    │  Fluentd)│     │ (Logstash)     │     │  search) │
                        └──────────┘     └────────────────┘     └────┬─────┘
                                                                      │
                                                                ┌─────▼─────┐
                                                                │ Kibana /  │
                                                                │ Grafana   │
                                                                │ (search,  │
                                                                │  visualize)│
                                                                └───────────┘
    ```

    ### The ELK Stack

    **Elasticsearch + Logstash + Kibana** is the most common open-source logging stack:

    - **Elasticsearch** stores and indexes logs for fast full-text search
    - **Logstash** (or **Fluentd**) processes and transforms logs — parsing JSON, adding metadata, routing by level
    - **Kibana** provides search UI and visualization dashboards
    - **Filebeat** ships logs from servers to the processing pipeline

    **Wikipedia**, **LinkedIn**, and **Netflix** all use Elasticsearch for log search. The stack handles millions of log events per second at scale.

    ### Cloud-Native Alternatives

    **AWS CloudWatch Logs** — native to AWS, integrates with Lambda and other services. CloudWatch Logs Insights provides a SQL-like query language for searching logs.

    **Google Cloud Logging** — automatic ingestion from GCP services, integrates with BigQuery for analysis.

    **Datadog Logs** — SaaS solution that unifies logs with metrics and traces. Automatic parsing of common log formats. **Airbnb** and **Samsung** use Datadog for centralized logging.

    **Splunk** — enterprise-focused, handles massive log volumes. Widely used in security operations (SIEM).

=== "Correlation & Operations"

    ## Log Correlation

    In a microservices architecture, a single user request might touch 5-10 services. Without correlation, you have 10 separate sets of logs with no way to connect them.

    ### Request IDs

    The simplest and most important correlation mechanism: generate a unique ID when a request enters your system, propagate it through every service, and include it in every log line.

    ```
    Request enters API Gateway:
      → generates requestId: "abc-123"
      → passes it to downstream services via X-Request-ID header

    Service A logs: {"requestId": "abc-123", "message": "Validating order"}
    Service B logs: {"requestId": "abc-123", "message": "Charging payment"}
    Service C logs: {"requestId": "abc-123", "message": "Sending confirmation"}

    Debugging: search for requestId="abc-123" → see entire request journey
    ```

    ### Trace ID Correlation

    When using distributed tracing (see [Tracing](tracing.md)), include the trace ID in log entries. This lets you jump from a log line directly to the full trace visualization, and vice versa.

    ```
    {
      "timestamp": "2024-01-15T10:30:45Z",
      "level": "ERROR",
      "message": "Payment failed",
      "requestId": "abc-123",
      "traceId": "4bf92f3577b34da6",     ← click to see full trace
      "spanId": "00f067aa0ba902b7",
      "service": "payment-service"
    }
    ```

    **Datadog**, **New Relic**, and **Grafana Cloud** all support clicking from a log entry to its trace — and this correlation is one of the most powerful debugging tools in a microservices environment.

    ---

    ## Log Retention and Storage

    Different environments need different retention policies:

    | Environment | Retention | Storage Tier | Rationale |
    |---|---|---|---|
    | **Production** | 30-90 days | Hot (Elasticsearch, S3) | Active debugging, compliance |
    | **Archived** | 1-7 years | Cold (S3 Glacier, GCS Coldline) | Legal, audit requirements |
    | **Staging** | 7-14 days | Standard | Testing, shorter lifecycle |
    | **Development** | 1-3 days | Local/ephemeral | Debugging only |

    **Cost optimization:** Logs are expensive to store and index. Move logs older than 30 days to cold storage. Downsample verbose logs (keep 10% of DEBUG logs). Drop known-noisy log patterns. **Uber** processes over 100TB of logs per day — aggressive retention policies and sampling are essential at that scale.

    ---

    ## Performance Considerations

    Logging should never be the bottleneck. Key principles:

    **Write to stdout, not files.** Let the container orchestrator (Kubernetes) or log shipper handle collection. Writing to files adds I/O overhead and requires log rotation management.

    **Buffer and batch.** Don't send each log line individually to the centralized system. Buffer locally and send in batches — typically every 1-5 seconds or when the buffer reaches a size threshold.

    **Sample verbose logs.** Always log errors and warnings. For DEBUG and INFO logs at high volume, consider sampling — log 10% of routine requests but 100% of slow or failed requests.

    **Async logging.** Log emission should never block request processing. Use async transports that queue log entries and write them in the background.

---

## Key Takeaways

1. **Always use structured logging.** JSON-formatted logs with consistent field names are the foundation of searchable, analyzable logging. No exceptions in production.

2. **Centralize everything.** Logs scattered across servers are nearly useless. Ship all logs to a central system (ELK, CloudWatch, Datadog) where they can be searched and correlated.

3. **Correlate with request IDs.** Generate a unique ID at the edge and propagate it through every service. This single practice transforms debugging in distributed systems.

4. **Never log secrets.** Passwords, API keys, tokens, credit card numbers, and PII must never appear in logs. Sanitize or mask before logging.

5. **Log at the right level.** INFO for business events, ERROR for failures, WARN for degradation. If your logs are too noisy, the problem is the log level assignment, not logging itself.

6. **Mind the cost.** Logs at scale are expensive. Set retention policies, archive to cold storage, and sample verbose log categories.

---

## Related Topics

- **[Monitoring](monitoring.md)** — metrics complement logs for system health
- **[Tracing](tracing.md)** — correlate logs with distributed traces
- **[Alerting](alerting.md)** — trigger alerts from log patterns
