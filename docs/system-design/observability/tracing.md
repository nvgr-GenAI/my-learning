# Distributed Tracing

Metrics tell you *something* is slow. Logs tell you *what* happened in one service. Tracing tells you *where* time was spent across an entire request journey. In a microservices architecture where a single API call might touch 10 services, tracing is the only way to answer: "why did this request take 3 seconds?"

Distributed tracing follows a request from the moment it enters your system to the moment a response is returned, recording the time spent in each service, database call, and external API interaction along the way.

---

=== "Core Concepts"

    ## Core Concepts: Traces and Spans

    A **trace** represents the complete journey of a single request. A **span** represents one operation within that journey — an HTTP call, a database query, a cache lookup. Spans are nested to show causality: this service called that service, which queried that database.

    ```
    Trace ID: abc-123-def (one per request)
    │
    ├─ Span: GET /api/orders (total: 350ms) ────── [API Gateway]
    │  │
    │  ├─ Span: Auth check (15ms) ──────────────── [Auth Service]
    │  │
    │  ├─ Span: Get user (45ms) ────────────────── [User Service]
    │  │  └─ Span: PostgreSQL query (40ms) ──────── [Database]
    │  │
    │  └─ Span: Create order (280ms) ───────────── [Order Service]
    │     │
    │     ├─ Span: Process payment (220ms) ─────── [Payment Service]
    │     │  └─ Span: Stripe API call (200ms) ──── [External API]
    │     │
    │     └─ Span: Send confirmation (15ms) ────── [Email Service]
    ```

    Each span records:
    - **Span ID** — unique identifier
    - **Parent Span ID** — which span triggered this one
    - **Start time and duration**
    - **Service name and operation**
    - **Tags/attributes** — metadata like `http.status_code`, `db.statement`, `user.id`
    - **Events** — timestamped annotations within the span

    ---

    ## How Context Propagates

    For tracing to work across services, each service must pass trace context to the next. This happens through HTTP headers using the **W3C Trace Context** standard:

    ```
    Service A                           Service B                         Service C
        │                                   │                                 │
        │ traceparent: 00-{traceId}-{spanA}-01                               │
        │──────────────────────────────────→│                                 │
        │                                   │                                 │
        │                                   │ traceparent: 00-{traceId}-{spanB}-01
        │                                   │────────────────────────────────→│
        │                                   │                                 │
        │                      All spans share the same traceId               │
        │                      → they form a single trace                     │
    ```

    The trace ID stays the same across all services. Each service creates a new span ID and links it to its parent. The tracing backend collects all spans with the same trace ID and assembles them into the full trace tree.

=== "OpenTelemetry"

    ## OpenTelemetry

    **OpenTelemetry** (OTel) is the industry standard for instrumenting applications. It provides vendor-neutral APIs and SDKs for emitting traces (and metrics and logs) that can be exported to any backend — Jaeger, Zipkin, Datadog, New Relic, or Grafana Tempo.

    ### How Instrumentation Works

    There are two types:

    **Auto-instrumentation** — OTel libraries automatically trace HTTP calls, database queries, and framework operations with zero code changes. You add the SDK, configure an exporter, and traces appear. This covers 80% of what you need.

    **Manual instrumentation** — For custom business logic, you create spans explicitly:

    ```
    Start a span "process-order"
      Set attributes: order.id, user.id, amount
      │
      Start child span "validate-inventory"
        Query database, record result
      End child span
      │
      Start child span "charge-payment"
        Call Stripe API, record transaction ID
      End child span
      │
      If error: record exception on span
    End span
    ```

    ### Tracing Backends

    | Backend | Type | Strengths | Used By |
    |---|---|---|---|
    | **Jaeger** | Open-source | Feature-rich, strong Kubernetes integration | Uber (created it), Red Hat |
    | **Zipkin** | Open-source | Simple, lightweight, well-established | Twitter (created it) |
    | **Grafana Tempo** | Open-source | Cost-effective storage, Grafana integration | Grafana Labs ecosystem |
    | **Datadog APM** | SaaS | Unified metrics/logs/traces, ML-powered analysis | Airbnb, Samsung, Peloton |
    | **AWS X-Ray** | Cloud-native | Native AWS integration | AWS workloads |

    **Uber** built Jaeger to handle their massive microservices architecture — hundreds of services processing millions of traces per day. They open-sourced it, and it became a CNCF graduated project.

=== "Sampling & Debugging"

    ## Sampling: You Can't Trace Everything

    At high traffic volumes, tracing every request generates enormous amounts of data and adds measurable overhead. Sampling decides which requests get traced.

    ### Sampling Strategies

    ```
    Head-Based Sampling (decide at the start):
      Request arrives → roll the dice → trace or don't
      Simple but you might miss interesting traces

      10% sampling: capture 1 in 10 requests
      Always sample: errors, slow requests, specific users

    Tail-Based Sampling (decide at the end):
      Trace everything → after all spans arrive → keep interesting ones
      Better quality but requires a collector that buffers traces

      Keep: errors, latency > 1s, specific operations
      Discard: normal, fast, successful traces
    ```

    | Strategy | Overhead | Data Quality | Complexity |
    |---|---|---|---|
    | **Always sample** (100%) | High | Perfect | Low |
    | **Probabilistic** (10%) | Low | Good (statistical) | Low |
    | **Rate-limited** (100/sec) | Bounded | Good | Low |
    | **Tail-based** (keep interesting) | Medium | Excellent | High |

    **Recommended approach for production:** Use head-based probabilistic sampling (1-10% of traffic) as the default, but always sample errors and slow requests regardless. This captures enough data for statistical analysis while ensuring you never miss problems.

    **Google** traces a tiny fraction of their overall traffic but uses deterministic sampling to ensure end-to-end traces are complete (if a request is sampled in Service A, all downstream services also sample it).

    ---

    ## Finding Problems with Traces

    Tracing is most valuable for three debugging scenarios:

    ### Latency Investigation

    ```
    Slow trace: GET /api/dashboard (3.5s)
    ├─ Auth check (50ms) .................. OK
    ├─ Get user (100ms) ................... OK
    └─ Get dashboard data (3.3s) .......... SLOW
       ├─ Get posts (2.8s) ................ VERY SLOW
       │  ├─ DB query posts (100ms) ....... OK
       │  └─ Get authors (2.7s) ........... N+1 PROBLEM
       │     └─ DB query author (50ms) × 50 ← 50 individual queries!
       └─ Get notifications (500ms) ....... Moderate

    Root cause: N+1 query — fetching 50 authors individually
    Fix: Batch query or JOIN
    Expected improvement: 3.5s → 600ms
    ```

    ### Error Investigation

    Filter traces by error status, find the first failing span, check its error message and attributes. Then search logs using the trace ID for full context.

    ### Service Dependency Mapping

    Aggregate traces to build a dependency graph — which services call which other services, how often, and how fast. This reveals unexpected dependencies, single points of failure, and services that are called too frequently.

    ---

    ## Performance Impact

    Tracing adds overhead. Minimizing it requires intentional design:

    - **Async span processing.** Never block request handling to export spans. Buffer spans in memory and export in background batches.
    - **Batch exports.** Send spans to the collector in batches (every 500ms or every 100 spans), not individually.
    - **Sample aggressively.** In high-traffic production, 1-5% sampling captures enough data with minimal impact.
    - **Watch attribute size.** Large attributes (request bodies, SQL queries) increase memory usage and export time.

    With proper configuration (async processing, batching, reasonable sampling), tracing overhead is typically **under 1% of request latency** — well worth the debugging capability it provides.

    ---

    ## Service Mesh Integration

    Service meshes like **Istio** (using Envoy sidecars) can add tracing automatically to all inter-service traffic without any application code changes. The sidecar proxy creates spans for every request it handles, providing baseline visibility across the entire mesh.

    However, application-level instrumentation is still valuable for tracing operations within a service — database queries, cache lookups, and business logic. The best approach combines mesh-level tracing (automatic, covers all traffic) with application-level instrumentation (manual, covers internal operations).

---

## Key Takeaways

1. **Tracing answers "where is the time spent?"** In a distributed system, this question is impossible to answer with metrics or logs alone. Tracing shows the complete request path with timing.

2. **Use OpenTelemetry.** It's the vendor-neutral standard. Instrument once, export to any backend. Auto-instrumentation covers most needs; add manual spans for custom business logic.

3. **Sample strategically in production.** 1-10% head-based sampling for normal traffic, 100% for errors and slow requests. Tail-based sampling for the best data quality if you can afford the infrastructure.

4. **Propagate context through headers.** The W3C Trace Context standard (`traceparent` header) ensures all services contribute to the same trace. Without propagation, you get disconnected fragments.

5. **Correlate traces with logs.** Include `traceId` in every log entry. This lets you jump from a log line to the full trace visualization — the fastest path to root cause in distributed systems.

---

## Related Topics

- **[Monitoring](monitoring.md)** — metrics identify problems; traces pinpoint their location
- **[Logging](logging.md)** — logs provide detail within what traces reveal
- **[Alerting](alerting.md)** — alert on trace-derived metrics (P99 latency, error rates)
