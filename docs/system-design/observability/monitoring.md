# Monitoring

Monitoring is how you answer the question: "is the system healthy right now?" Without monitoring, you're flying blind — discovering problems only when users complain. With good monitoring, you can detect issues before users notice, understand trends for capacity planning, and diagnose problems quickly when they occur.

The core of monitoring is **metrics** — numerical measurements collected over time. CPU usage, request rate, error percentage, queue depth. These numbers, tracked continuously and displayed on dashboards, give you a real-time picture of system health.

---

=== "Golden Signals"

    ## The Four Golden Signals

    Google's Site Reliability Engineering book defines four signals that matter most for any user-facing system:

    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    Four Golden Signals                        │
    ├──────────────┬──────────────┬──────────────┬────────────────┤
    │   Latency    │   Traffic    │    Errors    │   Saturation   │
    │              │              │              │                │
    │ How long do  │ How much     │ How many     │ How "full" is  │
    │ requests     │ demand is    │ requests     │ the system?    │
    │ take?        │ on the       │ fail?        │                │
    │              │ system?      │              │                │
    │ P50: 50ms    │ 2,500 QPS    │ 0.1% error   │ CPU: 65%       │
    │ P95: 200ms   │              │ rate         │ Memory: 78%    │
    │ P99: 800ms   │              │              │ Disk: 45%      │
    └──────────────┴──────────────┴──────────────┴────────────────┘
    ```

    **Latency.** How long requests take to serve. Always track percentiles (P50, P95, P99), not averages — an average of 100ms hides the fact that 1% of users experience 5-second delays. Separate successful request latency from error latency, since errors often return fast (a 500 response takes milliseconds) and can skew the average downward.

    **Traffic.** The demand on your system — requests per second for a web service, transactions per second for a database, messages per second for a queue. This tells you how much load the system is under and whether traffic patterns are normal.

    **Errors.** The rate of failed requests. This includes explicit errors (HTTP 5xx), implicit errors (200 response with wrong content), and policy errors (responses slower than an SLA threshold). **Netflix** alerts when error rates exceed 0.01% of their millions-per-second request volume.

    **Saturation.** How close resources are to their limits. CPU at 90%, memory at 95%, disk 85% full. Saturation predicts future problems — a system at 95% memory utilization isn't broken yet, but it will be soon.

    ### RED and USE Methods

    Two complementary frameworks for deciding what to monitor:

    | Method | Focus | Metrics | Best For |
    |---|---|---|---|
    | **RED** | Services | Rate, Errors, Duration | APIs, web services, microservices |
    | **USE** | Resources | Utilization, Saturation, Errors | CPU, memory, disk, network |

    Use RED for your application services (what your users experience) and USE for your infrastructure (what your services run on). Together, they cover the full picture.

=== "Metrics & Architecture"

    ## Metric Types

    Monitoring systems support four fundamental metric types:

    | Type | Behavior | Use Case | Example |
    |---|---|---|---|
    | **Counter** | Only increases (resets on restart) | Count events | `http_requests_total`, `errors_total` |
    | **Gauge** | Goes up and down | Current state | `active_connections`, `queue_size` |
    | **Histogram** | Distribution in buckets | Latency, sizes | `request_duration_seconds` |
    | **Summary** | Pre-calculated percentiles | Latency | `job_duration_seconds` |

    **Counters** track cumulative totals. You query the *rate* of change: "how many requests per second?" not "how many requests total?"

    **Gauges** track current values. CPU usage, memory consumption, number of active connections — things that go up and down.

    **Histograms** track distributions. Instead of a single "average latency," they count how many requests fell into each bucket (0-10ms, 10-50ms, 50-100ms, etc.), enabling accurate percentile calculations.

    ---

    ## Monitoring Architecture

    ```
    Application Servers                   Monitoring Stack
    ┌─────────────────┐
    │ Service A        │──→ /metrics ──→ ┌──────────────┐
    │ (instrumented)   │                  │  Prometheus   │──→ ┌──────────┐
    └─────────────────┘                  │  (scrapes     │    │ Grafana  │
    ┌─────────────────┐                  │   every 15s)  │    │ (dashboards)
    │ Service B        │──→ /metrics ──→ │              │──→ └──────────┘
    │ (instrumented)   │                  │              │
    └─────────────────┘                  │              │──→ ┌──────────────┐
    ┌─────────────────┐                  │              │    │ AlertManager │
    │ Infrastructure   │──→ exporters ──→│              │    │ (notifications)
    │ (node, db, etc.) │                  └──────────────┘    └──────────────┘
    └─────────────────┘
    ```

    **Prometheus** is the dominant open-source monitoring system. It uses a **pull model** — scraping metrics from endpoints every 15-30 seconds. Each service exposes a `/metrics` endpoint. Prometheus stores time-series data locally and supports a powerful query language (PromQL). Used by **SoundCloud** (who created it), **DigitalOcean**, **Shopify**, and thousands of others.

    **Grafana** visualizes Prometheus data (and many other sources) in dashboards. It's the standard visualization layer for open-source monitoring stacks.

    **Cloud-native alternatives:** AWS CloudWatch, Google Cloud Monitoring, Azure Monitor. These integrate natively with their cloud platforms. **Datadog** and **New Relic** are popular SaaS platforms that combine metrics, logs, and traces in one place.

    ---

    ## Essential Dashboards

    Good dashboards answer specific questions. Don't create one massive dashboard — create focused views for different audiences and situations.

    **Service Overview (for on-call engineers):**
    - Request rate (QPS) — is traffic normal?
    - Error rate (%) — are we failing?
    - P50/P95/P99 latency — is performance acceptable?
    - Active connections — are we approaching limits?

    **Infrastructure (for capacity planning):**
    - CPU usage per node
    - Memory usage per node
    - Disk I/O and space
    - Network throughput

    **Database (for debugging slow queries):**
    - Query rate and slow query count
    - Connection pool utilization
    - Replication lag
    - Deadlock count

    **Business Metrics (for product teams):**
    - Orders per minute, revenue per hour
    - Active users, conversion rate
    - Feature usage, cart abandonment

    **Netflix** has thousands of dashboards across their microservices. **Uber** builds automated dashboards for every new service, populated from standardized metrics emitted by their service framework.

=== "SLIs, SLOs & Error Budgets"

    ## SLIs, SLOs, and Error Budgets

    These three concepts connect monitoring to business impact:

    ```
    SLI (Service Level Indicator):
      A measurement of service quality.
      Example: "99.2% of requests complete in <200ms"

    SLO (Service Level Objective):
      A target for an SLI.
      Example: "99.9% of requests should complete in <200ms"

    SLA (Service Level Agreement):
      An SLO with business consequences.
      Example: "99.9% uptime or customers get credits"

    Error Budget:
      The allowed amount of failure.
      SLO = 99.9% → Error budget = 0.1%
      In a 30-day month: 43.2 minutes of downtime allowed
      If 1M requests/month: 1,000 failures allowed
    ```

    **Error budgets** are a powerful concept from Google SRE. When the error budget is healthy, teams can ship features aggressively — even risky ones. When the error budget is nearly exhausted, teams freeze feature work and focus on reliability. This creates a natural, data-driven balance between velocity and stability.

    **Google** uses error budgets across all their services. **Spotify** adopted SLO-based monitoring to move from "is the system up?" to "is the system meeting user expectations?"

    ---

    ## What to Monitor

    ```
    Must Have:                          Nice to Have:
    ─────────                          ────────────
    Application:                       Business:
      ✓ Request rate                     ✓ Orders per minute
      ✓ Error rate                       ✓ Revenue per hour
      ✓ Latency (P50/P95/P99)           ✓ Active users
      ✓ Queue depth                      ✓ Conversion rate

    Infrastructure:                    Predictive:
      ✓ CPU usage                        ✓ Disk full prediction
      ✓ Memory usage                     ✓ Certificate expiry
      ✓ Disk I/O and space               ✓ Capacity forecasting
      ✓ Network throughput               ✓ Cost projections

    Database:
      ✓ Connection pool usage
      ✓ Query latency
      ✓ Replication lag
      ✓ Slow query count
    ```

    ### Metric Naming and Cardinality

    **Naming:** Use a consistent convention like `namespace_name_unit` — e.g., `http_request_duration_seconds`, `orders_created_total`. Include the unit in the name. Use labels for dimensions: `http_requests_total{method="GET", status="200"}` rather than `http_get_200_requests_total`.

    **Cardinality matters.** Every unique combination of label values creates a separate time series. `requests{user_id="..."}` with a million users creates a million time series — this will overwhelm Prometheus. Keep labels to bounded sets: HTTP methods (7 values), status codes (5 categories), service names (tens), not user IDs or request IDs (millions).

---

## Key Takeaways

1. **Monitor the four golden signals.** Latency, traffic, errors, and saturation cover the essential health of any service. Start here before adding more metrics.

2. **Track percentiles, not averages.** P95 and P99 latency reveal the experience of your worst-served users. An average of 50ms can hide a P99 of 5 seconds.

3. **Use SLOs to connect monitoring to business impact.** "99.9% of requests under 200ms" is meaningful to everyone. "CPU at 73%" is meaningful only to infrastructure teams.

4. **Error budgets balance velocity and reliability.** When the budget is healthy, ship fast. When it's depleted, focus on reliability. This removes the subjective "is it reliable enough?" debate.

5. **Watch cardinality.** High-cardinality labels (user IDs, request IDs) create millions of time series and can crash your monitoring system. Use bounded label values.

6. **Monitoring without alerting is a hobby; alerting without monitoring is noise.** They work together — see [Alerting](alerting.md).

---

## Related Topics

- **[Alerting](alerting.md)** — acting on monitoring data when things go wrong
- **[Logging](logging.md)** — discrete events that complement metric data
- **[Tracing](tracing.md)** — following individual requests across services
