# Monitoring

**Know what's happening in your system** | 📊 Metrics | 📈 Dashboards | 🔔 Alerts

---

## Overview

Monitoring provides visibility into system health through metrics collection, visualization, and alerting. Effective monitoring helps detect issues before they impact users.

**Why Monitor?**

- Early problem detection
- Performance optimization
- Capacity planning
- SLA compliance
- Root cause analysis

---

## The Four Golden Signals

=== "Overview"
    ```
    Google's Four Golden Signals:

    1. Latency
       - Time to serve requests
       - Distinguish success vs error latency
       - Track P50, P95, P99

    2. Traffic
       - Demand on the system
       - Requests per second
       - Transactions per second

    3. Errors
       - Rate of failed requests
       - Explicit (500s, 400s)
       - Implicit (wrong content)

    4. Saturation
       - How "full" the system is
       - CPU, memory, disk, network
       - Queue depth, thread pool usage
    ```

=== "RED Method"
    ```
    RED Method (for services):

    Rate     - Requests per second
    Errors   - Failed requests per second
    Duration - Time per request (latency)

    Best for: Request-driven services (APIs, web apps)
    ```

=== "USE Method"
    ```
    USE Method (for resources):

    Utilization - % time resource is busy
    Saturation  - Amount of queued work
    Errors      - Error count

    Best for: Infrastructure (CPU, disk, network)
    ```

---

## Prometheus

=== "Setup"
    ```yaml
    # prometheus.yml
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'production'
        region: 'us-east-1'

    # Alertmanager configuration
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
                - alertmanager:9093

    # Load rules
    rule_files:
      - "alerts/*.yml"
      - "recording_rules/*.yml"

    # Scrape configurations
    scrape_configs:
      # Application metrics
      - job_name: 'myapp'
        static_configs:
          - targets: ['localhost:3000']
            labels:
              environment: 'production'
              service: 'api'

      # Node exporter (system metrics)
      - job_name: 'node'
        static_configs:
          - targets: ['localhost:9100']

      # Kubernetes pods
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
    ```

=== "Instrumentation"
    ```javascript
    // Node.js with prom-client
    const express = require('express');
    const client = require('prom-client');

    const app = express();

    // Create a Registry
    const register = new client.Registry();

    // Add default metrics
    client.collectDefaultMetrics({ register });

    // Custom metrics
    const httpRequestDuration = new client.Histogram({
      name: 'http_request_duration_seconds',
      help: 'Duration of HTTP requests in seconds',
      labelNames: ['method', 'route', 'status_code'],
      buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    });

    const httpRequestTotal = new client.Counter({
      name: 'http_requests_total',
      help: 'Total number of HTTP requests',
      labelNames: ['method', 'route', 'status_code']
    });

    const activeConnections = new client.Gauge({
      name: 'active_connections',
      help: 'Number of active connections'
    });

    const jobDuration = new client.Summary({
      name: 'job_duration_seconds',
      help: 'Duration of background jobs',
      labelNames: ['job_name'],
      percentiles: [0.5, 0.9, 0.95, 0.99]
    });

    // Register metrics
    register.registerMetric(httpRequestDuration);
    register.registerMetric(httpRequestTotal);
    register.registerMetric(activeConnections);
    register.registerMetric(jobDuration);

    // Middleware to track metrics
    app.use((req, res, next) => {
      const start = Date.now();

      activeConnections.inc();

      res.on('finish', () => {
        const duration = (Date.now() - start) / 1000;
        const route = req.route ? req.route.path : req.path;

        httpRequestDuration
          .labels(req.method, route, res.statusCode)
          .observe(duration);

        httpRequestTotal
          .labels(req.method, route, res.statusCode)
          .inc();

        activeConnections.dec();
      });

      next();
    });

    // Business metrics
    const orderMetrics = {
      created: new client.Counter({
        name: 'orders_created_total',
        help: 'Total orders created',
        labelNames: ['payment_method']
      }),
      value: new client.Histogram({
        name: 'order_value_dollars',
        help: 'Order value in dollars',
        buckets: [10, 50, 100, 500, 1000, 5000]
      })
    };

    register.registerMetric(orderMetrics.created);
    register.registerMetric(orderMetrics.value);

    // Business logic with metrics
    app.post('/orders', async (req, res) => {
      const order = await createOrder(req.body);

      orderMetrics.created
        .labels(order.paymentMethod)
        .inc();

      orderMetrics.value.observe(order.totalAmount);

      res.json(order);
    });

    // Expose metrics endpoint
    app.get('/metrics', async (req, res) => {
      res.set('Content-Type', register.contentType);
      res.end(await register.metrics());
    });

    app.listen(3000);
    ```

=== "PromQL Queries"
    ```promql
    # Request rate (requests per second)
    rate(http_requests_total[5m])

    # Success rate
    sum(rate(http_requests_total{status_code!~"5.."}[5m])) /
    sum(rate(http_requests_total[5m]))

    # Error rate
    sum(rate(http_requests_total{status_code=~"5.."}[5m])) /
    sum(rate(http_requests_total[5m]))

    # Average latency
    rate(http_request_duration_seconds_sum[5m]) /
    rate(http_request_duration_seconds_count[5m])

    # P95 latency
    histogram_quantile(0.95,
      rate(http_request_duration_seconds_bucket[5m])
    )

    # P99 latency
    histogram_quantile(0.99,
      rate(http_request_duration_seconds_bucket[5m])
    )

    # CPU usage
    100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

    # Memory usage percentage
    (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) /
    node_memory_MemTotal_bytes * 100

    # Disk usage
    100 - ((node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)

    # Top 5 slowest endpoints
    topk(5,
      histogram_quantile(0.99,
        rate(http_request_duration_seconds_bucket[5m])
      )
    )

    # Requests by status code
    sum by (status_code) (rate(http_requests_total[5m]))

    # Compare this week vs last week
    rate(http_requests_total[5m]) /
    rate(http_requests_total[5m] offset 1w)

    # Predict disk full time (linear regression)
    predict_linear(node_filesystem_free_bytes[1h], 4 * 3600) < 0
    ```

---

## Grafana Dashboards

=== "Dashboard JSON"
    ```json
    {
      "dashboard": {
        "title": "Application Performance",
        "panels": [
          {
            "id": 1,
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "sum(rate(http_requests_total[5m])) by (service)",
                "legendFormat": "{{service}}"
              }
            ],
            "yAxis": {
              "label": "Requests/sec"
            }
          },
          {
            "id": 2,
            "title": "Error Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "sum(rate(http_requests_total{status_code=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100"
              }
            ],
            "alert": {
              "conditions": [
                {
                  "evaluator": {
                    "type": "gt",
                    "params": [1]
                  }
                }
              ]
            }
          },
          {
            "id": 3,
            "title": "P95 Latency",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
              }
            ]
          },
          {
            "id": 4,
            "title": "Active Connections",
            "type": "stat",
            "targets": [
              {
                "expr": "active_connections"
              }
            ]
          }
        ]
      }
    }
    ```

=== "Key Dashboards"
    ```
    Essential Dashboards:

    1. Service Overview
       - Request rate (QPS)
       - Error rate (%)
       - P50/P95/P99 latency
       - Active connections

    2. Infrastructure
       - CPU usage per node
       - Memory usage per node
       - Disk I/O
       - Network throughput

    3. Database
       - Query rate
       - Slow queries
       - Connection pool
       - Replication lag

    4. Business Metrics
       - Orders per minute
       - Revenue per hour
       - Active users
       - Conversion rate

    5. Kubernetes
       - Pod status
       - Container restarts
       - Resource requests vs usage
       - HPA scaling events
    ```

---

## CloudWatch

=== "Custom Metrics"
    ```javascript
    // AWS SDK for JavaScript
    const AWS = require('aws-sdk');
    const cloudwatch = new AWS.CloudWatch();

    // Put metric data
    async function publishMetric(metricName, value, unit = 'Count') {
      const params = {
        Namespace: 'MyApp',
        MetricData: [
          {
            MetricName: metricName,
            Value: value,
            Unit: unit,
            Timestamp: new Date(),
            Dimensions: [
              {
                Name: 'Environment',
                Value: 'production'
              },
              {
                Name: 'Service',
                Value: 'api'
              }
            ]
          }
        ]
      };

      await cloudwatch.putMetricData(params).promise();
    }

    // Batch metrics
    async function publishBatchMetrics(metrics) {
      const params = {
        Namespace: 'MyApp',
        MetricData: metrics.map(m => ({
          MetricName: m.name,
          Value: m.value,
          Unit: m.unit || 'Count',
          Timestamp: new Date(),
          Dimensions: m.dimensions || []
        }))
      };

      await cloudwatch.putMetricData(params).promise();
    }

    // Usage in application
    app.post('/orders', async (req, res) => {
      try {
        const order = await createOrder(req.body);

        await publishBatchMetrics([
          { name: 'OrdersCreated', value: 1 },
          { name: 'OrderValue', value: order.amount, unit: 'None' },
          { name: 'OrderLatency', value: Date.now() - startTime, unit: 'Milliseconds' }
        ]);

        res.json(order);
      } catch (error) {
        await publishMetric('OrderErrors', 1);
        throw error;
      }
    });
    ```

=== "CloudWatch Alarms"
    ```javascript
    // Create alarm
    const alarmParams = {
      AlarmName: 'HighErrorRate',
      ComparisonOperator: 'GreaterThanThreshold',
      EvaluationPeriods: 2,
      MetricName: 'ErrorRate',
      Namespace: 'MyApp',
      Period: 300,
      Statistic: 'Average',
      Threshold: 5.0,
      ActionsEnabled: true,
      AlarmActions: [
        'arn:aws:sns:us-east-1:123456789012:alerts'
      ],
      AlarmDescription: 'Alert when error rate exceeds 5%',
      Dimensions: [
        {
          Name: 'Environment',
          Value: 'production'
        }
      ]
    };

    await cloudwatch.putMetricAlarm(alarmParams).promise();
    ```

---

## DataDog

=== "Setup"
    ```javascript
    const StatsD = require('node-dogstatsd').StatsD;

    const dogstatsd = new StatsD('localhost', 8125);

    // Counter
    dogstatsd.increment('page.views', 1, ['page:home']);

    // Gauge (current value)
    dogstatsd.gauge('queue.size', queue.length);

    // Histogram (statistical distribution)
    dogstatsd.histogram('file.upload.size', fileSize, ['type:image']);

    // Timing
    const start = Date.now();
    await processRequest();
    dogstatsd.timing('request.duration', Date.now() - start);

    // Set (count unique values)
    dogstatsd.set('unique.users', userId);

    // Service checks
    dogstatsd.check('database.up', dogstatsd.OK);
    dogstatsd.check('api.health', dogstatsd.CRITICAL, {
      message: 'API is down'
    });
    ```

=== "APM Integration"
    ```javascript
    // Automatic instrumentation
    const tracer = require('dd-trace').init({
      service: 'my-api',
      env: 'production',
      version: '1.2.3',
      logInjection: true
    });

    const express = require('express');
    const app = express();

    // Automatic tracing of HTTP requests
    app.get('/api/users/:id', async (req, res) => {
      // Spans are created automatically
      const user = await db.users.findById(req.params.id);
      res.json(user);
    });

    // Custom spans
    app.post('/api/orders', async (req, res) => {
      const span = tracer.startSpan('create.order');

      try {
        span.setTag('order.amount', req.body.amount);
        span.setTag('payment.method', req.body.paymentMethod);

        const order = await createOrder(req.body);

        span.setTag('order.id', order.id);
        res.json(order);
      } catch (error) {
        span.setTag('error', true);
        span.log({ event: 'error', message: error.message });
        throw error;
      } finally {
        span.finish();
      }
    });
    ```

---

## Metric Types Comparison

| Type | Description | Use Case | Example | Aggregation |
|------|-------------|----------|---------|-------------|
| **Counter** | Cumulative value that only increases | Count events | `requests_total`, `orders_created` | `rate()`, `increase()` |
| **Gauge** | Value that can go up or down | Current state | `active_connections`, `queue_size` | `avg()`, `min()`, `max()` |
| **Histogram** | Distribution of values in buckets | Latency, sizes | `request_duration_seconds` | `histogram_quantile()` |
| **Summary** | Similar to histogram, calculates quantiles | Latency | `job_duration_seconds` | Pre-calculated percentiles |

---

## Best Practices

=== "What to Monitor"
    ```
    Application Metrics:
    ✅ Request rate
    ✅ Error rate
    ✅ Latency (P50, P95, P99)
    ✅ Throughput
    ✅ Queue depth
    ✅ Circuit breaker state

    Infrastructure Metrics:
    ✅ CPU usage
    ✅ Memory usage
    ✅ Disk I/O
    ✅ Network I/O
    ✅ File descriptors

    Business Metrics:
    ✅ Orders per minute
    ✅ Revenue
    ✅ Active users
    ✅ Conversion rate
    ✅ Cart abandonment

    Database Metrics:
    ✅ Connection pool usage
    ✅ Query latency
    ✅ Slow queries
    ✅ Replication lag
    ✅ Deadlocks
    ```

=== "Metric Naming"
    ```
    Good Naming Conventions:

    Format: <namespace>_<name>_<unit>

    ✅ http_requests_total
    ✅ http_request_duration_seconds
    ✅ database_connections_active
    ✅ orders_created_total
    ✅ payment_amount_dollars

    ❌ httpRequests (inconsistent)
    ❌ req_time (unclear unit)
    ❌ errors (too vague)
    ❌ db_conn (abbreviations)

    Labels vs Metric Names:
    ✅ http_requests_total{method="GET", status="200"}
    ❌ http_get_200_requests_total
    ```

=== "Performance"
    ```
    Optimization:

    1. Cardinality
       ✅ Low cardinality labels: method, status, endpoint
       ❌ High cardinality labels: user_id, request_id, timestamp

       # Bad: creates millions of time series
       requests_total{user_id="123456"}

       # Good: bounded set of values
       requests_total{method="GET", status="200"}

    2. Sampling
       # Sample expensive metrics
       if (Math.random() < 0.01) {
         recordDetailedMetrics();
       }

    3. Aggregation
       # Pre-aggregate in application
       const counters = new Map();
       setInterval(() => {
         for (const [key, value] of counters) {
           publishMetric(key, value);
         }
         counters.clear();
       }, 60000);
    ```

---

## SLIs and SLOs

=== "Definition"
    ```
    SLI (Service Level Indicator):
    - Quantitative measure of service level
    - Examples: latency, error rate, availability

    SLO (Service Level Objective):
    - Target value for an SLI
    - Example: "99.9% of requests succeed"

    SLA (Service Level Agreement):
    - Contract with consequences
    - Example: "99.9% uptime or money back"
    ```

=== "Implementation"
    ```yaml
    # Recording rules for SLIs
    groups:
      - name: sli_rules
        interval: 30s
        rules:
          # Request success rate
          - record: sli:http_requests:success_rate
            expr: |
              sum(rate(http_requests_total{status!~"5.."}[5m])) /
              sum(rate(http_requests_total[5m]))

          # Fast requests (< 200ms)
          - record: sli:http_requests:fast_rate
            expr: |
              sum(rate(http_request_duration_seconds_bucket{le="0.2"}[5m])) /
              sum(rate(http_request_duration_seconds_count[5m]))

          # Availability
          - record: sli:service:availability
            expr: up{job="myapp"}

      - name: slo_alerts
        rules:
          # Alert if burning through error budget too fast
          - alert: HighErrorBudgetBurn
            expr: |
              (
                sli:http_requests:success_rate < 0.999
                and
                sli:http_requests:success_rate < 0.995 offset 1h
              )
            labels:
              severity: critical
            annotations:
              summary: "Burning through error budget rapidly"
    ```

=== "Error Budget"
    ```
    Error Budget Calculation:

    SLO: 99.9% success rate
    Error budget: 100% - 99.9% = 0.1%

    Monthly budget:
    - 30 days = 2,592,000 seconds
    - 0.1% = 2,592 seconds of downtime allowed
    - = 43.2 minutes per month

    If 1M requests per month:
    - Can fail 1,000 requests
    - After that, stop risky deploys

    Usage:
    if (errorBudgetRemaining < 0.1) {
      // Freeze deployments
      // Focus on reliability
    } else {
      // Deploy new features
      // Take calculated risks
    }
    ```

---

## Interview Talking Points

**Q: How do you monitor microservices?**

✅ **Strong Answer:**
> "I'd implement the four golden signals: latency, traffic, errors, and saturation. For metrics, I'd use Prometheus with service discovery for auto-detection of new instances. Each service exposes a `/metrics` endpoint with RED metrics: request rate, error rate, and duration. I'd create Grafana dashboards showing service health and use distributed tracing with Jaeger or DataDog to track requests across services. For alerting, I'd set up SLO-based alerts in AlertManager based on error budgets rather than arbitrary thresholds. I'd also implement health check endpoints that aggregate dependencies so orchestrators like Kubernetes can make intelligent routing decisions."

**Q: What's the difference between metrics, logs, and traces?**

✅ **Strong Answer:**
> "They're complementary observability tools. Metrics are numerical measurements over time - like request rate or CPU usage - great for dashboards and alerting but lack context. Logs are discrete events with full context - like 'user 123 logged in' - useful for debugging specific issues but hard to aggregate. Traces show the path of a single request through distributed systems - revealing where time is spent across services. In practice, I'd use metrics for real-time monitoring and alerting, logs for root cause analysis, and traces to understand interactions between services. Modern tools like DataDog and New Relic unify all three for correlation."

---

## Related Topics

- [Alerting](alerting.md) - Set up intelligent alerts
- [Logging](logging.md) - Structured logging practices
- [Tracing](tracing.md) - Distributed tracing
- [Deployment](../deployment/ci-cd.md) - Monitor deployments

---

**Monitor everything, alert intelligently! 📊**
