# Distributed Tracing

**Track requests across services** | 🔍 Jaeger | 🎯 Zipkin | 📍 OpenTelemetry

---

## Overview

Distributed tracing tracks a request's journey through multiple services, helping identify bottlenecks, latency issues, and service dependencies in microservices architectures.

**Why Tracing?**

- Visualize request flow across services
- Identify performance bottlenecks
- Debug failures in distributed systems
- Understand service dependencies
- Measure end-to-end latency

---

## Core Concepts

=== "Traces and Spans"
    ```
    Trace: Complete journey of a request

    Request flow in microservices:
    API Gateway → Auth Service → User Service → Database
              ↓
           Order Service → Payment Service → Email Service

    Trace Structure:
    Trace ID: abc123def456 (unique per request)
    │
    ├─ Span: GET /api/orders (100ms) ────── [Root Span]
    │  │
    │  ├─ Span: Auth check (15ms) ───────── [Auth Service]
    │  │
    │  ├─ Span: Get user (25ms) ─────────── [User Service]
    │  │  └─ Span: DB query (20ms) ──────── [Database]
    │  │
    │  └─ Span: Create order (60ms) ────── [Order Service]
    │     │
    │     ├─ Span: Process payment (45ms) ─ [Payment Service]
    │     │  └─ Span: Stripe API (40ms) ─── [External API]
    │     │
    │     └─ Span: Send email (10ms) ────── [Email Service]

    Each Span contains:
    - Span ID (unique identifier)
    - Parent Span ID (for hierarchy)
    - Start timestamp
    - Duration
    - Service name
    - Operation name
    - Tags (metadata)
    - Logs (events within span)
    ```

=== "Trace Context"
    ```
    W3C Trace Context Standard:

    traceparent: 00-{trace-id}-{span-id}-{trace-flags}
    tracestate: vendor-specific data

    Example:
    traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
                 │   │                                │                │
                 │   └─ Trace ID (128-bit)            └─ Span ID       └─ Sampled
                 └─ Version

    Propagation via HTTP headers:
    Client: Adds traceparent header
    Service A: Extracts trace-id, creates child span
    Service B: Extracts trace-id, creates child span
    All spans share same trace-id for correlation
    ```

---

## OpenTelemetry

=== "Node.js Setup"
    ```javascript
    const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
    const { registerInstrumentations } = require('@opentelemetry/instrumentation');
    const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
    const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
    const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
    const { BatchSpanProcessor } = require('@opentelemetry/sdk-trace-base');
    const { Resource } = require('@opentelemetry/resources');
    const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');

    // Create provider with service information
    const provider = new NodeTracerProvider({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: 'api-service',
        [SemanticResourceAttributes.SERVICE_VERSION]: '1.2.3',
        [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: 'production'
      })
    });

    // Configure Jaeger exporter
    const jaegerExporter = new JaegerExporter({
      endpoint: 'http://localhost:14268/api/traces',
      tags: [
        { key: 'environment', value: process.env.NODE_ENV }
      ]
    });

    // Add span processor
    provider.addSpanProcessor(new BatchSpanProcessor(jaegerExporter, {
      maxQueueSize: 100,
      maxExportBatchSize: 10,
      scheduledDelayMillis: 500
    }));

    // Register provider
    provider.register();

    // Auto-instrument HTTP and Express
    registerInstrumentations({
      instrumentations: [
        new HttpInstrumentation(),
        new ExpressInstrumentation()
      ]
    });

    // Manual instrumentation
    const { trace, SpanStatusCode } = require('@opentelemetry/api');
    const express = require('express');
    const app = express();

    app.get('/api/users/:id', async (req, res) => {
      const tracer = trace.getTracer('api-service');

      // Create a span
      const span = tracer.startSpan('get-user', {
        attributes: {
          'http.method': req.method,
          'http.url': req.url,
          'user.id': req.params.id
        }
      });

      try {
        // Simulate database call
        const dbSpan = tracer.startSpan('db-query', {
          parent: span,
          attributes: {
            'db.system': 'postgresql',
            'db.statement': 'SELECT * FROM users WHERE id = $1'
          }
        });

        const user = await db.users.findById(req.params.id);
        dbSpan.setStatus({ code: SpanStatusCode.OK });
        dbSpan.end();

        // Add event to span
        span.addEvent('user-fetched', {
          'user.email': user.email
        });

        span.setStatus({ code: SpanStatusCode.OK });
        res.json(user);
      } catch (error) {
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: error.message
        });
        span.recordException(error);
        res.status(500).json({ error: error.message });
      } finally {
        span.end();
      }
    });

    // Trace async operations
    async function processOrder(orderId) {
      const tracer = trace.getTracer('order-service');

      return tracer.startActiveSpan('process-order', async (span) => {
        try {
          span.setAttribute('order.id', orderId);

          // Payment processing
          await tracer.startActiveSpan('payment', async (paymentSpan) => {
            paymentSpan.setAttribute('payment.method', 'stripe');
            await stripe.charge(order.amount);
            paymentSpan.end();
          });

          // Send notification
          await tracer.startActiveSpan('notification', async (notifSpan) => {
            notifSpan.setAttribute('notification.type', 'email');
            await sendEmail(order.userEmail);
            notifSpan.end();
          });

          span.setStatus({ code: SpanStatusCode.OK });
          return { success: true };
        } catch (error) {
          span.setStatus({ code: SpanStatusCode.ERROR });
          span.recordException(error);
          throw error;
        } finally {
          span.end();
        }
      });
    }
    ```

=== "Go Instrumentation"
    ```go
    package main

    import (
        "context"
        "go.opentelemetry.io/otel"
        "go.opentelemetry.io/otel/attribute"
        "go.opentelemetry.io/otel/codes"
        "go.opentelemetry.io/otel/exporters/jaeger"
        "go.opentelemetry.io/otel/sdk/resource"
        "go.opentelemetry.io/otel/sdk/trace"
        semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
    )

    func initTracer() (*trace.TracerProvider, error) {
        // Create Jaeger exporter
        exporter, err := jaeger.New(
            jaeger.WithCollectorEndpoint(
                jaeger.WithEndpoint("http://localhost:14268/api/traces"),
            ),
        )
        if err != nil {
            return nil, err
        }

        // Create tracer provider
        tp := trace.NewTracerProvider(
            trace.WithBatcher(exporter),
            trace.WithResource(resource.NewWithAttributes(
                semconv.SchemaURL,
                semconv.ServiceNameKey.String("api-service"),
                semconv.ServiceVersionKey.String("1.2.3"),
                attribute.String("environment", "production"),
            )),
        )

        otel.SetTracerProvider(tp)
        return tp, nil
    }

    // HTTP handler with tracing
    func GetUser(w http.ResponseWriter, r *http.Request) {
        ctx := r.Context()
        tracer := otel.Tracer("api-service")

        ctx, span := tracer.Start(ctx, "get-user")
        defer span.End()

        userID := mux.Vars(r)["id"]
        span.SetAttributes(
            attribute.String("user.id", userID),
            attribute.String("http.method", r.Method),
        )

        // Database query with child span
        user, err := getUserFromDB(ctx, userID)
        if err != nil {
            span.SetStatus(codes.Error, err.Error())
            span.RecordError(err)
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }

        span.SetStatus(codes.Ok, "User retrieved successfully")
        json.NewEncoder(w).Encode(user)
    }

    func getUserFromDB(ctx context.Context, userID string) (*User, error) {
        tracer := otel.Tracer("api-service")
        ctx, span := tracer.Start(ctx, "db-query")
        defer span.End()

        span.SetAttributes(
            attribute.String("db.system", "postgresql"),
            attribute.String("db.statement", "SELECT * FROM users WHERE id = $1"),
        )

        user, err := db.QueryUserByID(ctx, userID)
        if err != nil {
            span.SetStatus(codes.Error, err.Error())
            return nil, err
        }

        span.SetStatus(codes.Ok, "")
        return user, nil
    }

    // Trace context propagation
    func CallExternalService(ctx context.Context) error {
        tracer := otel.Tracer("api-service")
        ctx, span := tracer.Start(ctx, "external-api-call")
        defer span.End()

        req, err := http.NewRequestWithContext(ctx, "GET", "https://api.example.com", nil)
        if err != nil {
            return err
        }

        // Propagate trace context via HTTP headers
        otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))

        resp, err := http.DefaultClient.Do(req)
        if err != nil {
            span.SetStatus(codes.Error, err.Error())
            return err
        }
        defer resp.Body.Close()

        span.SetAttributes(attribute.Int("http.status_code", resp.StatusCode))
        return nil
    }
    ```

=== "Python Instrumentation"
    ```python
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from flask import Flask

    # Initialize tracer
    def init_tracer():
        resource = Resource(attributes={
            SERVICE_NAME: "api-service"
        })

        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )

        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

    app = Flask(__name__)
    init_tracer()

    # Auto-instrument Flask
    FlaskInstrumentor().instrument_app(app)
    RequestsInstrumentor().instrument()

    # Manual instrumentation
    tracer = trace.get_tracer(__name__)

    @app.route('/api/users/<user_id>')
    def get_user(user_id):
        with tracer.start_as_current_span("get-user") as span:
            span.set_attribute("user.id", user_id)

            try:
                # Database query
                with tracer.start_as_current_span("db-query") as db_span:
                    db_span.set_attribute("db.system", "postgresql")
                    user = db.get_user(user_id)

                span.set_status(trace.Status(trace.StatusCode.OK))
                return jsonify(user)
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                return jsonify({"error": str(e)}), 500

    # Async operations
    async def process_order(order_id):
        with tracer.start_as_current_span("process-order") as span:
            span.set_attribute("order.id", order_id)

            # Payment
            with tracer.start_as_current_span("payment") as payment_span:
                payment_span.set_attribute("payment.method", "stripe")
                await charge_payment(order_id)

            # Email
            with tracer.start_as_current_span("send-email") as email_span:
                email_span.set_attribute("notification.type", "email")
                await send_notification(order_id)

            span.add_event("order-processed")
    ```

---

## Jaeger

=== "Docker Setup"
    ```yaml
    # docker-compose.yml
    version: '3.8'

    services:
      jaeger:
        image: jaegertracing/all-in-one:1.52
        environment:
          - COLLECTOR_ZIPKIN_HOST_PORT=:9411
          - COLLECTOR_OTLP_ENABLED=true
        ports:
          - "5775:5775/udp"   # Agent (deprecated)
          - "6831:6831/udp"   # Agent
          - "6832:6832/udp"   # Agent
          - "5778:5778"       # Config
          - "16686:16686"     # UI
          - "14250:14250"     # gRPC
          - "14268:14268"     # HTTP
          - "14269:14269"     # Admin
          - "9411:9411"       # Zipkin
          - "4317:4317"       # OTLP gRPC
          - "4318:4318"       # OTLP HTTP
        volumes:
          - jaeger-data:/badger

    volumes:
      jaeger-data:
    ```

=== "Querying Traces"
    ```bash
    # Jaeger UI (http://localhost:16686)

    # Find traces by service
    Service: api-service
    Operation: GET /api/orders

    # Find slow traces
    Min Duration: 1s

    # Find traces with errors
    Tags: error=true

    # Find specific trace
    Trace ID: abc123def456

    # Find traces by custom tag
    Tags: user.id=12345
    ```

---

## Zipkin

=== "Setup"
    ```yaml
    # docker-compose.yml
    version: '3.8'

    services:
      zipkin:
        image: openzipkin/zipkin:2.24
        ports:
          - "9411:9411"
        environment:
          - STORAGE_TYPE=elasticsearch
          - ES_HOSTS=elasticsearch:9200

      elasticsearch:
        image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
        environment:
          - discovery.type=single-node
    ```

=== "Node.js Integration"
    ```javascript
    const { ZipkinExporter } = require('@opentelemetry/exporter-zipkin');

    const exporter = new ZipkinExporter({
      url: 'http://localhost:9411/api/v2/spans',
      serviceName: 'api-service'
    });

    provider.addSpanProcessor(new BatchSpanProcessor(exporter));
    ```

---

## Sampling Strategies

=== "Sampling Types"
    ```javascript
    const { TraceIdRatioBasedSampler, ParentBasedSampler, AlwaysOnSampler, AlwaysOffSampler } = require('@opentelemetry/sdk-trace-base');

    // 1. Always sample (development)
    const alwaysSampler = new AlwaysOnSampler();

    // 2. Never sample
    const neverSampler = new AlwaysOffSampler();

    // 3. Probability-based (10% of traces)
    const probabilitySampler = new TraceIdRatioBasedSampler(0.1);

    // 4. Parent-based (follow parent decision)
    const parentBasedSampler = new ParentBasedSampler({
      root: new TraceIdRatioBasedSampler(0.1), // Root spans: 10%
      remoteParentSampled: new AlwaysOnSampler(), // If parent sampled: always
      remoteParentNotSampled: new AlwaysOffSampler() // If parent not sampled: never
    });

    // 5. Custom sampler (sample errors always)
    class ErrorSampler {
      shouldSample(context, traceId, spanName, spanKind, attributes) {
        // Always sample if there's an error
        if (attributes['error'] === true) {
          return {
            decision: SamplingDecision.RECORD_AND_SAMPLED
          };
        }

        // Otherwise, sample 10%
        return {
          decision: Math.random() < 0.1
            ? SamplingDecision.RECORD_AND_SAMPLED
            : SamplingDecision.NOT_RECORD
        };
      }
    }

    // Use in provider
    const provider = new NodeTracerProvider({
      sampler: parentBasedSampler
    });
    ```

=== "Sampling Configuration"
    ```
    Sampling Strategies by Environment:

    Development:
    - Sample: 100%
    - Reason: Debug all requests

    Staging:
    - Sample: 50%
    - Reason: Test with representative data

    Production (Low Traffic):
    - Sample: 10-25%
    - Reason: Capture enough traces without overhead

    Production (High Traffic):
    - Sample: 1-5%
    - Reason: Minimize performance impact

    Always Sample:
    - Errors and exceptions
    - Slow requests (> 1s)
    - Requests from specific users/features
    ```

---

## Performance Impact

=== "Overhead"
    ```
    Tracing Overhead:

    No Tracing:
    - Baseline: 100ms request

    With Tracing (Synchronous):
    - Overhead: +2-5ms per request
    - Impact: 2-5%

    With Tracing (Async Batching):
    - Overhead: +0.1-0.5ms per request
    - Impact: <1%

    Mitigation Strategies:
    1. Use async span processors
    2. Batch export spans
    3. Sample aggressively in production
    4. Use tail-based sampling
    ```

=== "Best Practices"
    ```
    ✅ DO:
    - Use async span processors
    - Batch span exports
    - Sample based on traffic
    - Propagate trace context via headers
    - Add meaningful attributes
    - Record exceptions in spans

    ❌ DON'T:
    - Sample 100% in high-traffic production
    - Create too many spans (span explosion)
    - Include sensitive data in attributes
    - Block on span export
    - Ignore sampling
    ```

---

## Service Mesh Integration

=== "Istio"
    ```yaml
    # Enable tracing in Istio
    apiVersion: install.istio.io/v1alpha1
    kind: IstioOperator
    spec:
      meshConfig:
        enableTracing: true
        defaultConfig:
          tracing:
            sampling: 10.0
            zipkin:
              address: zipkin.istio-system:9411

    # Inject sidecar
    kubectl label namespace default istio-injection=enabled

    # Trace context automatically propagated between services
    ```

=== "Linkerd"
    ```yaml
    # Linkerd automatically adds tracing headers
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: linkerd-config
      namespace: linkerd
    data:
      config: |
        tracing:
          enabled: true
          collector: jaeger-collector.jaeger:14268
          sample_rate: 0.1
    ```

---

## Trace Analysis

=== "Finding Bottlenecks"
    ```
    Steps to Identify Bottlenecks:

    1. Sort traces by duration (slowest first)
    2. Look for longest spans
    3. Check for sequential vs parallel execution
    4. Identify N+1 queries
    5. Find excessive external API calls

    Example Analysis:

    Trace: GET /api/dashboard (3.5s)
    ├─ Auth check (50ms) ✅ Fast
    ├─ Get user (100ms) ✅ Fast
    └─ Get dashboard data (3.3s) ❌ SLOW
       ├─ Get posts (2.8s) ❌ VERY SLOW
       │  ├─ DB query posts (100ms) ✅
       │  └─ Get authors (2.7s) ❌ N+1 PROBLEM
       │     ├─ DB query author (50ms) x 50 ❌
       └─ Get notifications (500ms) ⚠️  Slow

    Findings:
    - N+1 query problem: fetching 50 authors individually
    - Solution: Use JOIN or batch query
    - Expected improvement: 3.5s → 600ms (83% faster)
    ```

=== "Error Investigation"
    ```
    Finding Root Cause of Errors:

    1. Filter traces by error status
    2. Find first failed span in trace
    3. Check error message and stack trace
    4. Correlate with logs using trace ID
    5. Identify common patterns

    Example:

    Trace ID: abc123
    Status: ERROR
    ├─ API Gateway (OK)
    ├─ Auth Service (OK)
    ├─ Order Service (OK)
    └─ Payment Service (ERROR)
       Error: "Payment declined"
       Attributes:
         - payment.method: "credit_card"
         - error.code: "insufficient_funds"
         - user.id: "12345"

    Next Steps:
    - Query logs for trace ID abc123
    - Check payment service error logs
    - Review user's payment history
    ```

---

## Interview Talking Points

**Q: What's the difference between logging and tracing?**

✅ **Strong Answer:**
> "Logs are discrete events that happen within a single service - like 'payment processed' or 'error occurred'. They're great for debugging a specific service but lack context across services. Tracing tracks a single request's journey through multiple services, showing the complete call graph with timing and dependencies. For example, if a request is slow, logs might show 'API responded in 2s' but tracing shows exactly which service caused the delay - maybe a database query in the user service took 1.8s. In practice, I use tracing to identify where problems occur across services, then dive into logs for detailed context. Modern systems correlate logs and traces using trace IDs."

**Q: How do you handle trace sampling in production?**

✅ **Strong Answer:**
> "I use parent-based sampling with a low sample rate for normal requests but always sample errors and slow requests. In low-traffic systems, I might sample 10-25%, but in high-traffic production, I sample 1-5% to minimize overhead. The key is using async span processors with batching so span export doesn't block request handling. I'd also implement tail-based sampling where the backend decides whether to keep a trace after seeing all spans - this lets us sample 100% of errors while keeping overall rate low. For critical user flows like checkout, I might sample at a higher rate. The tracing overhead should be under 1% with proper configuration."

---

## Related Topics

- [Logging](logging.md) - Correlate logs with traces
- [Monitoring](monitoring.md) - Metrics and tracing together
- [Alerting](alerting.md) - Alert on trace anomalies

---

**Trace everything, find anything! 🔍**
