# Logging

**Track application behavior** | 📝 Structured | 📚 Centralized | 🔍 Searchable

---

## Overview

Logging captures discrete events that occur in your system, providing context for debugging, auditing, and understanding system behavior.

**Why Structured Logging?**

- Machine-parseable
- Searchable and filterable
- Correlation across services
- Better observability

---

## Log Levels

=== "Standard Levels"
    ```
    Log Level Hierarchy (most to least severe):

    FATAL/CRITICAL - System is unusable
    ├─ Database connection lost
    ├─ Out of memory
    └─ Cannot start application

    ERROR - Something failed
    ├─ Payment processing failed
    ├─ External API returned 500
    └─ Failed to save to database

    WARN - Potential problem
    ├─ Deprecated API usage
    ├─ Slow query (> 1s)
    └─ Retry attempt

    INFO - Normal operation
    ├─ Request started/completed
    ├─ User logged in
    └─ Service started

    DEBUG - Detailed diagnostic info
    ├─ Variable values
    ├─ Function entry/exit
    └─ Configuration values

    TRACE - Very detailed info
    ├─ SQL queries with parameters
    ├─ Full request/response bodies
    └─ Step-by-step execution flow
    ```

=== "Best Practices"
    ```
    ✅ DO:
    - Use ERROR for failures requiring attention
    - Use WARN for degraded but functional state
    - Use INFO for business events (signup, purchase)
    - Use DEBUG for development troubleshooting
    - Log errors with stack traces

    ❌ DON'T:
    - Log sensitive data (passwords, credit cards)
    - Log at TRACE in production (too verbose)
    - Use ERROR for expected conditions
    - Log every single database query
    - Mix log levels inconsistently
    ```

---

## Structured Logging

=== "Winston (Node.js)"
    ```javascript
    const winston = require('winston');

    // Create logger
    const logger = winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: winston.format.combine(
        winston.format.timestamp({
          format: 'YYYY-MM-DD HH:mm:ss'
        }),
        winston.format.errors({ stack: true }),
        winston.format.splat(),
        winston.format.json()
      ),
      defaultMeta: {
        service: 'api-service',
        environment: process.env.NODE_ENV,
        version: process.env.APP_VERSION
      },
      transports: [
        // Write errors to error.log
        new winston.transports.File({
          filename: 'logs/error.log',
          level: 'error',
          maxsize: 10485760, // 10MB
          maxFiles: 5
        }),
        // Write all logs to combined.log
        new winston.transports.File({
          filename: 'logs/combined.log',
          maxsize: 10485760,
          maxFiles: 10
        })
      ]
    });

    // Console transport for development
    if (process.env.NODE_ENV !== 'production') {
      logger.add(new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple()
        )
      }));
    }

    // Request logging middleware
    function requestLogger(req, res, next) {
      const start = Date.now();

      res.on('finish', () => {
        const duration = Date.now() - start;

        logger.info('HTTP Request', {
          method: req.method,
          url: req.url,
          status: res.statusCode,
          duration,
          ip: req.ip,
          userAgent: req.get('user-agent'),
          userId: req.user?.id,
          requestId: req.id
        });
      });

      next();
    }

    // Business event logging
    logger.info('User registered', {
      userId: user.id,
      email: user.email,
      registrationMethod: 'email',
      referralSource: req.query.ref
    });

    // Error logging with context
    try {
      await processPayment(order);
    } catch (error) {
      logger.error('Payment processing failed', {
        error: error.message,
        stack: error.stack,
        orderId: order.id,
        userId: order.userId,
        amount: order.amount,
        paymentMethod: order.paymentMethod
      });
      throw error;
    }

    // Performance logging
    const timer = logger.startTimer();
    await heavyOperation();
    timer.done({ message: 'Operation completed', operationType: 'batch-import' });
    ```

=== "Logrus (Go)"
    ```go
    package main

    import (
        "github.com/sirupsen/logrus"
        "os"
    )

    var log = logrus.New()

    func initLogger() {
        // JSON formatter for production
        log.SetFormatter(&logrus.JSONFormatter{
            TimestampFormat: "2006-01-02 15:04:05",
            FieldMap: logrus.FieldMap{
                logrus.FieldKeyTime: "timestamp",
                logrus.FieldKeyLevel: "level",
                logrus.FieldKeyMsg: "message",
            },
        })

        // Output to stdout
        log.SetOutput(os.Stdout)

        // Log level from environment
        level, err := logrus.ParseLevel(os.Getenv("LOG_LEVEL"))
        if err != nil {
            level = logrus.InfoLevel
        }
        log.SetLevel(level)

        // Add default fields
        log.WithFields(logrus.Fields{
            "service":     "api-service",
            "environment": os.Getenv("ENV"),
            "version":     os.Getenv("VERSION"),
        })
    }

    // HTTP request logging
    func LogRequest(r *http.Request, statusCode int, duration time.Duration) {
        log.WithFields(logrus.Fields{
            "method":     r.Method,
            "url":        r.URL.Path,
            "status":     statusCode,
            "duration":   duration.Milliseconds(),
            "ip":         r.RemoteAddr,
            "user_agent": r.UserAgent(),
            "request_id": r.Header.Get("X-Request-ID"),
        }).Info("HTTP Request")
    }

    // Error logging
    func ProcessOrder(order Order) error {
        logger := log.WithFields(logrus.Fields{
            "order_id": order.ID,
            "user_id":  order.UserID,
            "amount":   order.Amount,
        })

        logger.Info("Processing order")

        if err := validateOrder(order); err != nil {
            logger.WithError(err).Error("Order validation failed")
            return err
        }

        if err := chargePayment(order); err != nil {
            logger.WithFields(logrus.Fields{
                "payment_method": order.PaymentMethod,
                "error_code":     err.Code,
            }).Error("Payment failed")
            return err
        }

        logger.Info("Order processed successfully")
        return nil
    }
    ```

=== "Python Logging"
    ```python
    import logging
    import json
    from datetime import datetime

    # Custom JSON formatter
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'logger': record.name,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            # Add extra fields
            if hasattr(record, 'extra'):
                log_record.update(record.extra)

            # Add exception info
            if record.exc_info:
                log_record['exception'] = self.formatException(record.exc_info)

            return json.dumps(log_record)

    # Configure logger
    def setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Console handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        return logger

    logger = setup_logger(__name__)

    # Usage
    logger.info('User logged in', extra={
        'user_id': user.id,
        'ip': request.remote_addr,
        'session_id': session.id
    })

    try:
        result = process_payment(order)
        logger.info('Payment processed', extra={
            'order_id': order.id,
            'amount': order.amount,
            'transaction_id': result.transaction_id
        })
    except PaymentError as e:
        logger.error('Payment failed', extra={
            'order_id': order.id,
            'error_code': e.code,
            'error_message': str(e)
        }, exc_info=True)
    ```

---

## Centralized Logging

=== "ELK Stack"
    ```yaml
    # docker-compose.yml
    version: '3.8'

    services:
      elasticsearch:
        image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
        environment:
          - discovery.type=single-node
          - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
          - xpack.security.enabled=false
        ports:
          - "9200:9200"
        volumes:
          - es-data:/usr/share/elasticsearch/data

      logstash:
        image: docker.elastic.co/logstash/logstash:8.11.0
        volumes:
          - ./logstash/pipeline:/usr/share/logstash/pipeline
        ports:
          - "5044:5044"
          - "9600:9600"
        depends_on:
          - elasticsearch

      kibana:
        image: docker.elastic.co/kibana/kibana:8.11.0
        ports:
          - "5601:5601"
        environment:
          - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
        depends_on:
          - elasticsearch

      filebeat:
        image: docker.elastic.co/beats/filebeat:8.11.0
        user: root
        volumes:
          - ./filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
          - /var/lib/docker/containers:/var/lib/docker/containers:ro
          - /var/run/docker.sock:/var/run/docker.sock:ro
        command: filebeat -e -strict.perms=false
        depends_on:
          - logstash

    volumes:
      es-data:
    ```

    ```yaml
    # filebeat/filebeat.yml
    filebeat.inputs:
      - type: container
        paths:
          - '/var/lib/docker/containers/*/*.log'
        processors:
          - add_docker_metadata:
              host: "unix:///var/run/docker.sock"
          - decode_json_fields:
              fields: ["message"]
              target: ""
              overwrite_keys: true

    output.logstash:
      hosts: ["logstash:5044"]
    ```

    ```ruby
    # logstash/pipeline/logstash.conf
    input {
      beats {
        port => 5044
      }
    }

    filter {
      # Parse JSON logs
      json {
        source => "message"
      }

      # Add timestamp
      date {
        match => ["timestamp", "ISO8601"]
        target => "@timestamp"
      }

      # Grok for non-JSON logs
      if ![level] {
        grok {
          match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:message}" }
        }
      }

      # Classify log levels
      if [level] == "ERROR" or [level] == "FATAL" {
        mutate {
          add_tag => ["error"]
        }
      }
    }

    output {
      elasticsearch {
        hosts => ["elasticsearch:9200"]
        index => "logs-%{+YYYY.MM.dd}"
      }

      # Debug output
      stdout {
        codec => rubydebug
      }
    }
    ```

=== "CloudWatch Logs"
    ```javascript
    const AWS = require('aws-sdk');
    const cloudwatchlogs = new AWS.CloudWatchLogs();

    class CloudWatchTransport {
      constructor(options) {
        this.logGroupName = options.logGroupName;
        this.logStreamName = options.logStreamName;
        this.sequenceToken = null;
      }

      async log(info) {
        const params = {
          logGroupName: this.logGroupName,
          logStreamName: this.logStreamName,
          logEvents: [
            {
              message: JSON.stringify(info),
              timestamp: Date.now()
            }
          ]
        };

        if (this.sequenceToken) {
          params.sequenceToken = this.sequenceToken;
        }

        try {
          const response = await cloudwatchlogs.putLogEvents(params).promise();
          this.sequenceToken = response.nextSequenceToken;
        } catch (error) {
          console.error('Failed to send logs to CloudWatch:', error);
        }
      }
    }

    // Add to Winston
    const winston = require('winston');
    const logger = winston.createLogger({
      transports: [
        new CloudWatchTransport({
          logGroupName: '/aws/application/my-app',
          logStreamName: `${process.env.HOSTNAME}-${Date.now()}`
        })
      ]
    });
    ```

=== "Fluentd"
    ```ruby
    # fluent.conf
    <source>
      @type forward
      port 24224
      bind 0.0.0.0
    </source>

    # Parse JSON logs
    <filter app.**>
      @type parser
      key_name log
      <parse>
        @type json
        time_key timestamp
        time_format %Y-%m-%dT%H:%M:%S.%L%z
      </parse>
    </filter>

    # Add metadata
    <filter app.**>
      @type record_transformer
      <record>
        hostname ${hostname}
        environment ${ENV["ENVIRONMENT"]}
      </record>
    </filter>

    # Route by log level
    <match app.**>
      @type rewrite_tag_filter
      <rule>
        key level
        pattern /^(ERROR|FATAL)$/
        tag error.${tag}
      </rule>
      <rule>
        key level
        pattern /.*/
        tag normal.${tag}
      </rule>
    </match>

    # Send errors to separate index
    <match error.**>
      @type elasticsearch
      host elasticsearch
      port 9200
      index_name errors-%Y.%m.%d
      type_name _doc
      <buffer>
        flush_interval 5s
      </buffer>
    </match>

    # Send normal logs
    <match normal.**>
      @type elasticsearch
      host elasticsearch
      port 9200
      index_name logs-%Y.%m.%d
      type_name _doc
      <buffer>
        flush_interval 10s
      </buffer>
    </match>
    ```

---

## Log Correlation

=== "Request ID"
    ```javascript
    const { v4: uuidv4 } = require('uuid');

    // Middleware to add request ID
    function requestIdMiddleware(req, res, next) {
      req.id = req.get('X-Request-ID') || uuidv4();
      res.set('X-Request-ID', req.id);
      next();
    }

    // Child logger with request ID
    function loggerMiddleware(req, res, next) {
      req.logger = logger.child({
        requestId: req.id,
        userId: req.user?.id
      });
      next();
    }

    app.use(requestIdMiddleware);
    app.use(loggerMiddleware);

    // All logs automatically include request ID
    app.get('/api/users/:id', async (req, res) => {
      req.logger.info('Fetching user');
      const user = await db.users.findById(req.params.id);
      req.logger.info('User fetched', { userId: user.id });
      res.json(user);
    });
    ```

=== "Trace Context"
    ```javascript
    // OpenTelemetry integration
    const { trace } = require('@opentelemetry/api');

    function correlateWithTrace(logger) {
      return logger.child({
        traceId: trace.getActiveSpan()?.spanContext().traceId,
        spanId: trace.getActiveSpan()?.spanContext().spanId
      });
    }

    // Usage
    app.get('/api/orders/:id', async (req, res) => {
      const logger = correlateWithTrace(req.logger);

      logger.info('Processing order request');
      const order = await getOrder(req.params.id);
      logger.info('Order retrieved', { orderId: order.id });

      res.json(order);
    });
    ```

---

## Log Search Queries

=== "Kibana/Elasticsearch"
    ```
    # Find errors in last hour
    level:"ERROR" AND @timestamp:[now-1h TO now]

    # Find slow requests
    duration:>1000 AND method:"GET"

    # Find user activity
    userId:"12345" AND (action:"login" OR action:"purchase")

    # Find errors for specific endpoint
    url:"/api/payments" AND status:>499

    # Aggregate errors by service
    {
      "aggs": {
        "errors_by_service": {
          "terms": {
            "field": "service.keyword"
          }
        }
      }
    }

    # Find error spike
    {
      "query": {
        "bool": {
          "must": [
            { "match": { "level": "ERROR" }},
            { "range": { "@timestamp": { "gte": "now-15m" }}}
          ]
        }
      }
    }
    ```

=== "CloudWatch Logs Insights"
    ```
    # Find errors in last hour
    fields @timestamp, level, message, error
    | filter level = "ERROR"
    | sort @timestamp desc
    | limit 100

    # Latency percentiles
    fields @timestamp, duration
    | filter method = "GET"
    | stats avg(duration), pct(duration, 50), pct(duration, 95), pct(duration, 99)

    # Error rate over time
    fields @timestamp, level
    | stats count(*) as total, count(level = "ERROR") as errors by bin(5m)
    | fields bin(5m), (errors / total * 100) as error_rate

    # Top error messages
    fields @timestamp, message
    | filter level = "ERROR"
    | stats count(*) as count by message
    | sort count desc
    | limit 10
    ```

---

## Best Practices

=== "What to Log"
    ```
    ✅ DO LOG:
    - HTTP requests (method, URL, status, duration)
    - Authentication events (login, logout, failed attempts)
    - Business events (order created, payment processed)
    - Errors with full context and stack traces
    - External API calls (URL, duration, status)
    - Background job start/completion
    - Configuration changes
    - Database query slow logs

    ❌ DON'T LOG:
    - Passwords or API keys
    - Credit card numbers
    - Social Security Numbers
    - Personal health information
    - Authentication tokens
    - Encryption keys
    - Every single database query (too verbose)
    - User-generated content without sanitization
    ```

=== "Log Format"
    ```json
    {
      "timestamp": "2024-01-15T10:30:45.123Z",
      "level": "INFO",
      "message": "Order processed successfully",
      "service": "order-service",
      "environment": "production",
      "version": "1.2.3",
      "requestId": "abc123",
      "traceId": "def456",
      "userId": "user789",
      "orderId": "order101",
      "amount": 99.99,
      "duration": 234,
      "host": "server-01",
      "pid": 1234
    }
    ```

=== "Performance"
    ```javascript
    // Sampling for verbose logs
    function shouldLog(level, sampleRate = 1.0) {
      if (level === 'ERROR' || level === 'WARN') {
        return true; // Always log errors/warnings
      }
      return Math.random() < sampleRate;
    }

    if (shouldLog('DEBUG', 0.1)) {
      logger.debug('Verbose debug info', { data });
    }

    // Async logging
    class AsyncLogger {
      constructor() {
        this.queue = [];
        this.flushInterval = setInterval(() => this.flush(), 1000);
      }

      log(entry) {
        this.queue.push(entry);
        if (this.queue.length >= 100) {
          this.flush();
        }
      }

      async flush() {
        if (this.queue.length === 0) return;

        const batch = this.queue.splice(0, this.queue.length);
        await sendLogsToRemote(batch);
      }
    }

    // Lazy evaluation for expensive operations
    logger.info(() => {
      const expensiveData = computeExpensiveDebugInfo();
      return `Debug info: ${JSON.stringify(expensiveData)}`;
    });
    ```

---

## Log Retention

| Environment | Retention | Storage | Why |
|-------------|-----------|---------|-----|
| **Production** | 30-90 days | Hot storage (S3, EBS) | Troubleshooting, compliance |
| **Archived** | 1-7 years | Cold storage (Glacier) | Legal, audit requirements |
| **Staging** | 7-14 days | Standard storage | Testing, shorter retention needed |
| **Development** | 1-3 days | Local/ephemeral | Debugging only |

---

## Interview Talking Points

**Q: What's the difference between structured and unstructured logging?**

✅ **Strong Answer:**
> "Unstructured logs are plain text strings like 'User logged in', which are human-readable but hard to parse and search. Structured logs use key-value pairs in JSON format like `{\"event\": \"login\", \"userId\": 123, \"timestamp\": \"2024-01-15T10:30:45Z\"}`, making them machine-parseable and searchable. With structured logging, I can easily query 'show all failed logins from user 123' or 'calculate P95 latency for API calls'. It's essential for modern observability stacks like ELK or DataDog because you can aggregate, filter, and analyze logs at scale. I always use structured logging in production with consistent field names across services."

**Q: How do you handle logging in microservices?**

✅ **Strong Answer:**
> "I'd use centralized logging with correlation IDs to track requests across services. Each service logs to stdout in JSON format, and a log aggregator like Fluentd or Filebeat ships logs to Elasticsearch. Every request gets a unique ID propagated through HTTP headers, so I can trace a single request's path through multiple services. I'd include consistent metadata in every log: service name, version, environment, and trace ID. For log levels, I follow INFO for business events, ERROR for failures, and DEBUG for troubleshooting. I'd set up alerts on error rate spikes and use Kibana dashboards to visualize errors by service. Log retention depends on compliance needs - typically 30 days in hot storage and longer in cold storage."

---

## Related Topics

- [Monitoring](monitoring.md) - Metrics and dashboards
- [Tracing](tracing.md) - Distributed tracing
- [Alerting](alerting.md) - Log-based alerts

---

**Log everything important, search anything! 📝**
