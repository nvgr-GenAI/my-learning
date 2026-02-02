# Alerting

**Get notified when things go wrong** | 🚨 Alerts | 📞 On-Call | 🔔 Notifications

---

## Overview

Alerting notifies engineers when systems deviate from normal behavior, enabling rapid response to incidents before they impact users.

**Why Alert?**

- Early problem detection
- Reduce mean time to resolution (MTTR)
- Minimize user impact
- Enable proactive fixes
- Maintain SLAs

---

## Alert Principles

=== "Philosophy"
    ```
    Good Alerting Philosophy:

    1. Every alert should be actionable
       ❌ "CPU is high"
       ✅ "CPU >90% for 10min, requests queuing"

    2. Alerts should indicate impact
       ❌ "Database connection pool full"
       ✅ "50% of requests failing due to DB connections"

    3. Alert on symptoms, not causes
       ❌ "Redis is down"
       ✅ "Cache hit rate <20%, API latency increased"

    4. Reduce alert fatigue
       - Group related alerts
       - Set appropriate thresholds
       - Implement alert deduplication
       - Use severity levels correctly

    5. Test your alerts
       - Chaos engineering
       - Regular fire drills
       - Verify escalation paths
    ```

=== "Alert Severity"
    ```
    Severity Levels:

    P0 - CRITICAL (Page immediately)
    ├─ Complete service outage
    ├─ Data loss in progress
    ├─ Security breach
    └─ SLA violation
    Response: Immediate (5min)
    Escalation: Manager + VP if not resolved in 30min

    P1 - HIGH (Page on-call)
    ├─ Partial service degradation
    ├─ High error rate (>5%)
    ├─ Database replication lag >10min
    └─ Key feature broken
    Response: 15 minutes
    Escalation: Manager if not resolved in 1hr

    P2 - MEDIUM (Ticket during business hours)
    ├─ Non-critical feature issues
    ├─ Performance degradation
    ├─ Elevated error rates (1-5%)
    └─ Resource warnings
    Response: Same day
    Escalation: None

    P3 - LOW (Ticket, low priority)
    ├─ Minor issues
    ├─ Warnings about future problems
    ├─ Certificate expiring in 30 days
    └─ Disk >80% (but not growing)
    Response: This week
    Escalation: None
    ```

---

## Prometheus AlertManager

=== "Alert Rules"
    ```yaml
    # alerts/api-alerts.yml
    groups:
      - name: api_alerts
        interval: 30s
        rules:
          # High error rate
          - alert: HighErrorRate
            expr: |
              (
                sum(rate(http_requests_total{status=~"5.."}[5m]))
                /
                sum(rate(http_requests_total[5m]))
              ) > 0.05
            for: 10m
            labels:
              severity: critical
              team: backend
              service: api
            annotations:
              summary: "High error rate detected"
              description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
              runbook: "https://wiki.company.com/runbooks/high-error-rate"
              dashboard: "https://grafana.company.com/d/api-overview"

          # API latency
          - alert: HighLatency
            expr: |
              histogram_quantile(0.99,
                rate(http_request_duration_seconds_bucket[5m])
              ) > 1
            for: 5m
            labels:
              severity: warning
              team: backend
            annotations:
              summary: "API latency is high"
              description: "P99 latency is {{ $value }}s (threshold: 1s)"

          # Service down
          - alert: ServiceDown
            expr: up{job="api-service"} == 0
            for: 2m
            labels:
              severity: critical
              team: backend
            annotations:
              summary: "Service {{ $labels.instance }} is down"
              description: "{{ $labels.instance }} has been down for more than 2 minutes"

          # High memory usage
          - alert: HighMemoryUsage
            expr: |
              (
                process_resident_memory_bytes{job="api-service"}
                /
                node_memory_MemTotal_bytes
              ) > 0.9
            for: 10m
            labels:
              severity: warning
              team: infrastructure
            annotations:
              summary: "High memory usage on {{ $labels.instance }}"
              description: "Memory usage is {{ $value | humanizePercentage }}"

          # Database connection pool exhausted
          - alert: DatabaseConnectionPoolExhausted
            expr: |
              (
                database_connections_active
                /
                database_connections_max
              ) > 0.9
            for: 5m
            labels:
              severity: critical
              team: backend
            annotations:
              summary: "Database connection pool nearly exhausted"
              description: "{{ $value | humanizePercentage }} of connections in use"

          # Disk space
          - alert: DiskSpaceRunningOut
            expr: |
              (
                node_filesystem_avail_bytes{mountpoint="/"}
                /
                node_filesystem_size_bytes{mountpoint="/"}
              ) < 0.1
            for: 10m
            labels:
              severity: warning
              team: infrastructure
            annotations:
              summary: "Disk space running out on {{ $labels.instance }}"
              description: "Only {{ $value | humanizePercentage }} disk space remaining"

          # Deployment anomaly detection
          - alert: DeploymentAnomalyDetected
            expr: |
              (
                rate(http_requests_total{status=~"5.."}[5m])
                /
                rate(http_requests_total{status=~"5.."}[5m] offset 1h)
              ) > 5
            for: 5m
            labels:
              severity: critical
              team: backend
            annotations:
              summary: "Possible bad deployment detected"
              description: "Error rate is 5x higher than 1 hour ago"
    ```

=== "AlertManager Config"
    ```yaml
    # alertmanager.yml
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

    # Alert routing
    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s        # Wait before sending initial notification
      group_interval: 10s    # Wait before sending updates
      repeat_interval: 12h   # Resend alert if still firing

      receiver: 'default'

      routes:
        # Critical alerts go to PagerDuty
        - match:
            severity: critical
          receiver: 'pagerduty'
          continue: true  # Also send to other receivers

        # Critical alerts also go to Slack
        - match:
            severity: critical
          receiver: 'slack-critical'

        # Warnings go to Slack only
        - match:
            severity: warning
          receiver: 'slack-warnings'

        # Database alerts go to DB team
        - match_re:
            service: database
          receiver: 'db-team'

    # Inhibition rules (suppress alerts)
    inhibit_rules:
      # If service is down, don't alert on high latency
      - source_match:
          alertname: 'ServiceDown'
        target_match:
          alertname: 'HighLatency'
        equal: ['instance']

      # If node is down, don't alert on disk space
      - source_match:
          alertname: 'NodeDown'
        target_match_re:
          alertname: '(DiskSpace|HighMemory|HighCPU)'
        equal: ['instance']

    # Receivers (notification channels)
    receivers:
      - name: 'default'
        webhook_configs:
          - url: 'http://localhost:5001/alert'

      - name: 'pagerduty'
        pagerduty_configs:
          - service_key: 'YOUR_PAGERDUTY_KEY'
            description: '{{ .CommonAnnotations.summary }}'
            details:
              firing: '{{ .Alerts.Firing | len }}'
              resolved: '{{ .Alerts.Resolved | len }}'

      - name: 'slack-critical'
        slack_configs:
          - channel: '#alerts-critical'
            title: '🚨 CRITICAL: {{ .CommonAnnotations.summary }}'
            text: |
              {{ range .Alerts }}
              *Alert:* {{ .Labels.alertname }}
              *Severity:* {{ .Labels.severity }}
              *Description:* {{ .Annotations.description }}
              *Runbook:* {{ .Annotations.runbook }}
              {{ end }}
            color: 'danger'

      - name: 'slack-warnings'
        slack_configs:
          - channel: '#alerts-warnings'
            title: '⚠️ WARNING: {{ .CommonAnnotations.summary }}'
            color: 'warning'

      - name: 'db-team'
        email_configs:
          - to: 'db-team@company.com'
            from: 'alerts@company.com'
            smarthost: 'smtp.gmail.com:587'
            auth_username: 'alerts@company.com'
            auth_password: 'password'
    ```

=== "Testing Alerts"
    ```bash
    # Test alert manually
    curl -X POST http://localhost:9093/api/v1/alerts -d '[
      {
        "labels": {
          "alertname": "TestAlert",
          "severity": "warning"
        },
        "annotations": {
          "summary": "This is a test alert"
        }
      }
    ]'

    # Check active alerts
    curl http://localhost:9093/api/v1/alerts

    # Silence an alert
    amtool silence add alertname=HighErrorRate --duration=1h --comment="Maintenance window"

    # View silences
    amtool silence query
    ```

---

## CloudWatch Alarms

=== "Creating Alarms"
    ```javascript
    const AWS = require('aws-sdk');
    const cloudwatch = new AWS.CloudWatch();

    // High error rate alarm
    async function createErrorRateAlarm() {
      const params = {
        AlarmName: 'HighErrorRate',
        ComparisonOperator: 'GreaterThanThreshold',
        EvaluationPeriods: 2,
        MetricName: 'ErrorRate',
        Namespace: 'MyApp',
        Period: 300,  // 5 minutes
        Statistic: 'Average',
        Threshold: 5.0,
        ActionsEnabled: true,
        AlarmActions: [
          'arn:aws:sns:us-east-1:123456789012:critical-alerts'
        ],
        AlarmDescription: 'Alert when error rate exceeds 5%',
        TreatMissingData: 'notBreaching',
        Dimensions: [
          {
            Name: 'Environment',
            Value: 'production'
          }
        ]
      };

      await cloudwatch.putMetricAlarm(params).promise();
    }

    // Composite alarm (multiple conditions)
    async function createCompositeAlarm() {
      const params = {
        AlarmName: 'ServiceDegraded',
        AlarmRule: 'ALARM(HighErrorRate) OR ALARM(HighLatency)',
        ActionsEnabled: true,
        AlarmActions: [
          'arn:aws:sns:us-east-1:123456789012:critical-alerts'
        ]
      };

      await cloudwatch.putCompositeAlarm(params).promise();
    }

    // Anomaly detection alarm
    async function createAnomalyAlarm() {
      const params = {
        AlarmName: 'AnomalousTraffic',
        ComparisonOperator: 'LessThanLowerOrGreaterThanUpperThreshold',
        EvaluationPeriods: 2,
        Metrics: [
          {
            Id: 'm1',
            ReturnData: true,
            MetricStat: {
              Metric: {
                Namespace: 'MyApp',
                MetricName: 'RequestCount'
              },
              Period: 300,
              Stat: 'Average'
            }
          },
          {
            Id: 'ad1',
            Expression: 'ANOMALY_DETECTION_BAND(m1, 2)'
          }
        ],
        ThresholdMetricId: 'ad1'
      };

      await cloudwatch.putMetricAlarm(params).promise();
    }
    ```

=== "SNS Notifications"
    ```javascript
    const AWS = require('aws-sdk');
    const sns = new AWS.SNS();

    // Create SNS topic
    async function createAlertTopic() {
      const result = await sns.createTopic({
        Name: 'critical-alerts'
      }).promise();

      // Subscribe email
      await sns.subscribe({
        TopicArn: result.TopicArn,
        Protocol: 'email',
        Endpoint: 'oncall@company.com'
      }).promise();

      // Subscribe SMS
      await sns.subscribe({
        TopicArn: result.TopicArn,
        Protocol: 'sms',
        Endpoint: '+1234567890'
      }).promise();

      // Subscribe Lambda for custom logic
      await sns.subscribe({
        TopicArn: result.TopicArn,
        Protocol: 'lambda',
        Endpoint: 'arn:aws:lambda:us-east-1:123456789012:function:AlertHandler'
      }).promise();

      return result.TopicArn;
    }

    // Lambda function to handle alerts
    exports.handler = async (event) => {
      const message = JSON.parse(event.Records[0].Sns.Message);

      if (message.NewStateValue === 'ALARM') {
        // Send to PagerDuty
        await fetch('https://events.pagerduty.com/v2/enqueue', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            routing_key: process.env.PAGERDUTY_KEY,
            event_action: 'trigger',
            payload: {
              summary: message.AlarmName,
              severity: 'critical',
              source: 'CloudWatch'
            }
          })
        });

        // Send to Slack
        await fetch(process.env.SLACK_WEBHOOK, {
          method: 'POST',
          body: JSON.stringify({
            text: `🚨 ${message.AlarmName}: ${message.NewStateReason}`
          })
        });
      }
    };
    ```

---

## DataDog Monitors

=== "Metric Monitor"
    ```python
    from datadog_api_client import ApiClient, Configuration
    from datadog_api_client.v1.api.monitors_api import MonitorsApi
    from datadog_api_client.v1.model.monitor import Monitor

    configuration = Configuration()
    with ApiClient(configuration) as api_client:
        api_instance = MonitorsApi(api_client)

        # High error rate monitor
        monitor = Monitor(
            type="metric alert",
            query="avg(last_5m):sum:api.errors{env:prod} by {service} > 100",
            name="High Error Rate",
            message="""
                Error rate is above threshold.

                Service: {{service.name}}
                Current value: {{value}}

                @slack-alerts-critical
                @pagerduty-api-team
            """,
            tags=["team:backend", "env:production"],
            options={
                "thresholds": {
                    "critical": 100,
                    "warning": 50
                },
                "notify_no_data": True,
                "no_data_timeframe": 10,
                "require_full_window": False,
                "notify_audit": False,
                "include_tags": True
            }
        )

        result = api_instance.create_monitor(body=monitor)
    ```

=== "APM Monitor"
    ```python
    # Monitor P99 latency
    monitor = Monitor(
        type="query alert",
        query="avg(last_15m):p99:trace.web.request{env:prod,service:api} > 1",
        name="High P99 Latency",
        message="""
            API P99 latency is above 1 second.

            Current P99: {{value}}s

            Check traces: https://app.datadoghq.com/apm/traces

            @slack-alerts-warnings
        """,
        tags=["team:backend", "priority:high"]
    )
    ```

=== "Anomaly Detection"
    ```python
    # Detect anomalous traffic patterns
    monitor = Monitor(
        type="query alert",
        query="avg(last_1h):anomalies(avg:api.requests{env:prod}, 'agile', 2) >= 1",
        name="Anomalous Request Rate",
        message="""
            Unusual traffic pattern detected.

            This could indicate:
            - DDoS attack
            - Traffic surge
            - Deployment issue

            @slack-alerts-warnings
        """,
        tags=["team:backend", "type:anomaly"]
    )
    ```

---

## Alert Fatigue Prevention

=== "Strategies"
    ```
    Reducing Alert Fatigue:

    1. Alert Aggregation
       - Group related alerts
       - Use time windows
       - Deduplicate similar alerts

    2. Dynamic Thresholds
       - Use anomaly detection
       - Adjust based on time of day
       - Consider historical baselines

    3. Alert Routing
       - Route by severity
       - Route by team
       - Route by time (business hours vs off-hours)

    4. Silence Periods
       - Maintenance windows
       - Known issues
       - Deploy windows

    5. Alert Tuning
       - Review alert effectiveness
       - Adjust thresholds regularly
       - Remove noisy alerts

    6. Actionable Alerts Only
       - Every alert needs a runbook
       - Clear next steps
       - Known resolution paths
    ```

=== "Alert Metrics"
    ```
    Measure Alert Health:

    1. Alert Volume
       - Alerts per day
       - Alerts per engineer
       - Trend over time

    2. False Positive Rate
       - Alerts that auto-resolve
       - Alerts closed as "not an issue"
       - Target: <10%

    3. Time to Acknowledge
       - How long to respond
       - Target: <5min for critical

    4. Time to Resolve
       - Mean time to resolution (MTTR)
       - Track by severity
       - Identify patterns

    5. Alert Effectiveness
       - Did alert prevent outage?
       - Was action needed?
       - Was alert helpful?
    ```

---

## On-Call Best Practices

=== "Runbooks"
    ```markdown
    # Runbook: High Error Rate

    ## Symptoms
    - Error rate >5%
    - Users reporting issues
    - Increased latency

    ## Impact
    - Users cannot complete orders
    - Revenue impact: ~$X per minute

    ## Diagnosis Steps

    1. Check Grafana dashboard
       https://grafana.company.com/d/api-overview

    2. Query recent errors
       ```
       kubectl logs -l app=api --since=10m | grep ERROR
       ```

    3. Check external dependencies
       - Database: https://status.db.com
       - Payment API: https://status.stripe.com

    4. Review recent deployments
       ```
       kubectl rollout history deployment/api
       ```

    ## Resolution Steps

    1. If bad deployment:
       ```bash
       kubectl rollout undo deployment/api
       ```

    2. If external dependency down:
       - Enable circuit breaker
       - Notify users
       - Escalate to vendor

    3. If database issue:
       - Check connection pool
       - Review slow queries
       - Consider read replica

    ## Escalation
    - After 15min: Escalate to @tech-lead
    - After 30min: Escalate to @director-engineering

    ## Post-Incident
    - Write incident report
    - Update runbook
    - Create tickets for improvements
    ```

=== "On-Call Schedule"
    ```
    On-Call Rotation Best Practices:

    1. Schedule
       - 1 week rotations
       - Primary + Secondary on-call
       - Follow-the-sun for global teams
       - No single person >2 weeks/quarter

    2. Handoff
       - Document ongoing issues
       - Share context from previous week
       - Review recent incidents
       - Update runbooks

    3. Compensation
       - On-call pay
       - Comp time for incidents
       - Flex schedule after incidents

    4. Training
       - Shadow before taking on-call
       - Regular incident drills
       - Runbook reviews
       - Access verification

    5. Boundaries
       - Escalation criteria clear
       - No P3 alerts during off-hours
       - Respect time zones
       - Mental health support
    ```

---

## SLO-Based Alerting

=== "Error Budget"
    ```yaml
    # Alert when error budget is burning too fast
    groups:
      - name: slo_alerts
        rules:
          # SLO: 99.9% success rate
          # Error budget: 0.1%

          # Fast burn (14.4x rate) - page immediately
          - alert: ErrorBudgetFastBurn
            expr: |
              (
                (
                  sum(rate(http_requests_total{status!~"5.."}[1h]))
                  /
                  sum(rate(http_requests_total[1h]))
                ) < 0.999
              ) and
              (
                (
                  sum(rate(http_requests_total{status!~"5.."}[5m]))
                  /
                  sum(rate(http_requests_total[5m]))
                ) < 0.999
              )
            labels:
              severity: critical
            annotations:
              summary: "Error budget burning rapidly"
              description: "At current rate, error budget will be exhausted in < 2 days"

          # Slow burn (6x rate) - warning
          - alert: ErrorBudgetSlowBurn
            expr: |
              (
                (
                  sum(rate(http_requests_total{status!~"5.."}[6h]))
                  /
                  sum(rate(http_requests_total[6h]))
                ) < 0.999
              ) and
              (
                (
                  sum(rate(http_requests_total{status!~"5.."}[30m]))
                  /
                  sum(rate(http_requests_total[30m]))
                ) < 0.999
              )
            labels:
              severity: warning
            annotations:
              summary: "Error budget burning slowly"
              description: "Error budget will be exhausted this month at current rate"
    ```

---

## Interview Talking Points

**Q: How do you prevent alert fatigue?**

✅ **Strong Answer:**
> "Alert fatigue happens when engineers get too many alerts, especially false positives, causing them to ignore important ones. I prevent this by ensuring every alert is actionable and has a clear runbook. I use proper severity levels - P0 for service outages that page immediately, P1 for degradations that notify on-call, and P2/P3 for tickets during business hours. I implement alert grouping and deduplication so related issues don't create alert storms. I also use dynamic thresholds and anomaly detection instead of static thresholds - for example, traffic patterns differ between 3 AM and 3 PM. Finally, I regularly review alert metrics: if an alert has a high false positive rate or low time-to-resolution, I tune or remove it. The goal is engineers trust that every alert matters."

**Q: What's the difference between monitoring and alerting?**

✅ **Strong Answer:**
> "Monitoring is continuous observation of system health through metrics, logs, and traces - it tells you what's happening. Alerting is taking action when metrics cross thresholds - it tells you when intervention is needed. For example, I'd monitor CPU usage continuously in a dashboard, but alert only when it's >90% for 10 minutes and causing request queuing. The key is alerting on symptoms that affect users, not every internal metric. I'd alert on 'error rate >5%' rather than 'cache miss rate high' because the former directly impacts users. Monitoring provides visibility for debugging and capacity planning, while alerting drives immediate action for incidents."

---

## Related Topics

- [Monitoring](monitoring.md) - Collect metrics for alerts
- [Logging](logging.md) - Debug alert conditions
- [Tracing](tracing.md) - Investigate alert root causes

---

**Alert smart, not often! 🚨**
