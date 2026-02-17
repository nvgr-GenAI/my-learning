# Alerting

Monitoring tells you what's happening. Alerting tells you when to act. The difference matters: you don't need a human looking at dashboards 24/7 if the system can detect problems and notify the right person automatically. But poorly designed alerting is worse than no alerting — alert fatigue causes engineers to ignore notifications, and the one critical alert gets lost in the noise.

The goal of alerting is simple: **every alert should require human action, and every action-worthy event should trigger an alert.** Nothing more, nothing less.

---

=== "Design Principles"

    ## Alert Design Principles

    ### Alert on Symptoms, Not Causes

    ```
    Bad (alerting on causes):
      "Redis CPU is at 90%"
      → Maybe that's normal. Maybe it's a problem. What should I do?

    Good (alerting on symptoms):
      "Cache hit rate dropped below 50%, API latency increased 3x"
      → Users are affected. I need to investigate.
    ```

    The Redis CPU alert might fire during a normal cache warm-up. The symptom-based alert fires only when users are actually impacted. Alert on what matters to users — latency, errors, availability — not on internal system metrics that might or might not affect the user experience.

    ### Every Alert Must Be Actionable

    If an alert fires and the on-call engineer can't do anything about it, it shouldn't be an alert. It should be a dashboard metric, a log entry, or an automated remediation — but not a notification that wakes someone up.

    For each alert, ask:
    - What action should the responder take?
    - Is there a runbook for this alert?
    - Can this be automated instead?

    ---

    ## Severity Levels

    Not all problems are equally urgent. A clear severity model prevents unnecessary pages at 3 AM.

    ```
    P0 — CRITICAL (page immediately, any hour)
      Complete outage, data loss in progress, security breach
      Response: within 5 minutes
      Examples: all API requests failing, database unreachable, data corruption
      Escalation: manager + VP if not resolved in 30 minutes

    P1 — HIGH (page on-call engineer)
      Partial degradation, high error rate, key feature broken
      Response: within 15 minutes
      Examples: error rate >5%, payment processing down, replication lag >10min
      Escalation: manager if not resolved in 1 hour

    P2 — MEDIUM (ticket during business hours)
      Non-critical issues, performance degradation, elevated error rates
      Response: same business day
      Examples: error rate 1-5%, slow queries increasing, non-critical service down

    P3 — LOW (ticket, low priority)
      Warnings about future problems, minor issues
      Response: this week
      Examples: certificate expiring in 30 days, disk at 80%, deprecation warnings
    ```

    **PagerDuty** reports that organizations with well-defined severity levels have 40% faster mean time to resolution (MTTR) compared to those without.

=== "SLO-Based Alerting"

    ## SLO-Based Alerting

    The most effective alerting approach: alert when you're **burning through your error budget** too fast.

    ```
    SLO: 99.9% success rate
    Error budget: 0.1% = ~43 minutes of downtime per month

    Normal:
      ─────────────────── Budget: 43 min remaining
      Day 1    Day 15    Day 30
      (budget consumed slowly, everything fine)

    Fast burn (page immediately):
      ──────\
             \────────── Budget will exhaust in <2 days at current rate
      Day 1    Day 3
      (something is very wrong right now)

    Slow burn (warning):
      ──────────\
                  \───── Budget will exhaust this month at current rate
      Day 1    Day 15   Day 30
      (degradation that needs attention)
    ```

    ### Multi-Window Burn Rate

    The industry best practice uses two time windows to reduce false positives:

    | Alert | Short Window | Long Window | Severity |
    |---|---|---|---|
    | **Fast burn** | 5-minute error rate exceeds budget | 1-hour error rate exceeds budget | Critical (page) |
    | **Slow burn** | 30-minute error rate exceeds budget | 6-hour error rate exceeds budget | Warning (ticket) |

    Both windows must be in violation before the alert fires. This prevents paging for brief spikes (short window fires but long window is fine) and catches real degradation (both windows consistently bad).

    **Google** uses multi-window, multi-burn-rate alerting across their services. This approach is documented in their SRE book and has been adopted widely — by **Spotify**, **Shopify**, and many others.

=== "Routing & Fatigue"

    ## Alert Routing and Escalation

    Alerts need to reach the right person through the right channel at the right time.

    ```
    Alert fires
        │
        ├─ P0/P1 → PagerDuty → phone call to on-call engineer
        │              │
        │              └─ Not acknowledged in 5min → escalate to backup
        │                     │
        │                     └─ Not resolved in 30min → escalate to manager
        │
        ├─ P2 → Slack #alerts-warnings → on-call reviews during business hours
        │
        └─ P3 → Jira ticket → team backlog
    ```

    ### Inhibition Rules

    When a server is down, don't also alert on "high CPU," "disk space," and "memory usage" for that server. **Inhibition** suppresses downstream alerts when a root cause alert is already firing.

    Common inhibitions:
    - Service down → suppress latency and error rate alerts for that service
    - Node down → suppress all resource alerts for that node
    - Network partition → suppress cross-service dependency alerts

    ### Alert Grouping

    When 50 pods fail simultaneously, you want one alert "50 pods in service X are down," not 50 individual alerts. Group alerts by service, cluster, and alert type. Send one notification with a count, not a flood of identical pages.

    ---

    ## Alert Fatigue

    Alert fatigue is the #1 failure mode of alerting systems. Engineers receive so many alerts that they start ignoring them — and then miss the one that matters.

    ### Symptoms of Alert Fatigue

    - Engineers acknowledge alerts without investigating
    - Alert channels are muted or filtered
    - P0 alerts are treated the same as P3
    - On-call rotations are dreaded and cause burnout

    ### Prevention Strategies

    ```
    1. Audit regularly
       Review every alert monthly:
       → Was it actionable? If not, remove it.
       → Did it fire unnecessarily? If so, tune the threshold.
       → Can it be automated? If so, automate the response.

    2. Track alert metrics
       - Alerts per day per engineer (target: <5)
       - False positive rate (target: <10%)
       - Mean time to acknowledge
       - Mean time to resolve
       - Percentage of alerts auto-resolved without action

    3. Use dynamic thresholds
       Static: "alert if latency > 500ms"
       → Fires at 3 AM when traffic is low and 500ms is normal

       Dynamic: "alert if latency is 3x the historical average for this time"
       → Adapts to traffic patterns, fewer false positives

    4. Silence during maintenance
       Scheduled deploys, infrastructure upgrades, and known issues
       should have silence windows — but with expiration so they
       don't silently mask real problems.
    ```

    **PagerDuty** data shows that teams with more than 40 alerts per week have significantly higher MTTR and lower engineer satisfaction than teams with fewer than 10 alerts per week.

=== "Operations"

    ## Runbooks

    Every alert should link to a runbook — a document that tells the responder exactly what to do.

    A good runbook contains:

    ```
    Alert: High Error Rate (>5%)

    Impact:
      Users cannot complete purchases. Revenue impact: ~$X/minute.

    Diagnosis:
      1. Check dashboard: [link to Grafana]
      2. Check recent deploys: did something just ship?
      3. Check external dependencies: [Stripe status page], [AWS status]
      4. Review error logs: search for requestId in [Kibana link]

    Resolution:
      If bad deploy → rollback: kubectl rollout undo deployment/api
      If dependency down → enable circuit breaker, notify users
      If database issue → check connection pool, review slow queries

    Escalation:
      15 minutes → tech lead
      30 minutes → engineering director
    ```

    **Runbooks should be living documents.** After every incident, update the relevant runbook with what you learned. Over time, runbooks become increasingly accurate and reduce resolution time.

    ---

    ## On-Call Best Practices

    Alerting only works if someone is listening. On-call rotations are how teams ensure 24/7 coverage.

    **Rotation structure:** Weekly rotations with primary and secondary on-call. No individual should be on-call more than 2 weeks per quarter. Follow-the-sun for global teams to minimize overnight pages.

    **Handoff:** Document ongoing issues, share context from the previous week, and verify the incoming engineer has all necessary access (VPN, dashboards, deploy tools).

    **Compensation:** On-call duty should be compensated — whether through additional pay, comp time, or flexibility. **Google**, **Meta**, and most tech companies provide on-call compensation.

    **Post-incident:** After every significant incident, conduct a blameless post-mortem. What happened? Why? What can prevent it next time? Update runbooks, improve alerts, and address systemic issues.

    ---

    ## Alerting Tools

    | Tool | Type | Strengths |
    |---|---|---|
    | **Prometheus AlertManager** | Open-source | Powerful routing, grouping, inhibition. Standard for Prometheus stacks |
    | **PagerDuty** | SaaS | Sophisticated escalation, scheduling, analytics. Industry standard for on-call |
    | **Opsgenie** (Atlassian) | SaaS | Jira integration, team-based routing |
    | **AWS CloudWatch Alarms** | Cloud-native | Native AWS integration, anomaly detection |
    | **Datadog Monitors** | SaaS | Unified with metrics/logs/traces, ML-powered anomaly detection |

---

## Key Takeaways

1. **Alert on symptoms, not causes.** Users care about latency and errors, not CPU usage. Alert on what affects the user experience.

2. **SLO-based alerting is the gold standard.** Multi-window burn rate alerting catches real degradation while filtering noise. Alert when you're burning through your error budget too fast.

3. **Every alert needs a runbook.** An alert without a runbook is just an annoyance. Runbooks convert alerts into action.

4. **Fight alert fatigue relentlessly.** Audit alerts monthly. Track false positive rates. Remove or automate noisy alerts. Target fewer than 5 actionable alerts per engineer per day.

5. **Route by severity.** P0 pages immediately. P2 creates a ticket. Mixing these guarantees fatigue or missed critical issues.

6. **On-call is a team responsibility.** Compensate it, support it, and continuously improve it through post-incident reviews and runbook updates.

---

## Related Topics

- **[Monitoring](monitoring.md)** — collecting the metrics that drive alerts
- **[Logging](logging.md)** — debugging what alerts reveal
- **[Tracing](tracing.md)** — pinpointing issues across services
