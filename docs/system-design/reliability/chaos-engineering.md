# Chaos Engineering

Your system will fail in production. The only question is whether you discover how it
fails on your terms or your customers' terms. Chaos engineering is the discipline of
proactively injecting failures into production (or production-like) systems to build
confidence that the system can withstand real-world turbulence.

The core idea is counterintuitive: **deliberately break things to make them stronger.**
By regularly inducing server crashes, network partitions, latency spikes, and resource
exhaustion in controlled experiments, you discover weaknesses before they cause outages.
Netflix, the pioneer of chaos engineering, found that without regular failure injection,
engineers unconsciously assumed components were reliable — and were caught off guard
when they weren't.

---

## The Principles of Chaos Engineering

Chaos engineering is not randomly breaking things. It follows a scientific method:

```
CHAOS EXPERIMENT WORKFLOW

1. STEADY STATE        Define what "normal" looks like
   ↓                   (error rate < 0.1%, P99 latency < 200ms)
2. HYPOTHESIS          "If we kill 1 of 3 database replicas,
   ↓                    the system continues serving reads"
3. INJECT FAILURE      Kill the replica
   ↓
4. OBSERVE             Monitor dashboards, alerts, user impact
   ↓
5. ANALYZE             Did the system behave as hypothesized?
   │
   ├── YES → Confidence increased. Document and expand scope.
   └── NO  → Found a weakness! Fix it, then re-run experiment.
```

**Key principles:**

1. **Build a hypothesis around steady-state behavior.** Define measurable indicators of
   normal system behavior (request rate, error rate, latency percentiles). The
   experiment tests whether these indicators remain within bounds during failure.

2. **Vary real-world events.** Inject failures that actually happen: server crashes,
   network partitions, disk full, clock skew, dependency timeouts, traffic spikes.

3. **Run experiments in production.** Staging environments don't replicate production's
   traffic patterns, data volume, or service interactions. The most valuable chaos
   experiments run against production with real traffic.

4. **Automate experiments to run continuously.** A one-time experiment finds today's
   bugs. Continuous chaos catches tomorrow's regressions.

5. **Minimize blast radius.** Start small. Kill one instance, not the whole cluster.
   Affect 1% of traffic, not 100%. Have a kill switch to stop the experiment instantly.

---

=== "Failure Categories"

    ## Types of Failures to Inject

    ### Infrastructure Failures

    ```
    SERVER FAILURES
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Server 1 │ │ Server 2 │ │ Server 3 │
    │    ✅     │ │    💀     │ │    ✅     │
    └──────────┘ └──────────┘ └──────────┘
    Kill a VM or container. Does traffic reroute?
    Does the auto-scaler replace it? How long?
    ```

    | Failure | What You Learn | Common Findings |
    |---|---|---|
    | Kill a server instance | Failover speed, auto-scaling | Health checks too slow, no auto-replacement |
    | Fill disk to 100% | Disk-full handling, log rotation | Services crash instead of degrading gracefully |
    | Exhaust CPU/memory | Resource limit behavior | No resource limits set, noisy neighbors |
    | Kill a container | Kubernetes pod recovery | Readiness probes misconfigured |
    | Terminate an availability zone | Multi-AZ resilience | Services pinned to single AZ |

    ### Network Failures

    ```
    NETWORK PARTITION
    ┌─────────────┐          ┌─────────────┐
    │ Service A   │    ╳╳╳   │ Service B   │
    │ (US-East)   │  network  │ (US-West)   │
    │             │  partition │             │
    └─────────────┘          └─────────────┘
    Can each side continue independently?
    What happens when the partition heals?
    ```

    | Failure | What You Learn | Common Findings |
    |---|---|---|
    | Add 500ms latency between services | Timeout handling | Timeouts not configured, cascade failures |
    | Drop 10% of packets | Retry logic | No retries, or aggressive retries that amplify load |
    | Partition between regions | Split-brain behavior | No partition detection, data divergence |
    | DNS failure | Service discovery resilience | Hard-coded DNS TTLs, no caching |
    | Block a specific port | Dependency isolation | Services have undocumented dependencies |

    ### Application Failures

    | Failure | What You Learn |
    |---|---|
    | Return errors from a dependency | Circuit breaker behavior |
    | Slow down a dependency (add latency) | Timeout and fallback logic |
    | Return malformed responses | Input validation and error handling |
    | Exhaust connection pool | Connection management and backpressure |
    | Inject clock skew | Time-dependent logic correctness |

    ### State Failures

    | Failure | What You Learn |
    |---|---|
    | Kill the database primary | Failover and promotion time |
    | Corrupt a cache | Cache miss handling and thundering herd |
    | Fill the message queue | Backpressure and dead letter handling |
    | Expire all sessions | Re-authentication flow |

=== "Netflix's Approach"

    ## Netflix: The Pioneers

    Netflix built the foundational tools of chaos engineering because they were among the
    first to operate at massive scale on cloud infrastructure (AWS), where failures are
    frequent and unpredictable.

    ### Simian Army

    Netflix's original suite of chaos tools, each named after a primate:

    ```
    THE SIMIAN ARMY

    Chaos Monkey      Randomly kills instances in production
    │                 Runs during business hours so engineers
    │                 are awake to respond
    │
    Chaos Kong        Simulates an entire AWS region going down
    │                 Tests multi-region failover
    │
    Latency Monkey    Injects artificial delays in network calls
    │                 Reveals timeout and retry weaknesses
    │
    Conformity Monkey Finds instances that don't follow best practices
    │                 (no auto-scaling, no health checks)
    │
    Security Monkey   Finds security vulnerabilities and misconfigs
    │                 (open ports, missing encryption)
    │
    Janitor Monkey    Cleans up unused resources
                      (orphaned instances, unattached volumes)
    ```

    **Chaos Monkey** runs continuously during business hours (9am-3pm) so that engineers
    are available to respond if something unexpected happens. Every service at Netflix
    must be designed to survive random instance termination. This single practice
    transformed Netflix's engineering culture: **engineers build resilient systems by
    default because they know Chaos Monkey will test them.**

    **Chaos Kong** simulates the failure of an entire AWS region. Netflix runs active-
    active across three regions (US-East, US-West, EU-West). Chaos Kong redirects all
    traffic away from one region and verifies that the other two absorb the load without
    user impact. This discovered issues like:

    - DNS failover taking 15+ minutes (fixed with faster TTLs)
    - Regional caches not warming fast enough (fixed with pre-warming)
    - Stateful services losing data during region evacuation

    ### Key Lessons from Netflix

    1. **Culture matters more than tools.** Chaos engineering works because Netflix's
       leadership mandated that all services must survive Chaos Monkey. Without executive
       support, teams will opt out.

    2. **Start with Chaos Monkey, not Chaos Kong.** Single-instance failures are the
       most common and the easiest to fix. Region failures are rare and complex.

    3. **The goal is confidence, not breakage.** A successful chaos experiment that
       confirms resilience is just as valuable as one that finds a bug.

=== "Tools"

    ## Chaos Engineering Tools

    | Tool | Environment | Approach | Best For |
    |---|---|---|---|
    | **Chaos Monkey** (Netflix) | Production | Kill instances randomly | EC2/cloud VMs |
    | **Gremlin** | Prod/staging | SaaS platform, UI-driven | Teams wanting managed solution |
    | **Chaos Mesh** | Kubernetes | K8s-native, CRDs | K8s-first organizations |
    | **LitmusChaos** | Kubernetes | CNCF project, experiment hub | Open-source K8s chaos |
    | **Toxiproxy** | Any | TCP proxy for network faults | Testing network failures locally |
    | **AWS Fault Injection Simulator** | AWS | Managed service | AWS-native architectures |

    ### Chaos Mesh (Kubernetes)

    Chaos Mesh is the most popular open-source chaos engineering platform for Kubernetes.
    It uses Custom Resource Definitions (CRDs) to define experiments declaratively:

    ```
    CHAOS MESH EXPERIMENT TYPES

    PodChaos      → Kill pods, container crash
    NetworkChaos  → Latency, packet loss, partition
    IOChaos       → Disk latency, read/write errors
    StressChaos   → CPU and memory stress
    TimeChaos     → Clock skew
    HTTPChaos     → HTTP request/response faults
    DNSChaos      → DNS resolution failures

    Scheduling: run once, cron schedule, or continuous
    Scope: specific pods, namespaces, or labels
    Duration: 30 seconds to hours
    ```

    ### Gremlin (Managed Platform)

    Gremlin provides a commercial SaaS platform for chaos engineering with a web UI,
    team management, and compliance features. It's popular with enterprises that need
    audit trails and approval workflows for chaos experiments.

    **Key feature: Scenarios** — pre-built experiment sequences that test common failure
    modes (region failover, dependency failure, traffic spike) without requiring teams
    to design experiments from scratch.

=== "Running Experiments Safely"

    ## How to Start Safely

    ### Maturity Model

    ```
    CHAOS MATURITY LEVELS

    Level 0: Manual testing
    └── Ad-hoc "let's kill this server and see what happens"
        No hypothesis, no measurement, no automation

    Level 1: Scripted experiments in staging
    └── Written hypothesis, defined steady state
        Run against staging environment
        Manual trigger, manual observation

    Level 2: Automated experiments in staging
    └── Experiments run on schedule
        Automated steady-state verification
        Results tracked over time

    Level 3: Production chaos with guardrails
    └── Experiments run against production
        Blast radius limited (1-5% of traffic)
        Automatic halt if impact exceeds threshold

    Level 4: Continuous chaos in production
    └── Chaos Monkey style — always running
        Part of CI/CD pipeline
        Team culture embraces failure injection
    ```

    **Most teams should start at Level 1** and progress over months.

    ### Blast Radius Control

    ```
    START SMALL, EXPAND GRADUALLY

    Week 1:  Kill 1 pod in staging
    Week 2:  Kill 1 pod in production (non-peak)
    Week 3:  Inject 200ms latency to 5% of production traffic
    Week 4:  Kill a non-critical service instance in production
    Week 8:  Simulate availability zone failure in staging
    Week 12: Simulate AZ failure in production
    Month 6: Regional failover test
    ```

    ### Safety Mechanisms

    Every chaos experiment must have:

    1. **Kill switch** — instantly stop the experiment and revert to normal
    2. **Blast radius limit** — affect only a percentage of traffic or specific instances
    3. **Duration limit** — automatically stop after N minutes
    4. **Steady-state monitors** — halt if error rate exceeds threshold
    5. **Rollback plan** — documented steps to recover if things go wrong
    6. **Communication** — notify on-call engineers before experiments run

---

## Measuring Resilience

Chaos experiments produce measurable outcomes that improve over time:

| Metric | Before Chaos Program | After 6 Months | What It Means |
|---|---|---|---|
| Mean time to detect (MTTD) | 15 minutes | 2 minutes | Faster alerting |
| Mean time to recover (MTTR) | 45 minutes | 10 minutes | Better runbooks and automation |
| Incidents from known failure modes | 12/quarter | 2/quarter | Proactive fixes |
| Services surviving instance kill | 60% | 95% | Better resilience patterns |
| Auto-scaling response time | 8 minutes | 2 minutes | Tuned scaling policies |

**The real ROI:** Chaos engineering's value is in the outages that **don't happen**.
Netflix estimates that their chaos program prevents dozens of potential customer-facing
outages per year. Each major outage at Netflix's scale costs millions in lost revenue
and customer trust.

---

## Key Takeaways

1. **Chaos engineering is a scientific practice, not random destruction.** Every
   experiment starts with a hypothesis, measures steady-state behavior, and produces
   actionable findings.

2. **Start in staging, graduate to production.** Production chaos is the goal because
   staging doesn't replicate real traffic patterns, but you need to build confidence
   and tooling first.

3. **Culture change is the hardest part.** Engineers must believe that finding failures
   proactively is better than discovering them at 3am. Leadership must mandate
   participation — optional chaos programs die quickly.

4. **Begin with the simplest experiment:** kill one instance and verify traffic reroutes.
   If your system can't handle this, more complex experiments are premature.

5. **Automate and run continuously.** A one-time experiment finds today's bugs.
   Continuous chaos catches regressions introduced by new deployments, configuration
   changes, and infrastructure updates.

6. **Always have a kill switch.** The ability to instantly stop an experiment and revert
   is non-negotiable. Without it, a chaos experiment can itself become the outage.

---

## Related Topics

- [Fault Tolerance](fault-tolerance.md) — the patterns that chaos engineering validates
- [Disaster Recovery](disaster-recovery.md) — recovering from large-scale failures
- [Monitoring](../observability/monitoring.md) — the observability that chaos experiments depend on
- [Alerting](../observability/alerting.md) — detecting impact during chaos experiments
