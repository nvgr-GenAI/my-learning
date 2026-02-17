# Deployment Strategies

Deployment strategies govern how new versions of software reach production users. The
choice of strategy reflects a fundamental trade-off: speed of delivery versus risk of
failure. A payment processor serving millions of transactions cannot tolerate the same
risk profile as an internal dashboard used by a dozen engineers, so the deployment
approach must match the criticality and scale of the system.

Every strategy answers three questions. First, how much production traffic sees the new
version before full rollout? Second, how quickly can the system revert if something
breaks? Third, what infrastructure overhead does the approach demand?

## Strategy Comparison

| Strategy | Risk Level | Rollout Speed | Infra Cost | Rollback Speed | Best For |
|----------------|-----------|---------------|------------|----------------|--------------------------|
| Recreate | High | Instant | 1x | Minutes | Dev and staging only |
| Rolling | Medium | Minutes-hours | 1x | Slow (minutes) | Standard web services |
| Blue-Green | Low | Instant switch | 2x | Instant | Mission-critical systems |
| Canary | Very Low | Hours-days | ~1.1x | Seconds | High-scale user-facing |
| Shadow/Dark | Minimal | N/A (no users) | ~2x | N/A | Validating risky changes |

---

=== "Rolling and Blue-Green"

    ## Rolling Updates

    A rolling deployment replaces instances of the old version one at a time (or in small
    batches) while the remaining instances continue serving traffic. The load balancer
    drains connections from each instance before it is taken down, deploys the new version,
    waits for a health check to pass, then adds it back to the pool.

    The two key parameters are **maxUnavailable** (how many instances can be down at once)
    and **maxSurge** (how many extra instances can exist during the transition). A
    conservative setting of maxUnavailable=1, maxSurge=0 means only one instance is
    replaced at a time with no extra capacity, which is slow but safe. An aggressive
    setting of maxUnavailable=25%, maxSurge=25% speeds things up but temporarily reduces
    capacity or increases cost.

    ```
    Time -->

    Instance 1:  [--- v1.0 ---][drain][--- v2.0 --->
    Instance 2:  [--- v1.0 ----------][drain][--- v2.0 --->
    Instance 3:  [--- v1.0 -------------------][drain][--- v2.0 --->
    Instance 4:  [--- v1.0 ----------------------------][drain][--- v2.0 --->

    Load Balancer routes only to healthy instances throughout.
    Both versions coexist during the transition window.
    ```

    The critical requirement for rolling updates is **backward compatibility between v1 and
    v2**. During the rollout window, both versions serve traffic simultaneously and may
    share the same database. If v2 introduces a schema change that v1 cannot read, users
    hitting v1 instances will see errors. Kubernetes uses rolling updates as the default
    strategy, and most cloud-managed services (ECS, Cloud Run) follow the same pattern.

    LinkedIn uses rolling deployments across its thousands of microservices, relying on
    health checks and readiness probes to gate each step. With roughly 2,000 production
    deployments per day, the rolling approach keeps resource costs flat while maintaining
    zero-downtime releases for non-critical services.

    ## Blue-Green Deployments

    Blue-green maintains two identical production environments. At any time, one (say
    "blue") handles all live traffic while the other ("green") sits idle or runs the new
    version for testing. After the green environment passes smoke tests and load tests, the
    load balancer or DNS is switched so green receives 100% of traffic. Blue remains
    running as an instant rollback target.

    ```
    BEFORE SWITCH
    +-------------------+
    |   Load Balancer   |-----> 100% traffic
    +-------------------+
              |
              v
    +---------+---------+         +-------------------+
    |   BLUE (v1.0)     |         |  GREEN (v2.0)     |
    |   3 instances     |         |  3 instances      |
    |   serving traffic |         |  smoke testing    |
    +-------------------+         +-------------------+

    AFTER SWITCH (single routing change)
    +-------------------+
    |   Load Balancer   |-----> 100% traffic
    +-------------------+
              |
              v
    +---------+---------+         +-------------------+
    |   BLUE (v1.0)     |         |  GREEN (v2.0)     |
    |   idle / rollback |         |  serving traffic  |
    +-------------------+         +-------------------+
    ```

    The defining advantage is **instant rollback**. If green shows elevated errors after
    the switch, flipping the load balancer back to blue restores the previous version in
    seconds, not minutes. The trade-off is cost: you maintain double the infrastructure
    during and often between deployments.

    **AWS CodeDeploy Blue-Green Example.** AWS CodeDeploy automates blue-green for ECS and
    EC2. It provisions a replacement task set (green), runs health checks against the new
    target group, then shifts the ALB listener from the original target group to the
    replacement. A configurable "bake time" (for example, 10 minutes) keeps the original
    target group alive after the switch. If CloudWatch alarms fire during the bake period,
    CodeDeploy automatically rolls back by re-routing the listener to the original target
    group.

    Etsy adopted blue-green deployments for its core marketplace application, deploying to
    a standby fleet, validating with automated tests, then switching the load balancer.
    This approach lets them deploy 50+ times per day to a platform handling over 130
    million monthly visitors while keeping rollback times under 30 seconds.

    ## Zero-Downtime Deployment Requirements

    Both rolling and blue-green strategies achieve zero downtime only when certain
    preconditions are met. First, the application must support **graceful shutdown** --
    in-flight requests must complete before the process exits. Second, **health checks**
    must accurately reflect readiness (not just liveness). A container that is running but
    still loading a large ML model into memory should fail readiness checks until the model
    is loaded. Third, **database migrations must be backward-compatible**. The expand-
    contract pattern (add a nullable column, deploy code that writes to both, backfill,
    then drop the old column) prevents breakage during the transition window.

    ```
    # Pseudo-config for rolling update parameters
    strategy: RollingUpdate
    maxUnavailable: 2       # at most 2 pods down
    maxSurge: 1             # at most 1 extra pod
    readinessProbe:
      path: /health
      interval: 5s
      threshold: 3
    ```

=== "Canary and Progressive"

    ## Canary Deployment

    Canary deployment routes a small percentage of production traffic to the new version
    while the vast majority continues hitting the stable release. The name comes from the
    coal mining practice of sending a canary into the mine to detect toxic gas -- if the
    small canary population (the new version) shows distress, the full rollout is halted
    before it reaches all users.

    A typical canary progression looks like this:

    ```
    Stage 1: 5% canary
    +-------------------+
    |   Load Balancer   |
    +--------+----------+
             |
        +----+----+
        |         |
       95%       5%
        |         |
        v         v
    +-------+ +-------+
    | v1.0  | | v2.0  |  <-- monitor error rate, latency, CPU
    | fleet | | canary|
    +-------+ +-------+

    Stage 2 (if healthy): 25%    [monitor 10-30 min]
    Stage 3 (if healthy): 50%    [monitor 10-30 min]
    Stage 4 (if healthy): 100%   [full rollout]

    At ANY stage: if metrics degrade --> route 100% back to v1.0
    ```

    The key to canary deployment is **automated analysis between stages**. Rather than
    having an engineer manually check dashboards, modern canary systems compare the error
    rate, latency percentiles, and resource consumption of the canary population against
    the baseline (v1) population. If the canary's P99 latency is more than 10% higher or
    its error rate exceeds the baseline by any measurable amount, the system automatically
    halts the rollout and shifts all traffic back.

    **Netflix and Kayenta.** Netflix open-sourced Kayenta, their automated canary analysis
    tool, which runs statistical comparisons between canary and baseline metrics. When a
    Netflix engineer deploys a new version of a microservice (across a fleet serving 200+
    million subscribers), Spinnaker orchestrates a canary that sends roughly 5% of traffic
    to the new version. Kayenta collects metrics from both populations over a configured
    window (typically 30 minutes to 3 hours), applies a Mann-Whitney U test, and produces
    a pass/fail score. If the canary fails, Spinnaker rolls back automatically without
    human intervention. Netflix processes over 1 million requests per second across its
    microservices, so even a 5% canary represents significant production validation.

    ## Feature Flags as Deployment Mechanism

    Feature flags decouple deployment from release. The new code ships to all instances
    simultaneously, but a flag controls whether users actually see the new behavior. This
    allows canary-like progressive rollouts at the application layer rather than the
    infrastructure layer.

    ```
    # Pseudo-code: feature flag progressive rollout
    if feature_flags.enabled("new_checkout", user_id):
        return new_checkout_flow(request)
    else:
        return old_checkout_flow(request)
    ```

    The flag can be configured to enable the feature for 5% of users, then 25%, then 100%,
    exactly like an infrastructure-level canary. The advantage is that rollback is instant
    (flip the flag) and requires no infrastructure changes. The disadvantage is that the
    codebase accumulates conditional branches that must eventually be cleaned up.

    Google, Facebook, and LinkedIn all use feature flags extensively. LinkedIn's feature
    flag system (called "XLNT") manages over 15,000 active flags at any time, enabling
    engineers to ship code daily and gradually roll out features to their 900+ million
    members independently of deployment timing.

    ## Monitoring Gates Between Stages

    Progressive delivery systems place automated "gates" between each traffic increase.
    A gate is a set of conditions that must be satisfied before the rollout continues.
    Common gate conditions include:

    | Metric | Threshold Example | Rationale |
    |---------------------|-------------------------------|--------------------------------------|
    | Error rate (5xx) | Canary <= baseline + 0.1% | Catches regressions in correctness |
    | P99 latency | Canary <= baseline + 50ms | Catches performance regressions |
    | CPU utilization | Canary <= baseline + 20% | Catches resource leaks |
    | Business metric | Conversion rate >= baseline | Catches functional regressions |
    | Saturation | Memory < 80% of limit | Catches memory leaks early |

    If any gate fails, the system either pauses (waiting for human review) or
    automatically rolls back. Argo Rollouts and Flagger are two Kubernetes-native tools
    that implement this pattern, integrating with Prometheus, Datadog, or New Relic for
    metric evaluation.

=== "Other Strategies"

    ## A/B Testing Deployments

    A/B testing deployments route users to different versions based on user attributes
    rather than random traffic splitting. While canary deployments optimize for risk
    reduction (gradually increasing traffic), A/B testing optimizes for **measuring the
    impact of a change** on a specific metric, such as conversion rate or engagement.

    The traffic split is typically 50/50 between control (version A) and experiment
    (version B), and users are assigned deterministically -- the same user always sees
    the same version for the duration of the experiment. This consistency is achieved
    through hashing the user ID, which ensures that refreshing the page or returning
    later does not change the experience.

    ```
    User request arrives
           |
    +------+------+
    | Hash user ID |
    | mod 100      |
    +------+------+
           |
      +----+----+
      |         |
    0-49      50-99
      |         |
      v         v
    +-----+  +-----+
    | v1  |  | v2  |   Track: conversion rate, revenue,
    | (A) |  | (B) |   click-through, session duration
    +-----+  +-----+
    ```

    The critical difference from canary is intent. A canary asks "is the new version
    safe?" while an A/B test asks "is the new version better?" A/B tests typically run
    for days or weeks to achieve statistical significance, whereas canary deployments
    complete in hours. Companies like Booking.com run thousands of A/B tests concurrently,
    with their experimentation platform routing different user segments to different
    feature variants across their 200+ microservices.

    ## Shadow (Dark) Launches

    Shadow deployment sends a copy of production traffic to the new version without
    returning its responses to users. The new version processes real requests, and its
    outputs are compared against the production version's outputs, but only the production
    version's responses reach users.

    ```
    User request
         |
         v
    +-----------+
    |   Router  |----> v1.0 (production) ----> response to user
    |           |
    |           +----> v2.0 (shadow) ----> response discarded
    +-----------+                          (logged for comparison)
    ```

    This approach is particularly valuable for changes that are difficult to test
    synthetically, such as ML model replacements, search ranking algorithm changes, or
    database migration validation. Twitter used shadow traffic extensively when migrating
    from Ruby on Rails to their JVM-based stack, replaying production traffic against both
    systems and comparing response correctness and latency before cutting over.

    The main limitation is that shadow deployments only work for **read operations**.
    Shadowing write operations would cause duplicate data creation, double charges, or
    duplicate notifications. Systems that shadow writes must use a mechanism to absorb or
    discard the side effects.

    ## Recreate Deployment

    The recreate strategy stops all running instances of the old version, then starts
    instances of the new version. This causes downtime proportional to the application's
    startup time (typically 30 seconds to several minutes). It is the simplest strategy
    and appropriate only for development environments, batch processing systems, or
    applications where brief downtime windows are acceptable.

    ## Comparison Summary

    | Dimension | A/B Testing | Shadow/Dark | Canary | Blue-Green |
    |----------------------|-------------------|-----------------|----------------|----------------|
    | Traffic to new ver. | 50% (users) | 100% (mirrored) | 5-100% (grad.) | 0% then 100% |
    | User sees new ver.? | Yes (segment) | No | Yes (subset) | Yes (all) |
    | Primary goal | Measure impact | Validate safety | Reduce risk | Fast switchover|
    | Duration | Days-weeks | Hours-days | Hours | Minutes |
    | Rollback mechanism | Remove from test | Stop mirroring | Shift traffic | Switch LB |

=== "Rollback"

    ## Automated Rollback Triggers

    Automated rollback removes human reaction time from incident response. Instead of
    waiting for an engineer to notice elevated errors and manually revert, the deployment
    system monitors key metrics and initiates rollback when thresholds are breached.

    The most common triggers are:

    **Error rate spike.** If the 5xx error rate exceeds a threshold (such as 1% of
    requests over a 2-minute window), the deployment system reverts. This catches crashes,
    unhandled exceptions, and configuration errors.

    **Latency degradation.** If P99 latency increases beyond a threshold (for example,
    more than 200ms above baseline), the system reverts. Latency regressions often indicate
    inefficient queries, missing indexes, or resource contention introduced by the new
    version.

    **Health check failures.** If newly deployed instances fail readiness or liveness
    probes beyond the configured threshold, the orchestrator (Kubernetes, ECS) stops the
    rollout and reverts to the last known-good state.

    **Business metric anomalies.** Some organizations monitor business metrics like order
    completion rate or search result click-through. A sudden drop in these metrics after
    deployment triggers investigation or automatic rollback.

    ```
    Deployment starts
         |
         v
    [Deploy v2.0] ---> [Monitor 5 min] ---> Metrics OK? ---> Continue
                                                  |
                                                  No
                                                  |
                                                  v
                                          [Rollback to v1.0]
                                          [Alert on-call eng]
                                          [Log rollback reason]
    ```

    ## Database Migration Rollback Challenges

    Application rollback is straightforward (revert the binary), but database migration
    rollback is fundamentally harder. Once a migration runs (dropping a column, renaming
    a table, changing data types), the previous application version may not function
    against the new schema.

    The expand-contract pattern addresses this by splitting breaking changes into multiple
    safe steps:

    | Phase | Migration Action | App Version | Rollback Safe? |
    |-------|-------------------------------|-------------|----------------|
    | 1 | Add new column (nullable) | v1 (ignores it) | Yes |
    | 2 | Deploy v2 (writes to both) | v2 | Yes (v1 still works) |
    | 3 | Backfill old rows | v2 | Yes |
    | 4 | Drop old column | v2 only | No (point of no return) |

    Phase 4 is the "point of no return." Before executing it, teams typically wait several
    days to confirm v2 is stable. If a rollback to v1 is needed before phase 4, the
    database is still compatible. After phase 4, rolling back requires a new forward
    migration to restore the old schema.

    ## GitOps Rollback

    In a GitOps workflow, the desired state of production is defined in a Git repository.
    Deployments happen by merging pull requests that update image tags or Helm values.
    Rollback in this model is simply reverting the commit, which triggers the GitOps
    operator (ArgoCD, Flux) to reconcile the cluster back to the previous state.

    ```
    # GitOps rollback is a single git operation
    git revert abc123    # revert the deployment commit
    git push             # ArgoCD detects change, syncs cluster
    ```

    This approach provides a complete audit trail (every deployment is a commit), enables
    peer review of deployments (pull request approvals), and makes rollback as simple as
    any other code change. ArgoCD reports that organizations using GitOps reduce their
    mean time to recovery (MTTR) by 50-75% compared to manual deployment processes.

    ## Facebook's Deployment Pipeline

    Facebook (Meta) deploys code to its 2+ billion user platform through a multi-stage
    pipeline that combines canary, progressive rollout, and automated rollback. A typical
    deployment proceeds through these stages:

    1. **Internal dogfooding.** The new version runs on internal-facing servers used by
       Facebook employees. Engineers use the product normally, and automated tests run
       against the internal deployment for several hours.

    2. **Small canary (2%).** A small percentage of production traffic shifts to the new
       version. Automated systems compare error rates, latency, and engagement metrics
       between the canary and control populations.

    3. **Staged rollout (25%, 50%, 100%).** If the canary is healthy, traffic gradually
       increases with monitoring gates at each stage. The entire progression from 2% to
       100% takes 6-12 hours.

    4. **Automated rollback.** If any stage shows metric degradation beyond configured
       thresholds, the system automatically reverts to the previous version across all
       affected servers. Facebook's deployment system executes hundreds of automatic
       rollbacks per month, catching issues that would otherwise reach all users.

    This pipeline enables Facebook to ship code twice daily to production while maintaining
    reliability at a scale of over 10 billion daily content interactions.

---

## Key Takeaways

**Match strategy to risk tolerance.** Use rolling updates for routine services where
brief version coexistence is acceptable. Use blue-green for critical systems where instant
rollback justifies the infrastructure cost. Use canary for high-scale services where
gradual validation with real traffic is essential.

**Automate rollback decisions.** Human reaction time during incidents is measured in
minutes; automated rollback systems react in seconds. Define clear metric thresholds
(error rate, latency, business KPIs) and let the deployment system enforce them.

**Database migrations are the hard part.** Application binaries can be swapped
instantly, but schema changes are difficult to reverse. The expand-contract pattern and
phased migrations keep the database compatible with both old and new application versions
throughout the deployment window.

**Progressive delivery is the industry trend.** Netflix, Facebook, Google, and LinkedIn
all use some form of canary or progressive rollout with automated analysis. The tooling
(Spinnaker, ArgoCD, Flagger, LaunchDarkly) has matured to the point where progressive
delivery is accessible to teams of any size.

---

## Related Topics

- [CI/CD Pipelines](ci-cd.md) -- automating the deployment process end to end
- [Containers and Orchestration](containers.md) -- packaging applications for deployment
- [Monitoring and Observability](../observability/monitoring.md) -- tracking deployment health
- [Load Balancing](../networking/load-balancers.md) -- traffic routing during deployments
