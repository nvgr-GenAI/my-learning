# Load Balancing for Scalability

This page covers load balancing from the scalability perspective — how load balancers enable horizontal scaling, auto-scaling integration, and traffic management during scaling events. For a deep dive on load balancing algorithms, health checks, and technologies, see the comprehensive [Load Balancers](../networking/load-balancers.md) guide in the Networking section.

---

## Load Balancing as a Scaling Enabler

A load balancer is what makes horizontal scaling possible. Without it, adding more servers doesn't help — clients still point at one machine. The load balancer is the single entry point that distributes traffic across however many backend servers currently exist.

```
Single server (no LB):                With load balancer:

Client ──→ Server                     Client ──→ LB ──→ Server 1
                                                   ├──→ Server 2
Can't scale. Can't survive             ├──→ Server 3
a failure. Can't deploy                └──→ Server N
without downtime.
                                      Scale by changing N.
                                      Survive by removing failed servers.
                                      Deploy by rolling through servers.
```

The load balancer provides three capabilities essential for scalability:

1. **Traffic distribution** — spread requests across available servers
2. **Health-aware routing** — stop sending traffic to failed or degraded servers
3. **Dynamic membership** — add and remove servers without client-side changes

---

=== "Auto-Scaling Integration"

    ## Auto-Scaling Integration

    In a cloud environment, the load balancer works hand-in-hand with the auto-scaler. The auto-scaler decides *how many* instances to run; the load balancer decides *which instance* gets each request.

    ```
                       Metrics
                         │
                         ▼
                  ┌─────────────┐
                  │ Auto-Scaler │
                  │             │
                  │ CPU > 70%?  │──→ Launch new instance
                  │ CPU < 30%?  │──→ Terminate instance
                  └──────┬──────┘
                         │
                  Register/deregister
                         │
                         ▼
                  ┌─────────────┐
                  │Load Balancer│──→ Route traffic to healthy instances
                  └─────────────┘
    ```

    ### Scaling Up (Adding Instances)

    When the auto-scaler launches a new instance:

    1. Instance starts and passes health checks
    2. Auto-scaler registers instance with the load balancer
    3. Load balancer begins routing traffic to the new instance
    4. **Best practice:** Use a warm-up period — gradually increase traffic to the new instance so it can warm its caches and JIT-compile hot paths

    ### Scaling Down (Removing Instances)

    When the auto-scaler terminates an instance:

    1. Auto-scaler deregisters instance from the load balancer
    2. Load balancer stops sending *new* requests to the instance
    3. **Connection draining:** Existing in-flight requests are allowed to complete (typically 30-60 seconds)
    4. Instance terminates after drain period

    Connection draining is critical — without it, active requests are dropped, causing errors for users.

=== "Deployment Strategies"

    ## Zero-Downtime Deployments

    Load balancers enable deployment strategies that keep the system available throughout:

    ### Rolling Deployment

    Update servers one at a time. Remove server from load balancer, deploy new version, pass health checks, add back.

    ```
    Time 0:  LB ──→ [v1] [v1] [v1] [v1]    (all on v1)
    Time 1:  LB ──→ [v2] [v1] [v1] [v1]    (1 updated)
    Time 2:  LB ──→ [v2] [v2] [v1] [v1]    (2 updated)
    Time 3:  LB ──→ [v2] [v2] [v2] [v1]    (3 updated)
    Time 4:  LB ──→ [v2] [v2] [v2] [v2]    (all on v2)
    ```

    **Advantage:** Simple, uses existing capacity. **Drawback:** Both versions serve traffic simultaneously during rollout — ensure backward compatibility.

    ### Blue-Green Deployment

    Run two identical environments. Route all traffic from the old version (blue) to the new version (green) at once.

    ```
    Before:  LB ──→ [Blue: v1, v1, v1]    (100% traffic)
                    [Green: v2, v2, v2]    (0% traffic, fully tested)

    Switch:  LB ──→ [Green: v2, v2, v2]   (100% traffic)
                    [Blue: v1, v1, v1]     (standby for rollback)
    ```

    **Advantage:** Instant rollback (switch back to blue). **Drawback:** Requires double the infrastructure during deployment.

    ### Canary Deployment

    Route a small percentage of traffic to the new version. Monitor for errors. Gradually increase if healthy.

    ```
    Phase 1:  LB ──→ 95% [v1, v1, v1]  +  5% [v2]
    Phase 2:  LB ──→ 75% [v1, v1]      + 25% [v2]
    Phase 3:  LB ──→ 50% [v1]          + 50% [v2, v2]
    Phase 4:  LB ──→                     100% [v2, v2, v2]
    ```

    **Advantage:** Catches issues with minimal user impact. **Drawback:** Requires traffic splitting support and good monitoring to detect issues quickly.

    **Netflix** uses canary deployments extensively — new versions serve a small percentage of traffic for hours or days before full rollout. If error rates increase, the canary is automatically rolled back.

=== "Multi-Tier Load Balancing"

    ## Multi-Tier Load Balancing

    At scale, a single load balancer isn't enough. Traffic flows through multiple layers:

    ```
    Internet
       │
       ▼
    DNS Load Balancing (geographic routing)
       │
       ├──→ US Region                    ├──→ EU Region
       │    │                            │    │
       │    ▼                            │    ▼
       │   L4 Load Balancer (NLB)       │   L4 Load Balancer
       │    │                            │    │
       │    ├──→ L7 LB (ALB) ──→ API   │    ├──→ L7 LB ──→ API
       │    └──→ L7 LB (ALB) ──→ Web   │    └──→ L7 LB ──→ Web
    ```

    **Layer 1 (DNS):** Geographic routing. Send US users to US servers, EU users to EU servers. Coarse-grained but reduces latency.

    **Layer 2 (L4):** High-throughput TCP/UDP distribution. Handles millions of connections per second with minimal latency. Distributes across L7 load balancers.

    **Layer 3 (L7):** Content-aware routing. Routes `/api/*` to API servers, `/static/*` to CDN, `/ws/*` to WebSocket servers.

---

## Key Takeaways

1. **Load balancers are the foundation of horizontal scaling.** Without one, you can't distribute traffic, can't survive failures, and can't deploy without downtime.

2. **Connection draining prevents errors during scale-down.** Always allow in-flight requests to complete before terminating instances.

3. **Use canary deployments for safety.** Route small traffic percentages to new versions and monitor before full rollout.

4. **Multi-tier load balancing separates concerns.** DNS for geography, L4 for throughput, L7 for content-aware routing.

---

## Related Topics

- **[Load Balancers](../networking/load-balancers.md)** — algorithms, health checks, and technology deep dive
- **[Horizontal Scaling](horizontal-scaling.md)** — stateless design and auto-scaling patterns
- **[Scalability Patterns](patterns.md)** — the full scaling journey
- **[Proxies](../networking/proxies.md)** — reverse proxy patterns that complement load balancing
