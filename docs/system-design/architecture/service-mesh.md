# Service Mesh Architecture

A service mesh is a dedicated infrastructure layer that handles service-to-service communication within a microservices architecture. Rather than embedding networking logic -- retries, load balancing, encryption, observability -- into each service's application code, a service mesh extracts it into a transparent proxy layer that sits alongside every service instance.

```
Without mesh:                    With mesh:

  Service A -----> Service B       Service A -----> Service B
  (retry logic)   (TLS setup)     (just business   (just business
  (load balance)  (auth check)     logic)            logic)
  (metrics)       (metrics)            |                  |
                                   [Proxy] ---------- [Proxy]
                                   retries, mTLS,     retries, mTLS,
                                   metrics, tracing   metrics, tracing
                                          |
                                    [Control Plane]
                                    config, certs, policy
```

---

=== "Core Concepts"

    ## Sidecar Proxy Pattern

    Every service instance gets a companion proxy deployed alongside it. All traffic is intercepted by the sidecar transparently -- the application sends to `localhost` as if talking directly to the destination.

    ```
    Kubernetes Pod                          Kubernetes Pod
    +-------------------------------+       +-------------------------------+
    |  +----------+  +-----------+  |       |  +-----------+  +----------+  |
    |  |  Order   |->|  Sidecar  |--|--mTLS-|->|  Sidecar  |->| Payment  |  |
    |  | Service  |  |  (Envoy)  |  |       |  |  (Envoy)  |  | Service  |  |
    |  +----------+  +-----------+  |       |  +-----------+  +----------+  |
    +-------------------------------+       +-------------------------------+
    ```

    **Key advantage:** Zero application changes. Java, Python, Go services all get identical networking behavior. Teams adopt incrementally -- enable sidecars one service at a time.

    ### Ambient Mesh (Sidecar-less)

    The sidecar model has a known limitation: every pod runs its own proxy. Istio's **Ambient Mesh** separates the data plane into two layers:

    - **ztunnel** (per-node DaemonSet) -- handles mTLS and L4 traffic for all pods on the node
    - **waypoint proxy** (optional) -- handles L7 features only for services that need them

    | Model | Proxy Instances (500 pods) | Memory Overhead |
    |-------|---------------------------|-----------------|
    | Traditional sidecar | 500 proxies | ~25GB (Istio) |
    | Ambient mesh | ~10 ztunnels + few waypoints | ~2-5GB |

    ## Data Plane vs Control Plane

    | Layer | Role | Handles |
    |-------|------|---------|
    | **Data Plane** | All sidecar proxies across the cluster | Live traffic -- encryption, routing, metrics, policy |
    | **Control Plane** | Brain of the mesh | Config distribution, certificate issuance, telemetry aggregation |

    The control plane can be restarted without disrupting live traffic -- proxies continue operating with their last known config (**control plane resilience**).

    ```
    Control Plane: Config Manager + Cert Authority + Telemetry
         |              |                |
         v              v                v
    Data Plane: [Proxy A] [Proxy B] [Proxy C] [Proxy D] ...
                  |          |          |          |
                [Svc A]   [Svc B]   [Svc C]   [Svc D]
    ```

=== "Key Capabilities"

    ## Traffic Management

    | Capability | What It Does | Example |
    |-----------|-------------|---------|
    | **Canary deployments** | Shift traffic gradually (1% → 10% → 50% → 100%) | Deploy v2, monitor errors, instant rollback |
    | **Header-based routing** | Route by HTTP header values | A/B testing, tenant isolation |
    | **Traffic mirroring** | Copy live traffic to test service | Shadow-test v2 with production load |
    | **Fault injection** | Inject failures/delays deliberately | Chaos testing, resilience validation |
    | **Locality-aware LB** | Prefer same-zone endpoints | Reduce cross-zone latency and cost |

    ```
    Canary Rollout via Mesh Config
    ===============================
    Day 1:  v1 [##########] 100%    v2 [          ]   0%
    Day 2:  v1 [########  ]  80%    v2 [##        ]  20%  ← monitor
    Day 3:  v1 [#####     ]  50%    v2 [#####     ]  50%  ← compare
    Day 5:  v1 [          ]   0%    v2 [##########] 100%  ← done

    Errors spike? → v1 [##########] 100% instantly
    ```

    ## Security (Zero Trust)

    - **Mutual TLS (mTLS):** Every service gets a cryptographic identity (X.509 cert). Both sides verify before exchanging data. Applications send plaintext to localhost -- the mesh encrypts transparently.
    - **Auto cert rotation:** 24-hour TTL certificates rotated automatically. Compromise blast radius = hours, not months.
    - **Authorization policies:** "Only `checkout-service` can call `payment-service`" -- enforced at network level, not app code.

    ## Observability (Zero Instrumentation)

    Because every request flows through the proxy, the mesh automatically collects:

    - **Metrics:** Latency (p50/p95/p99), success rate, request volume per service pair
    - **Distributed tracing:** Trace headers injected automatically (apps must propagate them on outbound calls)
    - **Access logs:** Source, destination, status, latency for every request

    ## Resilience

    | Feature | Without Mesh | With Mesh |
    |---------|-------------|-----------|
    | Retries | Each team implements differently | Uniform policy with **retry budgets** (prevents retry storms) |
    | Timeouts | Hardcoded, inconsistent | Centrally managed, per-route |
    | Circuit breaking | Requires library (Hystrix) | Infrastructure-level, zero code |
    | Outlier detection | Manual health checks | Automatic ejection of unhealthy pods |

=== "Technology Comparison"

    | Aspect | Istio | Linkerd | Consul Connect | AWS App Mesh |
    |--------|-------|---------|---------------|-------------|
    | **Proxy** | Envoy (C++) | linkerd2-proxy (Rust) | Envoy | Envoy |
    | **Memory/sidecar** | ~50MB | ~20MB | ~40MB | ~40MB |
    | **Latency added** | ~2-5ms p99 | ~1-2ms p99 | ~2-4ms p99 | ~2-4ms p99 |
    | **Complexity** | High | Low | Moderate | Low (managed) |
    | **Traffic mgmt** | Comprehensive | Basic | Moderate | Moderate |
    | **Platform** | Any K8s | K8s only | K8s, VMs, bare metal | AWS only |
    | **Best for** | Large orgs, complex routing | Simplicity, low overhead | Hybrid/multi-platform | AWS-native |

    **Quick pick:**

    - Need advanced routing + features → **Istio**
    - Want simplicity + low overhead → **Linkerd**
    - Have hybrid infra (K8s + VMs) → **Consul Connect**
    - All-in on AWS → **App Mesh**

=== "When to Use"

    ## Decision Framework

    ```
    How many microservices?
        |
        +-- < 10 services → Probably NOT needed
        |                    Use library-based approaches
        |
        +-- 10-50 services → Only if specific pain:
        |                     mTLS requirement, inconsistent retries,
        |                     no cross-service observability
        |                     → Consider Linkerd (lowest overhead)
        |
        +-- 50-200 services → Likely YES if polyglot
        |                      Linkerd for simplicity, Istio for features
        |
        +-- 200+ services → Almost certainly YES
                             Library approaches don't scale here
    ```

    **Strong signals you need a mesh:**

    - Security compliance requires mTLS everywhere
    - Teams implement retries/circuit-breaking inconsistently across languages
    - No cross-service observability (debugging = SSH into each service)
    - Need canary deployments but pipeline can't do traffic splitting

    **When it's overkill:**

    - < 10 services, single team
    - All services use same language with mature networking library
    - Not on Kubernetes
    - Pain points aren't networking-related

=== "Real-World Examples"

    ## Airbnb: Istio at 1,000+ Services

    Migrated from Rails monolith to 1,000+ microservices. Adopted Istio incrementally.

    | Metric | Before Mesh | After Mesh |
    |--------|------------|------------|
    | mTLS coverage | ~30% of traffic | 100% of traffic |
    | Tracing setup | 4 different approaches | Unified distributed tracing |
    | Canary deploys | Manual, custom tooling | Traffic-splitting via config |
    | Networking incidents | ~200/quarter | ~140/quarter |

    **Challenge:** Envoy sidecar added ~5ms per hop + significant cluster resource usage.

    ## Lyft: Created Envoy

    Built Envoy to solve the problem of every team re-implementing networking logic in different languages. Deployed as out-of-process proxy alongside any service.

    | Before Envoy | After Envoy |
    |-------------|------------|
    | 500-2,000 lines of networking code per service | 0 lines (proxy handles it) |
    | Hours to debug cross-service issues | Minutes via unified tracing |
    | Partial, inconsistent mTLS | 100% of internal traffic |

    Envoy now handles 3M+ requests/sec at Lyft across 500+ services. Open-sourced in 2016, became the foundation for Istio, App Mesh, and Consul Connect.

    ## eBay: Traffic Management at Scale

    1.5 billion transactions/day. Adopted mesh primarily for traffic control:

    - **Dark launches:** Mirror production traffic to new versions without user impact
    - **Regional failover:** Shift traffic between data centers in seconds
    - **Progressive rollouts:** 1% increments with automatic rollback on error spikes

=== "Trade-offs"

    ## Costs of a Service Mesh

    **Latency overhead:** 1-5ms per hop (two proxies per call). Negligible for shallow call graphs (2-3 services), compounds for deep ones (10+ services in sequence).

    **Resource consumption:**

    | Mesh | Memory/sidecar | 500 services × 3 replicas |
    |------|---------------|--------------------------|
    | Istio (Envoy) | ~50MB | ~75GB RAM, ~150 cores |
    | Linkerd | ~20MB | ~30GB RAM, ~37 cores |
    | Consul Connect | ~40MB | ~60GB RAM, ~120 cores |

    **Operational complexity:** Requires a platform team. Debugging shifts from "why is my app broken?" to "is it the app, the sidecar, or the control plane?"

    ## Alternatives

    ```
    Simplest ──────────────────────────────────── Most Capable

    DNS-based       API Gateway +    Library-based    Full Service
    Discovery       app libraries    (Netflix OSS)    Mesh

    - Basic routing  - External       - Rich features  - All features
    - No mTLS          traffic mgmt   - One language   - Any language
    - < 10 services  - 10-30 svcs      dependency     - 50+ services
                                      - 20-100 svcs
    ```

    Many orgs progress through this spectrum as they grow. Netflix pioneered the library approach (Hystrix, Eureka, Ribbon) but has been migrating toward a mesh model as they added polyglot services.

---

## Key Takeaways

| Concept | Summary |
|---------|---------|
| **What it is** | Infrastructure layer (sidecar proxies + control plane) for service-to-service communication |
| **Core pattern** | Sidecar proxy intercepts all traffic transparently — zero app changes |
| **Key capabilities** | Traffic management, mTLS, observability, resilience (retries/circuit breaking) |
| **When to adopt** | 50+ polyglot services, mTLS compliance, need uniform observability |
| **When to skip** | < 10 services, single language, no K8s, no platform team |
| **Leading tools** | Istio (feature-rich), Linkerd (simple), Consul (multi-platform) |
| **Primary trade-off** | Powerful cross-cutting capabilities vs operational complexity + resource overhead |

---

## Related Topics

- [Microservices Architecture](microservices.md) -- Service mesh enables microservices at scale
- [Networking: Service Discovery](../networking/service-discovery.md) -- How services find each other
- [Resilience Patterns](../reliability/resilience-patterns.md) -- Circuit breaker, retry, bulkhead in depth
- [Observability](../observability/index.md) -- Monitoring, logging, tracing
