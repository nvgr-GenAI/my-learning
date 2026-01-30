# Reliability & Fault Tolerance

**Master building systems that never stop** | 🛡️ Resilience | 🔄 Recovery | 💼 Interview Ready

## Quick Reference

**Reliability** - Building systems that continue working even when things fail:

| Pattern | Purpose | Complexity | Downtime Prevention | Example |
|---------|---------|------------|-------------------|---------|
| **Redundancy (Active-Active)** | Multiple servers handle requests | Medium | 99.99% | 3 web servers, any can fail |
| **Circuit Breaker** | Stop calling failing services | Low | Prevents cascading | API fails, stop trying |
| **Retry with Backoff** | Retry failed operations | Low | Handles transient failures | Network blip, retry 3 times |
| **Bulkhead** | Isolate failures | Medium | Contain blast radius | Thread pools per feature |
| **Health Checks** | Detect failures early | Low | Route around failures | Ping every 10 seconds |
| **Graceful Degradation** | Reduce functionality when overloaded | Medium | Stay partially available | Disable recommendations under load |
| **Chaos Engineering** | Test failures proactively | High | Find issues before users do | Kill random servers |

**Key Metrics:**
- **99.9% (3 nines)** = 8.7 hours downtime/year
- **99.99% (4 nines)** = 52 minutes downtime/year
- **99.999% (5 nines)** = 5.3 minutes downtime/year

**Key Insight:** **Design for failure, not success.** Everything fails eventually. Build systems that keep working anyway.

---

=== "🎯 Understanding Reliability"

    ## What is Reliability?

    **Reliability** is your system's ability to keep working correctly even when components fail.

    ### The Airplane Analogy

    ```
    ✈️ Why Airplanes Are Reliable:
    ─────────────────────────────────────────────

    Single Points of Failure Eliminated:
    - 2 engines (one can fail) ✓
    - 3 hydraulic systems (redundancy) ✓
    - Multiple power sources (backup generators) ✓
    - Trained pilots + autopilot (human + machine) ✓

    Result: 1 in 10 million flights crash
    Availability: 99.99999% (7 nines!)

    Your System Should Be Similar:
    - Multiple servers (one can fail)
    - Multiple databases (replicas)
    - Multiple datacenters (region failure)
    - Automated failover (no human needed)
    ```

    ---

    ## The 9's of Availability

    ```mermaid
    graph LR
        A[99%<br/>3.65 days/year<br/>Unacceptable] --> B[99.9%<br/>8.7 hours/year<br/>Acceptable]
        B --> C[99.99%<br/>52 minutes/year<br/>Good]
        C --> D[99.999%<br/>5.3 minutes/year<br/>Excellent]
        D --> E[99.9999%<br/>31 seconds/year<br/>Amazon/Google]

        style A fill:#ff6b6b
        style D fill:#51cf66
        style E fill:#fab005
    ```

    | Availability | Downtime/Year | Downtime/Month | Downtime/Day | Use Case |
    |-------------|---------------|----------------|--------------|----------|
    | **90%** | 36.5 days | 3 days | 2.4 hours | Development |
    | **99%** | 3.65 days | 7.2 hours | 14 minutes | Internal tools |
    | **99.9%** | 8.7 hours | 43 minutes | 1.4 minutes | Standard SaaS |
    | **99.99%** | 52 minutes | 4.3 minutes | 8.6 seconds | Financial services |
    | **99.999%** | 5.3 minutes | 26 seconds | 0.86 seconds | Mission critical (AWS) |

    ---

    ## Reliability vs Availability

    | Aspect | Reliability | Availability |
    |--------|------------|-------------|
    | **Definition** | Works correctly without errors | System is accessible/operational |
    | **Focus** | Correctness | Uptime |
    | **Metric** | Error rate, MTBF | Uptime percentage |
    | **Example** | Returns correct data 99.9% of time | Responds to requests 99.9% of time |
    | **Failure** | Wrong data returned | System unreachable |

    **Key Difference:**
    ```
    Scenario: Website accessible but returns wrong prices
    ────────────────────────────────────────────────────
    Availability: 100% ✓ (site is up)
    Reliability: 0% ✗ (data is wrong)

    You need BOTH!
    ```

    ---

    ## The Cost of Downtime

    ```
    Real-World Impact:
    ─────────────────────────────────────────────

    Amazon:
    - 1 hour downtime = $220M lost revenue
    - 99.9% = 8.7 hours/year = $1.9B lost!

    Google:
    - 1 minute downtime = $500K lost

    Facebook:
    - 1 hour downtime = $90M lost

    Your Startup:
    - 99% availability (3.65 days down/year)
    - Users churn after 1 day of issues
    - Result: Dead company

    Lesson: High availability is NOT optional!
    ```

=== "🏗️ Fault Tolerance Patterns"

    ## The 7 Essential Patterns

    === "1. Redundancy"

        ### Eliminate Single Points of Failure

        **Active-Active (Hot Standby):**
        ```
        Load Balancer
                │
        ┌───────┼───────┐
        │       │       │
        ▼       ▼       ▼
     Server1 Server2 Server3
      (33%)   (33%)   (33%)

        All servers handle traffic
        One fails? Others absorb load
        No failover delay
        Better resource utilization

        Example: Netflix
        - 1000s of servers globally
        - Any server can fail
        - Traffic auto-routes to healthy servers
        - Users never notice
        ```

        ---

        **Active-Passive (Cold Standby):**
        ```
        Load Balancer
                │
            ┌───┴───┐
            │       │
            ▼       ▼
         Primary  Standby
         (100%)   (0%)

        Only primary handles traffic
        Standby waits idle
        Failure detected? Activate standby
        Failover: 30-60 seconds

        Example: Database
        - Primary handles all writes
        - Standby syncs from primary
        - Primary fails? Promote standby
        - Brief downtime during switch
        ```

        ---

        **N+1 Redundancy:**
        ```
        Capacity Planning:
        ─────────────────────────────────────────────
        Need: 10 servers (peak load)
        Deploy: 11 servers (N+1)

        Normal operation: 11 servers at 91% capacity
        1 server fails: 10 servers at 100% capacity ✓
        Still handles peak load!

        N+2 Redundancy (Better):
        ─────────────────────────────────────────────
        Need: 10 servers
        Deploy: 12 servers (N+2)

        Normal: 12 servers at 83% capacity
        1 fails: 11 servers at 91% capacity ✓
        2 fail: 10 servers at 100% capacity ✓
        More resilient!
        ```

        ---

        **Geographic Redundancy:**
        ```
        Multi-Region Deployment:
        ─────────────────────────────────────────────

        US-East (Primary)
        - 100 servers
        - Handles North America traffic

        EU-West (Secondary)
        - 100 servers
        - Handles Europe traffic

        Asia-Pacific (Secondary)
        - 100 servers
        - Handles Asia traffic

        Benefits:
        - US datacenter down? EU/Asia handle traffic
        - Low latency (users hit nearest region)
        - Disaster recovery (earthquake, fire, etc.)

        Cost: 3x infrastructure, but worth it
        ```

    === "2. Circuit Breaker"

        ### Stop Calling Failing Services

        **How It Works:**
        ```
        3 States:
        ─────────────────────────────────────────────

        CLOSED (Normal):
        ┌─────────────────────────────┐
        │ Requests flow through       │
        │ Track success/failure rate  │
        │ 5 failures in 10s? → OPEN   │
        └─────────────────────────────┘

        OPEN (Failing):
        ┌─────────────────────────────┐
        │ Block all requests          │
        │ Return error immediately    │
        │ Wait 30 seconds             │
        │ Timeout? → HALF-OPEN        │
        └─────────────────────────────┘

        HALF-OPEN (Testing):
        ┌─────────────────────────────┐
        │ Allow 3 test requests       │
        │ All succeed? → CLOSED       │
        │ Any fail? → OPEN            │
        └─────────────────────────────┘
        ```

        ---

        ### Real-World Example

        ```
        Scenario: Payment Service Failing
        ─────────────────────────────────────────────

        Without Circuit Breaker:
        ────────────────────────
        Every request tries to call payment service
        - Each attempt: 30-second timeout
        - 1000 concurrent users
        - 1000 threads blocked for 30s
        - All threads exhausted
        - Entire system crashes ❌

        With Circuit Breaker:
        ────────────────────────
        After 5 failures:
        - Circuit opens
        - Stop calling payment service
        - Return "Payment temporarily unavailable"
        - Users can still browse/add to cart
        - System stays up ✓

        Wait 30 seconds, try again:
        - Success? Resume normal operations
        - Failure? Keep circuit open

        Result: Contained failure, system available
        ```

        ---

        ### Configuration

        | Setting | Value | Reasoning |
        |---------|-------|-----------|
        | **Failure Threshold** | 5 failures | Balance: Not too sensitive |
        | **Timeout** | 30 seconds | Give service time to recover |
        | **Half-Open Attempts** | 3 requests | Test with small sample |
        | **Window** | 10 seconds | Recent failures matter most |

    === "3. Retry with Backoff"

        ### Handle Transient Failures

        **Retry Strategies:**

        ```
        Fixed Delay:
        ─────────────────────────────────────────────
        Attempt 1: Fail → Wait 1s
        Attempt 2: Fail → Wait 1s
        Attempt 3: Fail → Wait 1s

        Problem: Thundering herd (all retry at once)

        Exponential Backoff:
        ─────────────────────────────────────────────
        Attempt 1: Fail → Wait 1s
        Attempt 2: Fail → Wait 2s
        Attempt 3: Fail → Wait 4s
        Attempt 4: Fail → Wait 8s

        Better: Spreads out retries

        Exponential Backoff + Jitter:
        ─────────────────────────────────────────────
        Attempt 1: Fail → Wait 1s + random(0, 200ms)
        Attempt 2: Fail → Wait 2s + random(0, 400ms)
        Attempt 3: Fail → Wait 4s + random(0, 800ms)

        Best: Randomizes retries, avoids thundering herd
        ```

        ---

        ### When to Retry

        | Error Type | Should Retry? | Why |
        |-----------|--------------|-----|
        | **Network timeout** | ✓ Yes | Transient network issue |
        | **503 Service Unavailable** | ✓ Yes | Server overloaded, temporary |
        | **Connection refused** | ✓ Yes | Server restarting |
        | **400 Bad Request** | ✗ No | Client error, won't fix itself |
        | **401 Unauthorized** | ✗ No | Auth issue, retry won't help |
        | **404 Not Found** | ✗ No | Resource doesn't exist |

        ---

        ### Real-World Example

        ```
        Scenario: Microservice Call Fails
        ─────────────────────────────────────────────

        Call to user-service times out:
        Attempt 1 (0s): Timeout after 1s → Retry
        Attempt 2 (1s): Timeout after 1s → Retry
        Attempt 3 (3s): Success! ✓

        Total time: 3s (without retry: failure)
        User doesn't notice the transient issue

        Configuration:
        - Max retries: 3
        - Base delay: 1s
        - Max delay: 16s
        - Jitter: ±200ms
        - Total max time: ~31s
        ```

    === "4. Bulkhead Pattern"

        ### Isolate Failures

        **Resource Isolation:**
        ```
        Ship Design Analogy:
        ─────────────────────────────────────────────
        Ship with one compartment:
        - Hole in hull → Entire ship sinks ❌

        Ship with bulkheads (compartments):
        - Hole in hull → One compartment floods
        - Other compartments sealed
        - Ship stays afloat ✓

        Your System:
        ─────────────────────────────────────────────
        Single thread pool:
        - Slow API call → All threads blocked
        - Other features can't work ❌

        Separate thread pools (bulkheads):
        - Slow API pool: 20 threads (blocked)
        - User service pool: 50 threads (still working)
        - Admin pool: 10 threads (still working)
        - Other features continue! ✓
        ```

        ---

        ### Implementation Examples

        **1. Thread Pool Isolation:**
        ```
        Web Application:
        ─────────────────────────────────────────────
        ┌─────────────────────────────────────┐
        │ User Requests: 50 threads           │ ← Most important
        ├─────────────────────────────────────┤
        │ Search: 20 threads                  │ ← Medium priority
        ├─────────────────────────────────────┤
        │ Analytics: 10 threads               │ ← Can fail
        ├─────────────────────────────────────┤
        │ Admin: 5 threads                    │ ← Low traffic
        └─────────────────────────────────────┘

        Analytics fails? Only 10 threads affected
        User requests still work! ✓
        ```

        ---

        **2. Connection Pool Isolation:**
        ```
        Database Connections:
        ─────────────────────────────────────────────
        User DB pool: 50 connections
        Analytics DB pool: 10 connections

        Analytics query runs wild:
        - Consumes all 10 analytics connections
        - User DB pool unaffected
        - Users can still log in, browse
        ```

        ---

        **3. Resource Limits:**
        ```
        Docker/Kubernetes:
        ─────────────────────────────────────────────
        Container 1 (API service):
        - CPU limit: 2 cores
        - Memory limit: 4GB

        Container 2 (Background jobs):
        - CPU limit: 1 core
        - Memory limit: 2GB

        Background job goes crazy:
        - Maxes out 1 core, 2GB
        - API service unaffected (separate limits)
        ```

    === "5. Health Checks"

        ### Detect Failures Early

        **Health Check Types:**

        | Type | Checks | Response Time | Use Case |
        |------|--------|--------------|----------|
        | **Liveness** | Process alive? | <100ms | Container restart if fails |
        | **Readiness** | Ready to serve? | <500ms | Remove from load balancer |
        | **Deep Health** | Dependencies OK? | <2s | Full system status |

        ---

        ### Liveness Check (Shallow):
        ```
        GET /health/live

        Checks:
        - Application process running? ✓
        - Basic HTTP response? ✓

        Response: HTTP 200
        {
          "status": "alive",
          "timestamp": 1234567890
        }

        Fast: <100ms
        Purpose: Restart dead containers
        ```

        ---

        ### Readiness Check (Medium):
        ```
        GET /health/ready

        Checks:
        - Database reachable? ✓
        - Cache reachable? ✓
        - Sufficient memory? ✓
        - Not overloaded? ✓

        Response: HTTP 200
        {
          "status": "ready",
          "checks": {
            "database": "ok",
            "cache": "ok",
            "memory": "ok"
          }
        }

        Moderate: <500ms
        Purpose: Route traffic only to ready instances
        ```

        ---

        ### Deep Health Check (Comprehensive):
        ```
        GET /health/deep

        Checks:
        - All dependencies reachable
        - Database query works
        - Can write to cache
        - External API accessible
        - Disk space available
        - End-to-end workflow test

        Response: HTTP 200 (all OK) or 503 (issues)
        {
          "status": "healthy",
          "dependencies": {
            "database": {"status": "ok", "latency": "5ms"},
            "cache": {"status": "ok", "latency": "1ms"},
            "payment_api": {"status": "ok", "latency": "50ms"}
          }
        }

        Slow: <2s
        Purpose: Monitoring dashboard, debugging
        ```

        ---

        ### Configuration

        ```
        Load Balancer Settings:
        ─────────────────────────────────────────────
        Endpoint: GET /health/ready
        Interval: 10 seconds
        Timeout: 5 seconds
        Unhealthy threshold: 3 consecutive failures
        Healthy threshold: 2 consecutive successes

        Result:
        - Failure detected in 30 seconds (3 × 10s)
        - Recovery detected in 20 seconds (2 × 10s)
        - Server out of rotation 50 seconds total
        ```

    === "6. Graceful Degradation"

        ### Reduce Functionality Under Load

        **Degradation Levels:**

        ```
        Full Functionality (Normal):
        ─────────────────────────────────────────────
        Load: 5,000 RPS (50% capacity)
        Features: All working
        Performance: Fast (100ms P95)

        Example: E-commerce
        ✓ Product search with ML recommendations
        ✓ Real-time inventory
        ✓ Personalized pricing
        ✓ Related products
        ✓ Customer reviews

        Reduced Functionality (High Load):
        ─────────────────────────────────────────────
        Load: 8,000 RPS (80% capacity)
        Features: Non-essential disabled
        Performance: Slower (200ms P95)

        Example: E-commerce
        ✓ Product search (basic, no ML)
        ✓ Real-time inventory
        ✓ Standard pricing
        ✗ Related products (disabled)
        ✗ Reviews (cached version)

        Core Functionality (Overload):
        ─────────────────────────────────────────────
        Load: 9,500 RPS (95% capacity)
        Features: Only critical
        Performance: Degraded (500ms P95)

        Example: E-commerce
        ✓ Product search (cached results)
        ✓ Inventory (5-minute cache)
        ✗ Personalization (disabled)
        ✗ Related products (disabled)
        ✗ Reviews (disabled)

        Emergency Mode (Danger):
        ─────────────────────────────────────────────
        Load: 10,000+ RPS (100% capacity)
        Features: Minimal
        Performance: Slow (1000ms P95)

        Example: E-commerce
        ✓ View product (cached)
        ✓ Add to cart
        ✗ Search (show cached popular products)
        ✗ Everything else disabled

        Key: System stays up, users can still buy!
        ```

        ---

        ### Feature Flags Implementation

        ```
        Dynamic Feature Control:
        ─────────────────────────────────────────────

        CPU > 80%?
        → Disable ML recommendations

        Error rate > 5%?
        → Disable related products API call

        Response time > 500ms?
        → Return cached search results

        Database connections > 80%?
        → Disable real-time inventory, use cache

        Result: System degrades gracefully
        Users experience slower but working system
        Better than complete outage!
        ```

    === "7. Timeouts"

        ### Fail Fast, Don't Wait Forever

        **Timeout Hierarchy:**
        ```
        Must satisfy: Client > Load Balancer > Server
        ─────────────────────────────────────────────

        Client timeout: 60s
            ↓
        Load Balancer timeout: 50s
            ↓
        Server timeout: 45s
            ↓
        Database timeout: 30s

        Why? Each layer needs time to handle timeout
        ```

        ---

        ### Timeout Types

        | Type | Purpose | Typical Value | Impact of Too Long | Impact of Too Short |
        |------|---------|---------------|-------------------|-------------------|
        | **Connection** | Establish connection | 5-10s | Threads blocked | False failures |
        | **Request** | Complete request | 30-60s | Resources wasted | Premature failures |
        | **Database Query** | Query execution | 10-30s | Slow queries block | Queries interrupted |
        | **Idle** | Close unused connections | 5-10 min | Connection leaks | Reconnection overhead |

        ---

        ### Real-World Example

        ```
        Scenario: Slow Database Query
        ─────────────────────────────────────────────

        Without Timeout:
        ────────────────
        Query runs for 5 minutes
        - User waits 5 minutes
        - Thread blocked for 5 minutes
        - 100 concurrent users = 100 blocked threads
        - All threads exhausted
        - System hangs ❌

        With 30s Timeout:
        ─────────────────
        Query timeout after 30s
        - Return error to user
        - Thread freed after 30s
        - Other requests can use thread
        - System stays responsive ✓
        - Log slow query for investigation

        Result: Fail fast, system available
        ```

=== "💡 Interview Tips"

    ## Common Interview Questions

    **Q1: "What's the difference between reliability and availability?"**

    **Good Answer:**
    ```
    Availability = Uptime
    ─────────────────────────────────────────────
    - Is the system accessible?
    - Can users reach it?
    - Metric: % of time system is up
    - Example: 99.9% uptime = 8.7 hours down/year

    Reliability = Correctness
    ─────────────────────────────────────────────
    - Does the system work correctly?
    - Does it return right data?
    - Metric: Error rate, MTBF
    - Example: 99.9% reliability = 0.1% wrong responses

    Key Difference:
    ─────────────────────────────────────────────
    Scenario: Website up but shows wrong prices
    - Availability: 100% (site accessible)
    - Reliability: 0% (data is wrong)

    You Need Both!
    ─────────────────────────────────────────────
    Good system: 99.9% available AND 99.9% reliable
    - Users can reach it (available)
    - It returns correct data (reliable)
    ```

    ---

    **Q2: "Explain the circuit breaker pattern"**

    **Good Answer:**
    ```
    Purpose: Stop calling a failing service
    ─────────────────────────────────────────────

    Problem Without Circuit Breaker:
    ────────────────────────────────
    Payment service is down
    - Every request tries to call it
    - Each attempt: 30-second timeout
    - 1000 users = 1000 threads waiting 30s
    - All threads blocked
    - Entire system crashes (cascading failure)

    Solution With Circuit Breaker:
    ────────────────────────────────
    3 States:
    1. CLOSED (normal): Requests go through
    2. OPEN (failing): Block requests, fail fast
    3. HALF-OPEN (testing): Try a few requests

    Flow:
    ────────────────────────────────
    CLOSED: 5 failures detected → OPEN
    OPEN: Wait 30s → HALF-OPEN
    HALF-OPEN:
      - 3 successes → CLOSED (resume)
      - Any failure → OPEN (still broken)

    Result:
    ────────────────────────────────
    - Detect failure after 5 attempts
    - Stop trying (fail fast)
    - Periodic retry (self-healing)
    - Protect system from cascade
    - Users see "Service unavailable" instead of timeout

    Real Example: Netflix
    ────────────────────────────────
    Recommendation service fails
    - Circuit opens
    - Show generic recommendations (cached)
    - Users can still browse/watch
    - System stays up!
    ```

    ---

    **Q3: "How would you achieve 99.99% availability?"**

    **Good Answer:**
    ```
    99.99% = 52 minutes downtime/year
    ─────────────────────────────────────────────

    Step-by-Step Strategy:

    1. Eliminate Single Points of Failure:
    ─────────────────────────────────────────────
    - Multiple web servers (N+2 redundancy)
    - Multiple databases (primary + replicas)
    - Multiple load balancers (active-active)
    - Multiple datacenters (multi-region)

    2. Automated Failover:
    ─────────────────────────────────────────────
    - Health checks every 10s
    - Auto-remove failed instances
    - No manual intervention needed
    - Failover time: <30s

    3. Deployment Strategy:
    ─────────────────────────────────────────────
    - Blue-green deployments (zero downtime)
    - Canary releases (test on 1% traffic first)
    - Automated rollback on errors
    - Deploy during low-traffic hours

    4. Monitoring & Alerting:
    ─────────────────────────────────────────────
    - 24/7 monitoring
    - PagerDuty for critical alerts
    - Runbooks for common issues
    - On-call rotation

    5. Chaos Engineering:
    ─────────────────────────────────────────────
    - Kill random servers (Chaos Monkey)
    - Simulate datacenter failures
    - Test disaster recovery monthly
    - Find issues before users do

    Math Check:
    ─────────────────────────────────────────────
    Each layer: 99.9% availability
    Combined: 99.9% × 99.9% = 99.8% ❌
    Need: 99.99% per layer × redundancy

    With 2 datacenters (active-active):
    - Each datacenter: 99.9%
    - Combined: 1 - (0.001 × 0.001) = 99.9999% ✓

    Key: Redundancy + Automation + Testing
    ```

    ---

    **Q4: "What's the bulkhead pattern?"**

    **Good Answer:**
    ```
    Purpose: Isolate failures (contain blast radius)
    ─────────────────────────────────────────────

    Analogy: Ship Design
    ────────────────────────────────
    Ship without bulkheads:
    - Hole in hull → Entire ship floods → Sinks ❌

    Ship with bulkheads (compartments):
    - Hole in hull → One compartment floods
    - Seal compartment
    - Other compartments fine
    - Ship stays afloat ✓

    Your System:
    ────────────────────────────────
    Single thread pool (100 threads):
    - Slow API call consumes all 100 threads
    - No threads left for other requests
    - Entire system hangs ❌

    Separate thread pools (bulkheads):
    ────────────────────────────────
    - User requests: 50 threads
    - Search: 20 threads
    - Analytics: 10 threads
    - Admin: 5 threads

    Analytics query runs wild:
    - Consumes all 10 analytics threads
    - User pool still has 50 threads available
    - Search still works
    - System partially degraded, not dead ✓

    Implementation:
    ────────────────────────────────
    1. Thread pool isolation
    2. Connection pool isolation
    3. Resource limits (CPU, memory)
    4. Rate limiting per feature

    Real Example: Netflix
    ────────────────────────────────
    Separate thread pools per dependency:
    - User service: 50 threads
    - Recommendation service: 30 threads
    - Video metadata: 20 threads

    Recommendation service down?
    - Only 30 threads affected
    - Users can still browse, search, watch
    - Non-personalized experience, but working!
    ```

    ---

    **Q5: "How do you handle cascading failures?"**

    **Good Answer:**
    ```
    Cascading Failure = Domino Effect
    ─────────────────────────────────────────────

    Example Scenario:
    ────────────────────────────────
    Database slow → API slow → Web slow → All fail

    Prevention Strategies:

    1. Circuit Breaker:
    ────────────────────────────────
    - Detect failing service
    - Stop calling it (break the chain)
    - Fail fast instead of waiting

    2. Timeouts:
    ────────────────────────────────
    - Limit wait time (30s max)
    - Don't wait forever
    - Free up resources quickly

    3. Bulkheads:
    ────────────────────────────────
    - Isolate thread pools
    - One service fails, others continue
    - Contain the blast radius

    4. Rate Limiting:
    ────────────────────────────────
    - Limit requests to failing service
    - Protect it from overload
    - Give it time to recover

    5. Load Shedding:
    ────────────────────────────────
    - Under high load, drop low-priority requests
    - Analytics, recommendations can wait
    - Preserve capacity for critical features

    6. Graceful Degradation:
    ────────────────────────────────
    - Disable non-essential features
    - Return cached/stale data
    - System stays up (degraded but working)

    Real Example: AWS Outage
    ────────────────────────────────
    One service fails → Dependencies fail → Cascade

    Better: Netflix on AWS Outage
    ────────────────────────────────
    - Circuit breakers detect failures
    - Fallback to cached data
    - Disable personalization
    - Users can still watch (core feature works)
    - Degraded experience > Complete outage
    ```

    ---

    ## Interview Cheat Sheet

    **Quick Comparisons:**

    | Pattern | Purpose | Complexity | When to Use |
    |---------|---------|------------|-------------|
    | **Redundancy** | Backup systems | Medium | Always (multiple servers) |
    | **Circuit Breaker** | Stop calling failures | Low | Calling external services |
    | **Retry** | Handle transient failures | Low | Network calls, API calls |
    | **Bulkhead** | Isolate failures | Medium | Multiple features/dependencies |
    | **Health Check** | Detect failures | Low | Load balancing |
    | **Graceful Degradation** | Partial availability | Medium | High load, failures |
    | **Chaos Engineering** | Test failures | High | Production (controlled) |

    **Availability Targets:**

    ```
    99% (2 nines): Unacceptable (3.65 days down/year)
    99.9% (3 nines): Standard (8.7 hours down/year)
    99.99% (4 nines): Good (52 minutes down/year)
    99.999% (5 nines): Excellent (5.3 minutes down/year)
    99.9999% (6 nines): Amazon/Google scale
    ```

    **Cost of Each 9:**
    - 99% → 99.9%: 2x cost (redundancy)
    - 99.9% → 99.99%: 3x cost (multi-region)
    - 99.99% → 99.999%: 5x cost (chaos engineering, 24/7 on-call)

=== "⚠️ Common Mistakes"

    ## Reliability Pitfalls

    | Mistake | Problem | Solution |
    |---------|---------|----------|
    | **No redundancy** | Single point of failure | N+1 redundancy minimum |
    | **Synchronous calls** | Cascading failures | Circuit breaker + timeouts |
    | **No timeouts** | Threads block forever | Timeout everything (30-60s) |
    | **No health checks** | Route to dead servers | Health check every 10s |
    | **No monitoring** | Don't know system is down | 24/7 monitoring + alerts |
    | **No testing failures** | Surprises in production | Chaos engineering |
    | **Manual failover** | Slow recovery (hours) | Automated failover (<30s) |
    | **Optimistic retry** | Thundering herd | Exponential backoff + jitter |

    ---

    ## Design Pitfalls

    | Pitfall | Impact | Prevention |
    |---------|--------|-----------|
    | **Single database** | Database down = system down | Primary + 2 replicas |
    | **Single datacenter** | Region outage = full outage | Multi-region active-active |
    | **Single load balancer** | LB down = system down | 2+ load balancers |
    | **No backups** | Data loss = company death | Daily backups + monthly test |
    | **Tight coupling** | One service fails, all fail | Loose coupling + circuit breakers |
    | **No degradation** | All-or-nothing availability | Graceful degradation levels |

    ---

    ## Interview Red Flags

    **Avoid Saying:**
    - ❌ "We'll just use a bigger server" (single point of failure)
    - ❌ "Downtime is acceptable" (depends on use case, but usually not)
    - ❌ "We'll handle failures manually" (too slow, human error)
    - ❌ "One datacenter is enough" (regional failures happen)
    - ❌ "We'll add redundancy later" (too late when you need it)

    **Say Instead:**
    - ✅ "N+1 redundancy with automated failover"
    - ✅ "Target 99.9% availability (8.7 hours down/year max)"
    - ✅ "Circuit breakers + timeouts to prevent cascading failures"
    - ✅ "Multi-region active-active for disaster recovery"
    - ✅ "Chaos engineering to test failures proactively"

---

## 🎯 Key Takeaways

**The 10 Rules of Reliability:**

1. **Assume everything fails** - Servers, networks, databases, datacenters. Design for failure.

2. **Eliminate single points of failure** - N+1 redundancy minimum, N+2 better.

3. **Automate failover** - Manual is too slow (hours vs. seconds).

4. **Circuit breakers everywhere** - Stop calling failing services, prevent cascades.

5. **Timeout everything** - Don't wait forever. 30-60s max for most operations.

6. **Health checks** - Detect failures early, route around them automatically.

7. **Graceful degradation** - Partial availability > complete outage.

8. **Monitor continuously** - 24/7 monitoring + alerts. On-call rotation.

9. **Test failures** - Chaos engineering. Find issues before users do.

10. **Multi-region** - For mission-critical systems. Single region = vulnerable.

---

## 📚 Further Reading

**Master these related concepts:**

| Topic | Why Important | Read Next |
|-------|--------------|-----------|
| **Circuit Breaker** | Prevent cascading failures | [Resilience Patterns →](../reliability/circuit-breaker.md) |
| **Disaster Recovery** | Recover from major outages | [DR Planning →](../reliability/disaster-recovery.md) |
| **Chaos Engineering** | Test failures proactively | [Chaos Guide →](../reliability/chaos-engineering.md) |
| **Monitoring** | Detect issues early | [Observability →](../monitoring/index.md) |

**Practice with real systems:**
- [Design Netflix](../problems/netflix.md) - High availability, circuit breakers
- [Design WhatsApp](../problems/whatsapp.md) - 99.99% availability, multi-region
- [Design Banking System](../problems/banking.md) - Reliability, disaster recovery

---

**Master reliability and build systems that never stop! 🛡️**
