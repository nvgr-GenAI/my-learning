# Deployment Strategies

**Deploy without downtime** | 🔵🟢 Blue-Green | 🐤 Canary | 🔄 Rolling | 🔀 A/B Testing

---

## Overview

Deployment strategies determine how you release new versions of your application to production. The right strategy minimizes risk, reduces downtime, and enables quick rollbacks.

**Key Goal:** Deploy changes safely without disrupting users.

---

## Strategy Comparison

| Strategy | Downtime | Risk | Rollback Speed | Cost | Complexity | Best For |
|----------|----------|------|----------------|------|------------|----------|
| **Recreate** | Yes (minutes) | High | Medium | Low | Very Low | Dev/test only |
| **Rolling** | Zero | Medium | Slow | Low | Low | Standard apps |
| **Blue-Green** | Zero | Low | Instant | High (2x infra) | Medium | Critical apps |
| **Canary** | Zero | Very Low | Fast | Medium | High | Gradual rollouts |
| **A/B Testing** | Zero | Low | Fast | Medium | High | Feature testing |

---

## Recreate Deployment

=== "How It Works"
    **Simplest strategy: stop all, deploy new, start all**

    ```
    Step 1: Stop all v1.0 instances
    [STOPPED] [STOPPED] [STOPPED]

    Downtime window (1-5 minutes)

    Step 2: Deploy v2.0
    [v2.0] [v2.0] [v2.0]

    Step 3: Start all instances
    [RUNNING] [RUNNING] [RUNNING]
    ```

=== "Implementation"
    ```bash
    #!/bin/bash
    # Simple recreate deployment script

    echo "Stopping current version..."
    kubectl scale deployment myapp --replicas=0

    echo "Waiting for pods to terminate..."
    kubectl wait --for=delete pod -l app=myapp --timeout=60s

    echo "Updating image..."
    kubectl set image deployment/myapp myapp=myapp:v2.0

    echo "Starting new version..."
    kubectl scale deployment myapp --replicas=3

    echo "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=myapp --timeout=120s

    echo "Deployment complete!"
    ```

=== "Use Cases"
    **When to use:**
    - ✅ Development/staging environments
    - ✅ Internal tools with flexible uptime
    - ✅ Maintenance windows acceptable
    - ✅ Non-critical applications

    **Don't use for:**
    - ❌ Production user-facing apps
    - ❌ 24/7 services
    - ❌ SLA-bound applications

=== "Advantages & Disadvantages"
    **Advantages:**
    - ✅ Simplest to implement
    - ✅ No complex routing
    - ✅ Clean state (no mixed versions)
    - ✅ Lowest cost (no extra resources)

    **Disadvantages:**
    - ❌ Downtime during deployment
    - ❌ Users experience service interruption
    - ❌ Not suitable for production

---

## Rolling Deployment

=== "How It Works"
    **Gradually replace instances one by one**

    ```
    Initial State (5 instances, v1.0):
    [v1.0] [v1.0] [v1.0] [v1.0] [v1.0]

    Step 1: Stop 1, deploy v2.0:
    [v2.0] [v1.0] [v1.0] [v1.0] [v1.0]

    Step 2: Health check passes, continue:
    [v2.0] [v2.0] [v1.0] [v1.0] [v1.0]

    Step 3: Continue rolling:
    [v2.0] [v2.0] [v2.0] [v1.0] [v1.0]

    Step 4: Almost done:
    [v2.0] [v2.0] [v2.0] [v2.0] [v1.0]

    Final: All updated:
    [v2.0] [v2.0] [v2.0] [v2.0] [v2.0]
    ```

=== "Implementation"
    **Kubernetes RollingUpdate:**
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: myapp
    spec:
      replicas: 10
      strategy:
        type: RollingUpdate
        rollingUpdate:
          maxUnavailable: 2     # Max 2 pods down at once (20%)
          maxSurge: 2           # Max 2 extra pods during update
      template:
        metadata:
          labels:
            app: myapp
        spec:
          containers:
          - name: myapp
            image: myapp:v2.0
            readinessProbe:
              httpGet:
                path: /health
                port: 8080
              initialDelaySeconds: 10
              periodSeconds: 5
            livenessProbe:
              httpGet:
                path: /health
                port: 8080
              initialDelaySeconds: 30
              periodSeconds: 10
    ```

    **Manual script (for VMs/EC2):**
    ```bash
    #!/bin/bash
    SERVERS=("server1" "server2" "server3" "server4" "server5")

    for server in "${SERVERS[@]}"; do
        echo "Deploying to $server..."

        # Remove from load balancer
        aws elb deregister-instances-from-load-balancer \
            --load-balancer-name my-lb \
            --instances $server

        # Wait for connections to drain (30 seconds)
        sleep 30

        # Deploy new version
        ssh $server "sudo systemctl stop myapp"
        scp myapp-v2.0.jar $server:/opt/myapp/
        ssh $server "sudo systemctl start myapp"

        # Wait for health check
        until curl -f http://$server:8080/health; do
            echo "Waiting for $server to be healthy..."
            sleep 5
        done

        # Add back to load balancer
        aws elb register-instances-with-load-balancer \
            --load-balancer-name my-lb \
            --instances $server

        echo "$server deployment complete!"

        # Pause between servers
        sleep 10
    done

    echo "Rolling deployment complete!"
    ```

=== "Configuration Parameters"
    **Key settings to tune:**

    ```yaml
    # Conservative (slow, safe)
    maxUnavailable: 1    # Only 1 down at a time
    maxSurge: 0          # No extra pods

    # Aggressive (fast, riskier)
    maxUnavailable: 25%  # Up to 25% down
    maxSurge: 25%        # Up to 25% extra pods

    # Balanced (recommended)
    maxUnavailable: 2
    maxSurge: 1
    ```

    **Health checks are critical:**
    ```yaml
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 10   # Wait before first check
      periodSeconds: 5          # Check every 5 seconds
      failureThreshold: 3       # Fail after 3 failures
      successThreshold: 1       # Success after 1 success
    ```

=== "Monitoring During Rollout"
    ```bash
    # Watch rollout status
    kubectl rollout status deployment/myapp

    # See rollout history
    kubectl rollout history deployment/myapp

    # Pause rollout if issues detected
    kubectl rollout pause deployment/myapp

    # Resume after investigation
    kubectl rollout resume deployment/myapp

    # Rollback to previous version
    kubectl rollout undo deployment/myapp

    # Rollback to specific revision
    kubectl rollout undo deployment/myapp --to-revision=2
    ```

=== "Advantages & Disadvantages"
    **Advantages:**
    - ✅ Zero downtime
    - ✅ No extra infrastructure cost
    - ✅ Built into Kubernetes
    - ✅ Automatic health checking
    - ✅ Can pause/resume mid-rollout

    **Disadvantages:**
    - ❌ Slow rollback (must roll back each instance)
    - ❌ Both versions run simultaneously (compatibility issues)
    - ❌ Hard to test new version before full rollout
    - ❌ Database migrations can be tricky

---

## Blue-Green Deployment

=== "How It Works"
    **Maintain two identical environments, switch traffic instantly**

    ```
    Blue Environment (Current Production):
    ┌────────────────────────────────────┐
    │ Load Balancer (100% → Blue)        │
    └─────────────┬──────────────────────┘
                  ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  v1.0    │ │  v1.0    │ │  v1.0    │
    │  Blue-1  │ │  Blue-2  │ │  Blue-3  │
    └──────────┘ └──────────┘ └──────────┘

    Green Environment (New Version - Idle):
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  v2.0    │ │  v2.0    │ │  v2.0    │
    │ Green-1  │ │ Green-2  │ │ Green-3  │
    └──────────┘ └──────────┘ └──────────┘
         ↑           ↑           ↑
      (Testing - no production traffic)

    After Testing Passes:
    ┌────────────────────────────────────┐
    │ Load Balancer (100% → Green)       │ ← Switch!
    └─────────────┬──────────────────────┘
                  ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  v2.0    │ │  v2.0    │ │  v2.0    │
    │ Green-1  │ │ Green-2  │ │ Green-3  │
    └──────────┘ └──────────┘ └──────────┘

    Blue (Now Idle - Keep for Rollback):
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  v1.0    │ │  v1.0    │ │  v1.0    │
    └──────────┘ └──────────┘ └──────────┘
    ```

=== "Implementation"
    **Kubernetes with Service Selector:**
    ```yaml
    # Blue Deployment
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: myapp-blue
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: myapp
          version: blue
      template:
        metadata:
          labels:
            app: myapp
            version: blue
        spec:
          containers:
          - name: myapp
            image: myapp:v1.0
            ports:
            - containerPort: 8080

    ---
    # Green Deployment
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: myapp-green
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: myapp
          version: green
      template:
        metadata:
          labels:
            app: myapp
            version: green
        spec:
          containers:
          - name: myapp
            image: myapp:v2.0
            ports:
            - containerPort: 8080

    ---
    # Service (controls which version gets traffic)
    apiVersion: v1
    kind: Service
    metadata:
      name: myapp
    spec:
      selector:
        app: myapp
        version: blue  # Change to 'green' to switch
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8080
      type: LoadBalancer
    ```

    **Deployment script:**
    ```bash
    #!/bin/bash
    # Blue-Green deployment script

    CURRENT_VERSION=$(kubectl get service myapp -o jsonpath='{.spec.selector.version}')
    NEW_VERSION=$([ "$CURRENT_VERSION" = "blue" ] && echo "green" || echo "blue")

    echo "Current version: $CURRENT_VERSION"
    echo "Deploying to: $NEW_VERSION"

    # Deploy new version
    kubectl apply -f deployment-$NEW_VERSION.yaml

    # Wait for new version to be ready
    kubectl wait --for=condition=available --timeout=300s \
        deployment/myapp-$NEW_VERSION

    # Run smoke tests against new version
    echo "Running smoke tests..."
    NEW_POD=$(kubectl get pods -l version=$NEW_VERSION -o jsonpath='{.items[0].metadata.name}')
    kubectl port-forward $NEW_POD 8080:8080 &
    PORT_FORWARD_PID=$!

    sleep 5

    # Run tests
    curl -f http://localhost:8080/health || {
        echo "Health check failed!"
        kill $PORT_FORWARD_PID
        exit 1
    }

    curl -f http://localhost:8080/api/test || {
        echo "API test failed!"
        kill $PORT_FORWARD_PID
        exit 1
    }

    kill $PORT_FORWARD_PID

    # Switch traffic
    echo "Tests passed! Switching traffic to $NEW_VERSION..."
    kubectl patch service myapp -p "{\"spec\":{\"selector\":{\"version\":\"$NEW_VERSION\"}}}"

    echo "Deployment complete! Traffic now on $NEW_VERSION"
    echo "Old version ($CURRENT_VERSION) still running for quick rollback"
    ```

=== "Database Migrations"
    **Handling database changes in blue-green:**

    ```
    Strategy 1: Backward-Compatible Migrations

    v1.0 (Blue):          v2.0 (Green):
    users table           users table (same)
    - id                  - id
    - name                - name
    - email               - email
                          - phone (NEW - nullable!)

    Step 1: Add nullable column
    Step 2: Deploy Green (can read old + new data)
    Step 3: Switch traffic
    Step 4: Backfill phone numbers
    Step 5: Make column NOT NULL (later deployment)
    ```

    ```sql
    -- Migration for Blue-Green compatibility
    -- Phase 1: Add nullable column
    ALTER TABLE users ADD COLUMN phone VARCHAR(20) NULL;

    -- Application code must handle NULL phone
    SELECT id, name, email, COALESCE(phone, '') as phone
    FROM users;

    -- After traffic switch, backfill in background
    UPDATE users SET phone = '' WHERE phone IS NULL;

    -- Phase 2 (next deployment): Make NOT NULL
    ALTER TABLE users MODIFY COLUMN phone VARCHAR(20) NOT NULL;
    ```

=== "AWS Blue-Green with ALB"
    ```python
    import boto3

    def blue_green_switch():
        elbv2 = boto3.client('elbv2')

        # Get target groups
        blue_tg = 'arn:aws:elasticloadbalancing:...:targetgroup/blue'
        green_tg = 'arn:aws:elasticloadbalancing:...:targetgroup/green'

        # Get listener (currently pointing to blue)
        listener_arn = 'arn:aws:elasticloadbalancing:...:listener/...'

        # Check green health
        green_health = elbv2.describe_target_health(
            TargetGroupArn=green_tg
        )

        healthy_count = sum(1 for target in green_health['TargetHealthDescriptions']
                           if target['TargetHealth']['State'] == 'healthy')

        if healthy_count < 3:
            print(f"Only {healthy_count} healthy targets in green. Aborting!")
            return False

        # Switch listener to green
        elbv2.modify_listener(
            ListenerArn=listener_arn,
            DefaultActions=[{
                'Type': 'forward',
                'TargetGroupArn': green_tg
            }]
        )

        print("Traffic switched to green!")
        return True

    # Rollback function
    def rollback_to_blue():
        elbv2 = boto3.client('elbv2')
        elbv2.modify_listener(
            ListenerArn='arn:...:listener/...',
            DefaultActions=[{
                'Type': 'forward',
                'TargetGroupArn': 'arn:...:targetgroup/blue'
            }]
        )
        print("Rolled back to blue!")
    ```

=== "Advantages & Disadvantages"
    **Advantages:**
    - ✅ Zero downtime
    - ✅ Instant rollback (seconds)
    - ✅ Test in production environment
    - ✅ Full testing before switching traffic
    - ✅ Easy to rollback (just switch back)

    **Disadvantages:**
    - ❌ 2x infrastructure cost (during deployment)
    - ❌ Database migrations require careful planning
    - ❌ Stateful applications are complex
    - ❌ Need identical blue/green environments

---

## Canary Deployment

=== "How It Works"
    **Gradually increase traffic to new version**

    ```
    Phase 1: 5% Canary
    ┌─────────────────┐
    │  Load Balancer  │
    └────┬────────┬───┘
         │        │
        95%      5%
         ↓        ↓
    ┌────────┐ ┌────────┐
    │  v1.0  │ │  v2.0  │ (Canary)
    │  (95%) │ │  (5%)  │
    └────────┘ └────────┘

    Monitor: Error rate, latency, CPU, memory

    Phase 2: Increase if healthy (25%)
    Phase 3: Increase to 50%
    Phase 4: Increase to 100% (full rollout)

    If ANY issues: Instant rollback to 0%
    ```

=== "Implementation with Istio"
    ```yaml
    apiVersion: networking.istio.io/v1beta1
    kind: VirtualService
    metadata:
      name: myapp-canary
    spec:
      hosts:
      - myapp.example.com
      http:
      - match:
        - headers:
            # Route internal traffic to canary
            x-canary:
              exact: "true"
        route:
        - destination:
            host: myapp
            subset: v2
      - route:
        - destination:
            host: myapp
            subset: v1
          weight: 95
        - destination:
            host: myapp
            subset: v2
          weight: 5    # Start with 5% canary

    ---
    apiVersion: networking.istio.io/v1beta1
    kind: DestinationRule
    metadata:
      name: myapp
    spec:
      host: myapp
      subsets:
      - name: v1
        labels:
          version: v1
      - name: v2
        labels:
          version: v2
    ```

    **Automated canary with Flagger:**
    ```yaml
    apiVersion: flagger.app/v1beta1
    kind: Canary
    metadata:
      name: myapp
    spec:
      targetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: myapp
      service:
        port: 80
      analysis:
        interval: 1m
        threshold: 5
        maxWeight: 50
        stepWeight: 10
        metrics:
        - name: request-success-rate
          thresholdRange:
            min: 99
          interval: 1m
        - name: request-duration
          thresholdRange:
            max: 500
          interval: 1m
      webhooks:
        - name: load-test
          url: http://flagger-loadtester/
          timeout: 5s
          metadata:
            type: cmd
            cmd: "hey -z 1m -q 10 -c 2 http://myapp/"
    ```

=== "Application-Level Canary"
    ```javascript
    // Feature flag-based canary
    const unleash = require('unleash-client');

    unleash.initialize({
        url: 'http://unleash.example.com/api/',
        appName: 'myapp',
        instanceId: process.env.INSTANCE_ID
    });

    app.get('/api/data', async (req, res) => {
        const useNewVersion = unleash.isEnabled('new-api-version', {
            userId: req.user.id,
            sessionId: req.sessionID
        });

        if (useNewVersion) {
            // New version (canary)
            return res.json(await getDataV2(req.params));
        } else {
            // Old version (stable)
            return res.json(await getDataV1(req.params));
        }
    });
    ```

    **Gradual rollout script:**
    ```bash
    #!/bin/bash
    # Gradual canary rollout with monitoring

    PERCENTAGES=(5 10 25 50 100)

    for pct in "${PERCENTAGES[@]}"; do
        echo "Rolling out to $pct%..."

        # Update Istio VirtualService
        kubectl patch virtualservice myapp-canary --type merge -p "
        spec:
          http:
          - route:
            - destination:
                host: myapp
                subset: v1
              weight: $((100 - pct))
            - destination:
                host: myapp
                subset: v2
              weight: $pct
        "

        # Wait and monitor
        echo "Monitoring for 5 minutes at $pct%..."
        sleep 300

        # Check error rate
        ERROR_RATE=$(promtool query instant \
            'rate(http_requests_total{status=~"5.."}[5m])' | \
            jq '.data.result[0].value[1]' | bc)

        if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
            echo "Error rate too high ($ERROR_RATE)! Rolling back..."
            kubectl patch virtualservice myapp-canary --type merge -p "
            spec:
              http:
              - route:
                - destination:
                    host: myapp
                    subset: v1
                  weight: 100
            "
            exit 1
        fi

        echo "$pct% rollout successful!"
    done

    echo "Full canary deployment complete!"
    ```

=== "Monitoring Canary"
    **Key metrics to track:**

    ```yaml
    # Prometheus queries for canary monitoring

    # Error rate comparison
    sum(rate(http_requests_total{version="v2",status=~"5.."}[5m])) /
    sum(rate(http_requests_total{version="v2"}[5m]))
    vs
    sum(rate(http_requests_total{version="v1",status=~"5.."}[5m])) /
    sum(rate(http_requests_total{version="v1"}[5m]))

    # Latency comparison (P99)
    histogram_quantile(0.99,
      rate(http_request_duration_seconds_bucket{version="v2"}[5m]))
    vs
    histogram_quantile(0.99,
      rate(http_request_duration_seconds_bucket{version="v1"}[5m]))

    # CPU usage comparison
    avg(container_cpu_usage_seconds_total{version="v2"})
    vs
    avg(container_cpu_usage_seconds_total{version="v1"})
    ```

    **Automated rollback criteria:**
    ```
    Rollback if:
    - Error rate > 1% higher than v1
    - P99 latency > 200ms slower than v1
    - CPU usage > 50% higher than v1
    - Memory usage > 30% higher than v1
    - Any 5xx errors > 10/minute
    ```

=== "Advantages & Disadvantages"
    **Advantages:**
    - ✅ Lowest risk (blast radius limited)
    - ✅ Real production testing
    - ✅ Gradual rollout allows monitoring
    - ✅ Fast rollback (reduce to 0%)
    - ✅ Can target specific user segments

    **Disadvantages:**
    - ❌ Complex routing logic required
    - ❌ Need robust monitoring and alerting
    - ❌ Longer deployment process
    - ❌ Both versions run simultaneously

---

## A/B Testing Deployment

=== "How It Works"
    **Route traffic based on user attributes for testing**

    ```
    Similar to canary, but based on user segments:

    ┌─────────────────┐
    │  Load Balancer  │
    └────┬────────┬───┘
         │        │
    ┌────┴────┐ ┌┴───────┐
    │ Group A │ │Group B │
    │  v1.0   │ │  v2.0  │
    │ (50%)   │ │ (50%)  │
    └─────────┘ └────────┘

    Criteria:
    - User ID (consistent routing)
    - Geographic location
    - User agent (mobile vs desktop)
    - Feature flags
    ```

=== "Implementation"
    ```javascript
    // A/B testing with feature flags
    app.get('/checkout', async (req, res) => {
        const userId = req.user.id;

        // Consistent hashing: same user always gets same version
        const hash = crypto.createHash('md5')
            .update(userId.toString())
            .digest('hex');
        const hashInt = parseInt(hash.substr(0, 8), 16);
        const bucket = hashInt % 100;

        if (bucket < 50) {
            // Version A (control)
            res.render('checkout-v1', { layout: 'classic' });
        } else {
            // Version B (experiment)
            res.render('checkout-v2', { layout: 'modern' });
        }

        // Track which version shown
        analytics.track('checkout_shown', {
            userId,
            version: bucket < 50 ? 'A' : 'B'
        });
    });
    ```

---

## Interview Talking Points

**Q: When would you choose blue-green over canary deployment?**

✅ **Strong Answer:**
> "I'd choose blue-green for critical applications where we need to test the new version thoroughly before sending any production traffic to it. For example, a payment processing system where even 1% of failed transactions is unacceptable. Blue-green lets us deploy the new version, run extensive smoke tests and load tests against it, and only switch when we're confident. The instant rollback is also valuable. However, I'd use canary deployment when we want to validate with real production traffic incrementally - like a recommendation engine where we can monitor whether the new algorithm performs better with 5% of users before rolling out to everyone."

**Q: How do you handle database migrations in blue-green deployments?**

✅ **Strong Answer:**
> "The key is making migrations backward-compatible. I'd use an expand-contract pattern: first, expand the schema to support both old and new versions (like adding a nullable column), deploy the green environment that can work with both schemas, switch traffic, then contract by removing old columns in a later deployment. For example, if renaming a column, I'd add the new column, update the app to write to both columns, deploy, backfill data, then remove the old column. This ensures both blue and green can operate during the transition. For breaking changes, I'd use a maintenance window or consider if blue-green is the right strategy."

---

## Related Topics

- [CI/CD Pipelines](ci-cd.md) - Automate deployments
- [Containers & Orchestration](containers.md) - Package applications
- [Monitoring](../observability/monitoring.md) - Track deployment health
- [Load Balancing](../networking/load-balancers.md) - Traffic routing

---

**Deploy frequently, deploy safely! 🚀**
