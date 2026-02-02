# Containers & Orchestration

**Package once, run anywhere** | 🐳 Docker | ☸️ Kubernetes | 📦 Containerization

---

## Overview

Containers package applications with all their dependencies, ensuring consistent behavior across development, testing, and production environments.

**Why Containers?**
- Consistent environments
- Faster deployments
- Better resource utilization
- Simplified dependency management

---

## Docker Fundamentals

=== "Dockerfile Basics"
    ```dockerfile
    # Multi-stage build for smaller images
    FROM node:18-alpine AS builder
    WORKDIR /app
    COPY package*.json ./
    RUN npm ci --only=production
    COPY . .
    RUN npm run build

    # Production image
    FROM node:18-alpine
    WORKDIR /app

    # Run as non-root user
    RUN addgroup -g 1001 -S nodejs && \
        adduser -S nodejs -u 1001

    COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
    COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
    COPY --chown=nodejs:nodejs package.json ./

    USER nodejs
    EXPOSE 3000
    
    HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
        CMD node healthcheck.js

    CMD ["node", "dist/server.js"]
    ```

    **Best Practices:**
    ```dockerfile
    # ✅ Good practices

    # 1. Use specific version tags
    FROM node:18.16.0-alpine
    # NOT: FROM node:latest

    # 2. Minimize layers
    RUN apt-get update && apt-get install -y \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*
    # NOT: Multiple RUN commands

    # 3. Leverage build cache
    COPY package*.json ./
    RUN npm ci
    COPY . .
    # Copy package.json first (changes less frequently)

    # 4. Use .dockerignore
    # .dockerignore:
    node_modules
    npm-debug.log
    .git
    .env
    ```

=== "Docker Commands"
    ```bash
    # Build image
    docker build -t myapp:latest .

    # Build with build args
    docker build \
        --build-arg NODE_ENV=production \
        --build-arg VERSION=1.2.3 \
        -t myapp:1.2.3 .

    # Run container
    docker run -d \
        --name myapp \
        -p 3000:3000 \
        -e DATABASE_URL=postgres://... \
        --restart unless-stopped \
        myapp:latest

    # View logs
    docker logs -f myapp

    # Execute command in running container
    docker exec -it myapp sh

    # View resource usage
    docker stats

    # Clean up
    docker system prune -a
    ```

=== "Docker Compose"
    ```yaml
    version: '3.8'

    services:
      app:
        build:
          context: .
          dockerfile: Dockerfile
          args:
            NODE_ENV: production
        ports:
          - "3000:3000"
        environment:
          - DATABASE_URL=postgres://postgres:password@db:5432/myapp
          - REDIS_URL=redis://redis:6379
        depends_on:
          - db
          - redis
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
          interval: 30s
          timeout: 3s
          retries: 3

      db:
        image: postgres:14-alpine
        environment:
          POSTGRES_DB: myapp
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: password
        volumes:
          - postgres-data:/var/lib/postgresql/data
        ports:
          - "5432:5432"

      redis:
        image: redis:7-alpine
        ports:
          - "6379:6379"
        volumes:
          - redis-data:/data

    volumes:
      postgres-data:
      redis-data:
    ```

---

## Kubernetes Fundamentals

=== "Core Concepts"
    ```
    Kubernetes Architecture:

    Master Node:
    ├── API Server (kubectl talks to this)
    ├── Scheduler (assigns Pods to Nodes)
    ├── Controller Manager (maintains desired state)
    └── etcd (stores cluster state)

    Worker Nodes:
    ├── kubelet (manages Pods)
    ├── kube-proxy (networking)
    └── Container Runtime (Docker/containerd)

    Core Resources:
    - Pod: Smallest deployable unit
    - Deployment: Manages Pods
    - Service: Exposes Pods
    - ConfigMap: Configuration data
    - Secret: Sensitive data
    - Ingress: HTTP routing
    ```

=== "Pod"
    ```yaml
    apiVersion: v1
    kind: Pod
    metadata:
      name: myapp-pod
      labels:
        app: myapp
        version: v1
    spec:
      containers:
      - name: myapp
        image: myapp:1.2.3
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: database-url
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
    ```

=== "Deployment"
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: myapp
      labels:
        app: myapp
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: myapp
      strategy:
        type: RollingUpdate
        rollingUpdate:
          maxSurge: 1
          maxUnavailable: 1
      template:
        metadata:
          labels:
            app: myapp
            version: v1
        spec:
          containers:
          - name: myapp
            image: myapp:1.2.3
            ports:
            - containerPort: 3000
            env:
            - name: NODE_ENV
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: database-url
            resources:
              requests:
                memory: "256Mi"
                cpu: "200m"
              limits:
                memory: "512Mi"
                cpu: "500m"
            livenessProbe:
              httpGet:
                path: /health
                port: 3000
              initialDelaySeconds: 30
              periodSeconds: 10
              failureThreshold: 3
            readinessProbe:
              httpGet:
                path: /ready
                port: 3000
              initialDelaySeconds: 10
              periodSeconds: 5
              failureThreshold: 3
    ```

=== "Service"
    ```yaml
    # LoadBalancer Service (external access)
    apiVersion: v1
    kind: Service
    metadata:
      name: myapp
    spec:
      type: LoadBalancer
      selector:
        app: myapp
      ports:
      - protocol: TCP
        port: 80
        targetPort: 3000
      sessionAffinity: ClientIP

    ---
    # ClusterIP Service (internal only)
    apiVersion: v1
    kind: Service
    metadata:
      name: myapp-internal
    spec:
      type: ClusterIP
      selector:
        app: myapp
      ports:
      - protocol: TCP
        port: 3000
        targetPort: 3000
    ```

=== "ConfigMap & Secret"
    ```yaml
    # ConfigMap (non-sensitive data)
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: myapp-config
    data:
      API_URL: "https://api.example.com"
      LOG_LEVEL: "info"
      config.json: |
        {
          "feature_flags": {
            "new_ui": true
          }
        }

    ---
    # Secret (sensitive data)
    apiVersion: v1
    kind: Secret
    metadata:
      name: myapp-secrets
    type: Opaque
    data:
      # base64 encoded values
      database-url: cG9zdGdyZXM6Ly8uLi4=
      api-key: c2VjcmV0LWtleQ==

    # Create secret from file
    # kubectl create secret generic myapp-secrets \
    #   --from-file=database-url=./db-url.txt \
    #   --from-literal=api-key=secret-key
    ```

=== "Ingress"
    ```yaml
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: myapp-ingress
      annotations:
        kubernetes.io/ingress.class: "nginx"
        cert-manager.io/cluster-issuer: "letsencrypt-prod"
        nginx.ingress.kubernetes.io/rate-limit: "100"
    spec:
      tls:
      - hosts:
        - example.com
        - www.example.com
        secretName: myapp-tls
      rules:
      - host: example.com
        http:
          paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: myapp
                port:
                  number: 80
      - host: api.example.com
        http:
          paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: myapp-api
                port:
                  number: 80
    ```

---

## Kubernetes Commands

```bash
# Get cluster info
kubectl cluster-info
kubectl get nodes

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# View resources
kubectl get pods
kubectl get deployments
kubectl get services
kubectl get all

# Describe resource
kubectl describe pod myapp-pod-123
kubectl describe deployment myapp

# View logs
kubectl logs myapp-pod-123
kubectl logs -f myapp-pod-123  # Follow
kubectl logs myapp-pod-123 -c container-name  # Specific container

# Execute command in pod
kubectl exec -it myapp-pod-123 -- sh
kubectl exec myapp-pod-123 -- env

# Port forward (for debugging)
kubectl port-forward pod/myapp-pod-123 3000:3000
kubectl port-forward service/myapp 3000:80

# Scale deployment
kubectl scale deployment myapp --replicas=5

# Update image
kubectl set image deployment/myapp myapp=myapp:1.2.4

# Rollout management
kubectl rollout status deployment/myapp
kubectl rollout history deployment/myapp
kubectl rollout undo deployment/myapp
kubectl rollout undo deployment/myapp --to-revision=2

# Delete resources
kubectl delete pod myapp-pod-123
kubectl delete -f deployment.yaml
kubectl delete all -l app=myapp

# Debug
kubectl get events
kubectl top nodes
kubectl top pods
```

---

## Resource Management

=== "Requests vs Limits"
    ```yaml
    resources:
      requests:
        memory: "256Mi"  # Guaranteed
        cpu: "250m"      # 0.25 CPU
      limits:
        memory: "512Mi"  # Maximum allowed
        cpu: "500m"      # 0.5 CPU

    Behavior:
    - Pod scheduled if node has 256Mi and 250m available
    - Pod can use up to 512Mi and 500m
    - If exceeds memory limit: killed (OOMKilled)
    - If exceeds CPU limit: throttled
    ```

=== "Quality of Service"
    ```
    QoS Classes:

    1. Guaranteed (highest priority)
       - requests == limits for all containers
       resources:
         requests:
           memory: "256Mi"
           cpu: "250m"
         limits:
           memory: "256Mi"
           cpu: "250m"

    2. Burstable (medium priority)
       - requests < limits
       resources:
         requests:
           memory: "128Mi"
         limits:
           memory: "256Mi"

    3. BestEffort (lowest priority)
       - No requests or limits specified
       - Evicted first when node runs out of resources
    ```

---

## Scaling Strategies

=== "Horizontal Pod Autoscaler"
    ```yaml
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: myapp-hpa
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: myapp
      minReplicas: 2
      maxReplicas: 10
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
      - type: Resource
        resource:
          name: memory
          target:
            type: Utilization
            averageUtilization: 80
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
          - type: Percent
            value: 50
            periodSeconds: 60
        scaleUp:
          stabilizationWindowSeconds: 0
          policies:
          - type: Percent
            value: 100
            periodSeconds: 15
    ```

=== "Cluster Autoscaler"
    ```yaml
    # Automatically adds/removes nodes based on demand
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: cluster-autoscaler
      namespace: kube-system
    data:
      autoscaler-config: |
        {
          "minNodes": 2,
          "maxNodes": 10,
          "scaleDownDelayAfterAdd": "10m",
          "scaleDownUnneededTime": "10m"
        }
    ```

---

## Interview Talking Points

**Q: What's the difference between Docker and Kubernetes?**

✅ **Strong Answer:**
> "Docker is a containerization platform that packages applications into containers, while Kubernetes is an orchestration platform that manages those containers at scale. Docker lets you build and run containers on a single machine, but Kubernetes manages hundreds of containers across multiple machines - handling deployment, scaling, networking, and self-healing. You use Docker to create the container images, then Kubernetes to deploy and manage them in production. They're complementary tools, not alternatives."

**Q: How do you handle secrets in Kubernetes?**

✅ **Strong Answer:**
> "I'd use Kubernetes Secrets for basic use cases, but they're only base64-encoded by default, not encrypted. For production, I'd enable encryption at rest in etcd and use external secret management like AWS Secrets Manager, HashiCorp Vault, or Google Secret Manager with tools like External Secrets Operator. I'd also implement RBAC to limit which services can access which secrets, use short-lived credentials where possible, and audit secret access. Never commit secrets to Git or include them in container images."

---

## Related Topics

- [Deployment Strategies](strategies.md) - Blue-green, canary
- [CI/CD Pipelines](ci-cd.md) - Automated deployments
- [Infrastructure as Code](infrastructure.md) - Provision clusters
- [Monitoring](../observability/monitoring.md) - Monitor containers

---

**Containerize everything, orchestrate at scale! 🐳**
