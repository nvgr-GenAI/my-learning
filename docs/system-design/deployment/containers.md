# Containers and Orchestration

Containers solve one of the oldest problems in software delivery: the gap between the environment where code is written and the environment where it runs. Before containers, teams relied on lengthy setup documents, configuration management tools, and the hope that staging would behave like production. Containers collapse that gap by packaging an application together with every dependency it needs -- libraries, runtime, system tools -- into a single portable artifact. The result is a unit of software that behaves identically whether it runs on a developer laptop, a CI server, or a fleet of production machines.

This shift toward containerized workloads has been so thorough that by 2024, over 90% of organizations running cloud infrastructure use containers in some form. Kubernetes alone manages workloads at companies like Google (handling billions of containers per week), Spotify (running over 150 microservices), and Airbnb (supporting 1,000+ services across hundreds of engineers).

## Containers vs Virtual Machines

| Aspect | Containers | Virtual Machines |
| --- | --- | --- |
| **Abstraction level** | OS-level process isolation | Full hardware virtualization |
| **Startup time** | Milliseconds to seconds | Minutes |
| **Image size** | Tens of MBs (Alpine-based) | Gigabytes |
| **Resource overhead** | Shares host kernel, minimal overhead | Runs full guest OS per VM |
| **Isolation** | Process and namespace isolation | Hardware-level isolation |
| **Density** | Hundreds per host | Tens per host |
| **Use case** | Microservices, CI/CD, stateless workloads | Legacy apps, multi-tenant security, different OS |
| **Real-world density** | Netflix runs ~5,000 containers per host | Typical enterprise: 20-50 VMs per host |

The key insight is that containers share the host operating system kernel, while VMs each carry their own kernel. This makes containers dramatically lighter, but it also means container isolation is weaker. If an attacker escapes a container, they reach the host kernel directly. VMs provide a stronger security boundary because the hypervisor sits between the guest and the host hardware.

In practice, many organizations use both: VMs for the infrastructure layer (each VM hosts a container runtime) and containers for the application layer. AWS ECS and GKE both run containers inside VMs under the hood.

---

=== "Containers"

    ## How Containers Work

    A container is a standard Linux process with three kernel features restricting what it can see and do: **namespaces** control visibility (each container sees its own filesystem, process tree, and network stack), **cgroups** control resource consumption (CPU, memory, I/O limits), and **union filesystems** enable efficient image layering.

    ### The "Works on My Machine" Problem

    Before containers, deploying a Python application meant ensuring the production server had the correct Python version, the exact library versions, the right system packages, and matching OS-level configurations. A discrepancy in any of these layers could cause failures that were impossible to reproduce locally.

    Containers eliminate this class of problems entirely. The developer builds an image that captures the full stack from OS base through application code. That exact image -- byte for byte -- is what runs in production. There is no configuration drift because there is no configuration to drift.

    ### Image Layers and Build Efficiency

    Container images are built in layers, where each instruction in a build file creates a new read-only layer stacked on top of the previous one. These layers are cached and shared across images, which has two important consequences.

    First, builds are fast. If you change only your application code but not your dependencies, the layer containing `npm install` or `pip install` results is reused from cache. Only the changed layer and everything above it gets rebuilt. This is why dependency installation should come before copying application source in a build file.

    Second, storage and transfer are efficient. If ten microservices all use the same base image, that base layer exists only once on disk. When you push a new version that changes only the top layer, only that delta gets transferred to the registry.

    ```
    Image Layer Stack:

    +---------------------------+  <-- Layer 4: Application code (changes often)
    +---------------------------+  <-- Layer 3: Installed dependencies (changes sometimes)
    +---------------------------+  <-- Layer 2: System packages (changes rarely)
    +---------------------------+  <-- Layer 1: Base OS (alpine, debian, etc.)
    ```

    ### Container Runtime

    The container runtime is the component that actually creates and runs containers. Docker popularized containers but is no longer the only option. Modern Kubernetes clusters typically use **containerd** (donated by Docker to the CNCF) or **CRI-O** (built specifically for Kubernetes). Both implement the Container Runtime Interface (CRI) that Kubernetes requires.

    The practical difference for most teams is small. Docker images built with a standard Dockerfile work with any OCI-compliant runtime. The shift from Docker to containerd in Kubernetes (completed in v1.24) was largely invisible to application developers.

    ### Docker Basics

    A minimal Dockerfile captures the essence of how images are built. This five-line example installs dependencies before copying source code, ensuring the expensive install step is cached across builds:

    ```dockerfile
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt . && RUN pip install -r requirements.txt
    COPY . .
    CMD ["python", "app.py"]
    ```

    In production, teams typically use multi-stage builds (a build stage compiles or bundles the application, then a final stage copies only the artifacts into a minimal base image) and run processes as non-root users. These practices keep images small and reduce the attack surface.

    ### Security Considerations

    Container security operates on several levels. At the image level, using minimal base images like Alpine (5 MB) instead of full Ubuntu (72 MB) reduces the number of packages that could contain vulnerabilities. Scanning tools like Trivy or Snyk examine image layers for known CVEs before deployment.

    At the runtime level, containers should run as non-root users, mount filesystems as read-only where possible, and drop unnecessary Linux capabilities. Kubernetes PodSecurityStandards (replacing the older PodSecurityPolicies) enforce these constraints cluster-wide.

    At the supply chain level, image signing with tools like Cosign and Notary ensures that only trusted images reach production. Companies like Shopify and GitHub enforce signed images in their deployment pipelines to prevent tampering between build and deploy.

=== "Orchestration"

    ## Why You Need Orchestration

    Running a single container on a single machine is straightforward. The challenge emerges when you need to run hundreds of containers across dozens of machines and answer questions like: which machine has enough resources for this container? What happens when a machine fails? How do containers find each other on the network? How do you roll out a new version without downtime?

    Container orchestration platforms answer all of these questions. They accept a description of your desired state ("run three copies of this service, each with 512 MB of memory, behind a load balancer") and continuously work to make reality match that description. If a container crashes, the orchestrator restarts it. If a node fails, the orchestrator reschedules the affected containers onto healthy nodes.

    Google developed the first generation of this technology internally (called Borg, managing billions of containers per week) and then released an open-source successor called Kubernetes in 2014. Kubernetes has since become the industry standard, with managed offerings from every major cloud provider.

    ### Kubernetes Core Concepts

    **Pods** are the smallest deployable unit in Kubernetes. A pod contains one or more containers that share a network namespace (they communicate over localhost) and storage volumes. Most pods run a single application container, but sidecar patterns are common -- for example, a main application container alongside a log-shipping container or a service mesh proxy.

    **Deployments** manage the lifecycle of pods. Rather than creating pods directly, you declare a deployment that specifies the container image, replica count, resource requests, and health checks. The deployment controller ensures the correct number of healthy pods are always running and handles rolling updates when you push a new image version.

    **Services** provide stable network identities for a set of pods. Since pods are ephemeral (they get new IP addresses when restarted), a service acts as a permanent internal DNS name and load balancer. A service named `payment-api` in the `production` namespace is reachable at `payment-api.production.svc.cluster.local` from any pod in the cluster.

    **Namespaces** partition a cluster into logical segments. Teams commonly use namespaces to separate environments (dev, staging, production) or organizational boundaries (team-a, team-b). Resource quotas and network policies can be applied per namespace, providing both organizational clarity and security boundaries.

    ```
    Kubernetes Cluster Architecture:

    +---------------------------------------------------------------+
    |  Control Plane                                                 |
    |  +------------+  +-----------+  +------------+  +-----------+ |
    |  | API Server |  | Scheduler |  | Controller |  |   etcd    | |
    |  | (gateway)  |  | (placement)|  | Manager   |  | (state)   | |
    |  +------------+  +-----------+  +------------+  +-----------+ |
    +---------------------------------------------------------------+
            |                    |                    |
    +---------------+  +---------------+  +---------------+
    | Worker Node 1 |  | Worker Node 2 |  | Worker Node 3 |
    | +-----------+ |  | +-----------+ |  | +-----------+ |
    | | kubelet   | |  | | kubelet   | |  | | kubelet   | |
    | +-----------+ |  | +-----------+ |  | +-----------+ |
    | | kube-proxy| |  | | kube-proxy| |  | | kube-proxy| |
    | +-----------+ |  | +-----------+ |  | +-----------+ |
    | |  Pod A    | |  | |  Pod B    | |  | |  Pod A    | |
    | |  Pod C    | |  | |  Pod A    | |  | |  Pod D    | |
    | +-----------+ |  | +-----------+ |  | +-----------+ |
    +---------------+  +---------------+  +---------------+
    ```

    The control plane makes all scheduling and state decisions. The API server is the single entry point -- every kubectl command, every controller, and every kubelet communicates through it. The scheduler decides which node should run a new pod based on resource availability, affinity rules, and constraints. The controller manager runs a collection of control loops that watch the actual state of the cluster and take action to match the desired state stored in etcd.

    ### When Kubernetes Is Overkill

    Kubernetes introduces significant operational complexity: the control plane itself must be maintained (or paid for as a managed service), teams need to understand networking, RBAC, resource management, and the YAML configuration surface is large. For teams running fewer than ten services, the overhead often exceeds the benefit.

    A useful heuristic: if your application fits on a single machine and you do not need automated scaling, self-healing, or zero-downtime deployments, simpler tools like Docker Compose or a managed container service (ECS, Cloud Run) will get you to production faster with less operational burden. Kubernetes earns its complexity when you have many services, multiple teams, and requirements for automated scaling and resilience.

=== "Scaling and Networking"

    ## Scaling in Kubernetes

    ### Horizontal Pod Autoscaler

    The Horizontal Pod Autoscaler (HPA) adjusts the number of pod replicas based on observed metrics. In its simplest form, HPA watches CPU utilization: if average CPU across all pods exceeds a threshold (commonly 70%), it adds replicas; if utilization drops, it removes them.

    The autoscaler runs a control loop every 15 seconds by default. It calculates the desired replica count using the formula: `desired = ceil(current * (currentMetric / targetMetric))`. If you have 3 replicas at 90% CPU with a 70% target, the autoscaler calculates `ceil(3 * 90/70) = 4` and adds one pod.

    Modern HPA configurations (autoscaling/v2) support multiple metrics simultaneously -- CPU, memory, and custom metrics like request queue depth or response latency. This is critical because CPU is not always the right scaling signal. A service that is I/O-bound or waiting on database connections may need more replicas even at low CPU.

    The **Cluster Autoscaler** operates one level up: when the scheduler cannot place a pod because no node has sufficient resources, the cluster autoscaler provisions a new node from the cloud provider. When nodes are underutilized for a configurable period (typically 10 minutes), it drains and removes them. This combination of pod-level and node-level autoscaling creates an elastic infrastructure that scales from development workloads to production traffic spikes.

    ```
    Autoscaling Flow:

    Traffic Spike
        |
        v
    Metrics Server detects CPU > 70%
        |
        v
    HPA increases desired replicas: 3 --> 6
        |
        v
    Scheduler: "No room on existing nodes"
        |
        v
    Cluster Autoscaler provisions new node
        |
        v
    New pods scheduled on new node
        |
        v
    Traffic handled, latency returns to normal
    ```

    ### Service Discovery and Networking

    Every service in Kubernetes gets a DNS entry automatically. When a pod in the `checkout` service needs to call the `inventory` service, it simply sends requests to `http://inventory:8080`. The cluster DNS (CoreDNS) resolves this name to the ClusterIP of the inventory service, and kube-proxy routes traffic to a healthy pod behind that service.

    There are three service types that handle different networking scenarios. **ClusterIP** (the default) exposes the service only within the cluster -- suitable for service-to-service communication. **NodePort** opens a port on every node in the cluster, allowing external traffic to reach the service through any node's IP. **LoadBalancer** provisions a cloud load balancer (an AWS ELB, GCP load balancer, or Azure load balancer) that routes external traffic to the service.

    ### Ingress and External Traffic

    For HTTP traffic, Kubernetes Ingress provides a more sophisticated routing layer than LoadBalancer services. An Ingress resource defines rules that map hostnames and URL paths to backend services. A single Ingress controller (NGINX, Traefik, or a cloud-native option like AWS ALB Ingress) can handle routing for an entire cluster, terminating TLS, applying rate limits, and directing traffic based on request attributes.

    ```
    External Traffic Flow:

    Client Request: api.example.com/v2/orders
        |
        v
    Cloud Load Balancer (provisioned by Ingress)
        |
        v
    Ingress Controller (NGINX / Traefik)
        |--- api.example.com/v1/*  --> api-v1 Service
        |--- api.example.com/v2/*  --> api-v2 Service
        |--- app.example.com/*     --> frontend Service
        |
        v
    Service (ClusterIP) --> Pod 1, Pod 2, Pod 3
    ```

    ### Spotify's Kubernetes Migration

    Spotify migrated from a custom orchestration system to Kubernetes over several years, eventually running over 150 microservices on Kubernetes by 2020. Their migration illustrates both the benefits and the real cost of orchestration at scale.

    Before Kubernetes, Spotify's deployment tooling was custom-built and maintained by a dedicated infrastructure team. Each service had bespoke deployment scripts, and the system struggled with inconsistent environments across their 2,000+ engineers. After migrating, Spotify reported that deployment frequency increased by 200% because developers could use a standard interface (Kubernetes manifests and Helm charts) rather than service-specific deployment procedures.

    The migration was not without challenges. Spotify found that Kubernetes networking added latency compared to their previous bare-metal setup, requiring careful tuning of connection pooling and service mesh configuration. They also invested heavily in developer tooling (Backstage, their open-source developer portal, was born partly from the need to make Kubernetes accessible to non-infrastructure engineers).

    The key lesson from Spotify's experience: Kubernetes pays off at scale with many teams and services, but it requires investment in platform engineering to make it productive. The raw Kubernetes API is too complex for most application developers to use directly.

=== "Alternatives"

    ## Container Orchestration Alternatives

    Kubernetes is not the only option for running containers in production. Several alternatives trade some of Kubernetes' flexibility and power for simplicity, lower operational burden, or tighter cloud integration. Choosing the right tool depends on team size, number of services, scaling requirements, and how much infrastructure management you want to own.

    ### AWS ECS and Fargate

    Amazon's Elastic Container Service (ECS) offers a managed container orchestration experience tightly integrated with the AWS ecosystem. ECS organizes containers into **task definitions** (analogous to Kubernetes pods) and **services** (analogous to deployments). It handles scheduling, scaling, and load balancer integration, but with a simpler mental model than Kubernetes.

    **Fargate** takes this a step further by removing the need to manage the underlying EC2 instances entirely. You specify CPU and memory for each task, and AWS handles provisioning, patching, and scaling the infrastructure. You pay per vCPU-second and per GB-second, with no idle server costs.

    Companies like Samsung, Turner Broadcasting, and Capital One run production workloads on ECS/Fargate. The service works well for teams that are already invested in AWS and want container orchestration without managing a Kubernetes control plane. The trade-off is vendor lock-in and less flexibility in networking and scheduling compared to Kubernetes.

    Typical cost comparison: a small team running 5 services on Fargate pays $150-400/month with no infrastructure management overhead, versus $73/month for an EKS control plane alone plus EC2 node costs plus the operational time to manage Kubernetes.

    ### Docker Compose

    Docker Compose defines multi-container applications in a single YAML file and runs them with one command. It is the standard tool for local development environments (spinning up an application alongside its database, cache, and message queue) and works surprisingly well for small production deployments.

    For a startup running 2-5 services on a single server, Docker Compose with a process manager like systemd provides a production-ready setup without orchestration complexity. Companies at this scale rarely need automated pod scheduling, self-healing across nodes, or horizontal autoscaling -- a simple restart policy and health checks are sufficient.

    The ceiling for Docker Compose is a single host. Once you need to run containers across multiple machines for availability or capacity, you have outgrown Compose and need either a managed service or a proper orchestrator.

    ### HashiCorp Nomad

    Nomad is a workload orchestrator that handles containers, VMs, Java applications, and raw executables with a single tool. Its architecture is simpler than Kubernetes: a single binary acts as both server and client, cluster setup takes minutes rather than hours, and the configuration language (HCL) is more concise than Kubernetes YAML.

    Nomad does not include built-in service discovery, secret management, or networking -- those are handled by other HashiCorp tools (Consul, Vault) or external systems. This modular approach means you adopt only the components you need, but it also means more integration work for a full-featured platform.

    Roblox runs one of the largest Nomad deployments, managing millions of containers across their game server infrastructure. Cloudflare and CircleCI also use Nomad for specific workloads. Nomad tends to appeal to teams that find Kubernetes too opinionated or too complex but need multi-host orchestration.

    ### Decision Tree: Which Tool to Use

    ```
    How many services are you running?
        |
        +-- 1-3 services
        |       |
        |       +-- Single server sufficient?
        |               |
        |               +-- Yes --> Docker Compose
        |               +-- No  --> ECS/Fargate or Cloud Run
        |
        +-- 4-15 services
        |       |
        |       +-- Primarily on AWS?
        |               |
        |               +-- Yes --> ECS/Fargate
        |               +-- No  --> Managed Kubernetes (EKS/GKE/AKS)
        |
        +-- 15+ services
        |       |
        |       +-- Dedicated platform team?
        |               |
        |               +-- Yes --> Kubernetes (self-managed or managed)
        |               +-- No  --> Managed Kubernetes + platform tooling
        |
        +-- Mixed workloads (containers + VMs + batch)
                |
                +-- HashiCorp Nomad
    ```

    The most common mistake is adopting Kubernetes prematurely. A team of five engineers running three microservices does not need a multi-node Kubernetes cluster. Start with the simplest tool that meets your requirements and migrate to more powerful orchestration when the pain of the current approach exceeds the cost of the migration.

---

## Key Takeaways

Containers solve environment consistency by packaging applications with all dependencies into portable, reproducible units. The image layering system makes builds fast and storage efficient, while process-level isolation through namespaces and cgroups keeps containers lightweight compared to virtual machines.

Orchestration becomes necessary when you run many containers across multiple machines and need automated scheduling, scaling, self-healing, and networking. Kubernetes is the industry standard but carries significant operational complexity. Managed alternatives like ECS/Fargate reduce that burden at the cost of flexibility and portability.

The right tool depends on scale: Docker Compose for single-host development and small production, managed container services for small-to-medium production workloads, and Kubernetes for large-scale multi-team environments where its complexity is justified by the problems it solves.

## Related Topics

- [Deployment Strategies](strategies.md) - Blue-green, canary, and rolling deployments
- [CI/CD Pipelines](ci-cd.md) - Automating container builds and deployments
- [Infrastructure as Code](infrastructure.md) - Provisioning and managing clusters declaratively
- [Monitoring](../observability/monitoring.md) - Observability for containerized workloads
