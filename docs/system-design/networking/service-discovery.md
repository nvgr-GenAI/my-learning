# Service Discovery

**How services find and communicate with each other** | 🔍 Registry | 🏥 Health Checks | 🔄 Load Balancing

---

## Overview

*Content coming soon.*

In distributed systems with many services, hardcoding addresses doesn't scale. Service discovery automates how services register themselves and find each other.

---

## Topics to Cover

- **Client-side Discovery** — Client queries registry, picks instance (Netflix Eureka)
- **Server-side Discovery** — Load balancer queries registry (AWS ALB, Kubernetes)
- **Service Registry** — Central database of service instances (Consul, etcd, ZooKeeper)
- **Health Checking** — Heartbeats, TCP/HTTP checks, TTL-based expiry
- **DNS-based Discovery** — DNS SRV records, CoreDNS in Kubernetes
- **Comparison** — Client-side vs server-side vs DNS-based trade-offs
- **Real-world Examples** — Kubernetes kube-dns, Consul, Netflix Eureka, AWS Cloud Map

---

## Interview Relevance

- Comes up in any microservices design question
- Key decision: client-side vs server-side discovery
- Related to load balancing, health checks, and fault tolerance

---

## Related Topics

- [Load Balancers](load-balancers.md)
- [DNS](dns.md)
- [Microservices](../architecture/microservices.md)
- [Service Mesh](../architecture/service-mesh.md)
