# Real-time Communication

**Push data to clients instantly** | 🔌 WebSockets | 📡 SSE | 🔄 Long Polling

---

## Overview

*Content coming soon.*

Many systems need to push data to clients in real-time: chat messages, notifications, live scores, stock prices. The choice between WebSockets, SSE, and Long Polling depends on your use case.

---

## Topics to Cover

- **Long Polling** — Client polls repeatedly, server holds request until data available
- **Server-Sent Events (SSE)** — One-way server-to-client stream over HTTP
- **WebSockets** — Full-duplex bidirectional communication over TCP
- **Comparison** — Latency, scalability, browser support, connection overhead
- **Connection Management** — Heartbeats, reconnection, connection limits at scale
- **Scaling WebSockets** — Sticky sessions, pub/sub backends (Redis), connection state
- **Real-world Examples** — Slack (WebSocket), Twitter feed (SSE), chat apps, live dashboards

---

## Decision Framework

| Criterion | Long Polling | SSE | WebSocket |
|-----------|-------------|-----|-----------|
| Direction | Client → Server | Server → Client | Bidirectional |
| Complexity | Low | Low | Medium |
| Scalability | Poor | Good | Good |
| Use Case | Simple notifications | Live feeds, dashboards | Chat, gaming, collaboration |

---

## Interview Relevance

- Critical for: WhatsApp, Slack, notification system, live streaming designs
- Key trade-off: simplicity (SSE/polling) vs capability (WebSocket)
- Scaling challenge: maintaining millions of persistent connections

---

## Related Topics

- [API Design](api-design/index.md)
- [Messaging Patterns](messaging/patterns.md)
- [Load Balancing](../scalability/load-balancing.md)
