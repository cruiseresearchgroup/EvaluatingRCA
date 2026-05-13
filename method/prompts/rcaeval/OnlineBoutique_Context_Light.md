# Online Boutique — Operational System Documentation

**Asset:** Cloud-native e-commerce platform.
**Document scope:** Service inventory, request topology, and observability metrics for the Online Boutique system.

---

## System Overview

Online Boutique is an e-commerce platform deployed on Kubernetes. End users browse the product catalogue, add items to a shopping cart, and complete purchases through an online checkout flow. The system is composed of 11 services written in multiple languages (Go, Java, C#, Node.js, Python, Redis); each service is independently deployed and instrumented.

All services emit standard observability metrics — CPU usage, memory usage, and request latency — to a metrics historian. The historian publishes per-service columns following the convention `{service}_{metric_type}`, where metric types include `_cpu` (millicores), `_mem` (MB), and `_latency` / `_latency-90` (seconds, P90).

---

## Service Inventory

| Service | Language | Role | Calls |
|---|---|---|---|
| `frontend` | Go | Renders the web UI; orchestrates all backend services | all backend services |
| `cartservice` | C# | Manages shopping carts; uses Redis | redis-cart |
| `productcatalogservice` | Go | Lists and searches products (in-memory) | none |
| `currencyservice` | Node.js | Converts prices between currencies | none |
| `paymentservice` | Node.js | Processes (mock) credit-card payments | none |
| `shippingservice` | Go | Provides shipping cost estimates | none |
| `emailservice` | Python | Sends order confirmation emails (mock) | none |
| `checkoutservice` | Go | Orchestrates the full checkout flow | cart, product, currency, payment, shipping, email |
| `recommendationservice` | Python | Suggests related products | productcatalogservice |
| `adservice` | Java | Serves context-based advertisements | none |
| `redis-cart` | Redis | In-memory store for cart contents | none |

---

## Request Topology

```
User → frontend
         ├── productcatalogservice   (browse / search)
         ├── recommendationservice → productcatalogservice
         ├── currencyservice         (price conversion)
         ├── adservice               (ads)
         ├── cartservice → redis-cart
         └── checkoutservice
               ├── cartservice → redis-cart
               ├── productcatalogservice
               ├── currencyservice
               ├── paymentservice
               ├── shippingservice
               └── emailservice
```

---

## Metric Channel Reference

Each service exposes the following metric channels in the historian, named `{service}_{metric_type}`:

| Suffix | Description | Units |
|---|---|---|
| `_cpu` | CPU usage | millicores (m) |
| `_mem` | Resident memory usage | MB |
| `_latency` / `_latency-90` | P90 request latency | seconds |

**Example channel names:** `frontend_cpu`, `cartservice_latency`, `redis-cart_mem`, `checkoutservice_cpu`.
