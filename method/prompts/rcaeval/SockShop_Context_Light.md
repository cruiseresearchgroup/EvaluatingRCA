# Sock Shop — Operational System Documentation

**Asset:** Cloud-native e-commerce platform (online sock retailer).
**Document scope:** Service inventory, request topology, and observability metrics for the Sock Shop system.

---

## System Overview

Sock Shop is an e-commerce platform that handles end-to-end retail flows for an online sock retailer — catalogue browsing, cart management, user authentication, checkout, payment, and asynchronous shipping dispatch. The application is implemented as a polyglot microservice system (Node.js, Java, Go, Erlang) with each service deployed independently on Kubernetes alongside its persistent stores (MySQL, MongoDB, Redis) and a RabbitMQ message broker.

The platform is composed of three classes of pods:

- **Application services** — stateless or lightly-stateful pods that serve user traffic and orchestrate business logic.
- **Data-store pods** — every domain that owns persistent state runs its own database pod (`*-db`) co-deployed with the application service that owns it.
- **Messaging infrastructure** — a RabbitMQ broker and its exporter handle asynchronous shipping dispatch.

All pods emit standard observability metrics — CPU usage, memory usage, request workload (rate), and response-latency percentiles — to a metrics historian. Channels follow the convention `{pod}_{metric_type}`.

---

## Application Services

| Pod | Language | Role | Calls |
|---|---|---|---|
| `front-end`    | Node.js | Renders UI; orchestrates backend calls | all backend services |
| `orders`       | Java    | Manages order creation and history     | shipping, payment, carts, user |
| `catalogue`    | Go      | Product catalogue (sock listings)      | catalogue-db |
| `carts`        | Java    | Shopping cart management               | carts-db |
| `user`         | Go      | User accounts and authentication       | user-db |
| `payment`      | Go      | Payment processing (mock)              | none |
| `shipping`     | Go      | Shipping dispatch via RabbitMQ         | rabbitmq |
| `queue-master` | Java    | Consumes shipping queue from RabbitMQ  | rabbitmq |

---

## Data-Store Pods

| Pod | Type | Role |
|---|---|---|
| `catalogue-db` | MySQL   | Persistent store for catalogue data |
| `carts-db`     | MongoDB | Persistent store for cart data |
| `user-db`      | MongoDB | Persistent store for user data |
| `orders-db`    | MongoDB | Persistent store for orders |
| `session-db`   | Redis   | Session storage for `front-end` |

A database pod's CPU or memory anomaly is typically a downstream consequence of fault load on its owning application service. Conversely, an application service's latency anomaly may be caused by an underlying database pod (locking contention, disk pressure, slow query).

---

## Messaging Infrastructure

| Pod | Type | Role |
|---|---|---|
| `rabbitmq`          | Erlang     | Message broker for asynchronous shipping |
| `rabbitmq-exporter` | Prometheus exporter | Scrapes broker statistics for the historian |

`shipping` publishes shipping-dispatch messages to `rabbitmq`; `queue-master` consumes them. A backlog or processing-rate anomaly therefore typically appears in `rabbitmq` first, then in `queue-master`.

---

## Request Topology

```
User → front-end
         ├── catalogue → catalogue-db      (browse socks)
         ├── user      → user-db           (login / register)
         ├── session-db                    (session cookies)
         ├── carts     → carts-db          (add to cart)
         └── orders                        (checkout)
               ├── user    → user-db
               ├── carts   → carts-db
               ├── payment                 (charge)
               └── shipping → rabbitmq → queue-master
```

Checkout is the deepest transactional path: it touches authentication, cart, payment, and the asynchronous shipping pipeline.

---

## Metric Channel Reference

Each pod exposes a subset of the following metric channels in the historian, named `{pod}_{metric_type}`:

| Suffix | Description | Units | Typical coverage |
|---|---|---|---|
| `_cpu`        | CPU usage | millicores (m) | every pod |
| `_mem`        | Resident memory usage | MB | every application pod and database pod (except `catalogue-db` and `rabbitmq-exporter`, which only emit `_cpu`) |
| `_workload`   | Request rate / throughput | RPS | application services that receive HTTP traffic (`front-end`, `orders`, `catalogue`, `carts`, `user`, `payment`, `shipping`, `queue-master`) |
| `_latency-50` | P50 request latency | seconds | HTTP-facing application services (same set as `_workload`) |
| `_latency-90` | P90 request latency | seconds | HTTP-facing application services (same set as `_workload`) |

Coverage notes:
- **HTTP-facing application services** (8 pods listed above) typically expose all five channels.
- **Database pods** (`catalogue-db`, `carts-db`, `user-db`, `orders-db`, `session-db`) and **broker** (`rabbitmq`) expose only `_cpu` / `_mem`. They don't serve HTTP traffic, so workload and latency aren't directly instrumented.
- **`catalogue-db`** and **`rabbitmq-exporter`** expose only `_cpu`.

**Example channel names:** `front-end_cpu`, `orders_latency-90`, `catalogue_workload`, `carts-db_mem`, `rabbitmq_cpu`.
