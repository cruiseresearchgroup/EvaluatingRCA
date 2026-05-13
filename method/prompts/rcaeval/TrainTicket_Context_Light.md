# Train Ticket — Operational System Documentation

**Asset:** Railway ticketing platform.
**Document scope:** Service inventory, request topology, and observability metrics for the Train Ticket system.

---

## System Overview

Train Ticket is a railway ticketing platform that handles end-to-end travel booking for both high-speed and normal-speed rail networks — search, ticket reservation, payment, cancellation, refund, rebooking, and trip-time add-ons such as in-trip food and travel insurance. The application is implemented as a polyglot microservice system (Java/Spring, Node.js, Python, Go) with each service deployed independently on Kubernetes.

The platform is composed of two classes of pods:

- **Application services** — stateless or lightly-stateful pods that serve user traffic and orchestrate business logic. Each application service is responsible for one functional domain (e.g., seat allocation, payment processing, order management).
- **Data-store pods** — every domain that owns persistent state runs its own dedicated database pod (MongoDB, with one MySQL store), co-deployed with the application service that owns it. Database pods receive read/write traffic from their owning application service and emit their own observability metrics.

All pods emit standard observability metrics — CPU usage, memory usage, request workload (rate), and response-latency percentiles — to a metrics historian. Channels follow the convention `{pod}_{metric_type}`; all pod names share the prefix `ts-`, e.g., `ts-travel-service_cpu`, `ts-order-service_latency-90`, `ts-travel-mongo_mem`.

---

## Application Services

### User-Facing / Entry Points
| Pod | Role |
|---|---|
| `ts-ui-dashboard` | Main web UI; aggregates all user-facing flows |
| `ts-gateway-service` | API gateway; routes frontend requests to backend |

### Travel & Search Domain
| Pod | Role |
|---|---|
| `ts-travel-service` | High-speed train routes (G/D trains) |
| `ts-travel2-service` | Normal-speed train routes (Z/T/K trains) |
| `ts-route-service` | Route definitions |
| `ts-route-plan-service` | Multi-leg route planning |
| `ts-station-service` | Station information |
| `ts-train-service` | Train type information |
| `ts-config-service` | System-wide configuration |

### Booking & Order Domain
| Pod | Role |
|---|---|
| `ts-order-service` | Orders for high-speed trains |
| `ts-order-other-service` | Orders for normal-speed trains |
| `ts-seat-service` | Seat availability and reservation |
| `ts-basic-service` | Basic order info (fares, train info) |
| `ts-ticketinfo-service` | Ticket information queries |
| `ts-ticket-office-service` | Ticket-office storefront |

### Payment & Finance Domain
| Pod | Role |
|---|---|
| `ts-inside-payment-service` | Internal payment processing |
| `ts-payment-service` | External payment gateway |
| `ts-price-service` | Fare calculation |

### User & Auth Domain
| Pod | Role |
|---|---|
| `ts-user-service` | User account management |
| `ts-auth-service` | Authentication and token management |
| `ts-verification-code-service` | Email verification codes |
| `ts-contacts-service` | Saved passenger contacts |
| `ts-avatar-service` | User avatar storage |

### Trip Management Domain
| Pod | Role |
|---|---|
| `ts-preserve-service` | Ticket-booking orchestrator (high-speed) |
| `ts-preserve-other-service` | Ticket-booking orchestrator (normal-speed) |
| `ts-cancel-service` | Ticket cancellation |
| `ts-rebook-service` | Ticket rebooking |
| `ts-execute-service` | Trip execution status updates |
| `ts-travel-plan-service` | Multi-trip planning |

### Add-on Services
| Pod | Role |
|---|---|
| `ts-food-service` | In-trip food ordering |
| `ts-food-map-service` | Food provider mapping by station |
| `ts-assurance-service` | Travel insurance |
| `ts-consign-service` | Baggage consignment |
| `ts-consign-price-service` | Baggage price calculation |
| `ts-security-service` | Security checks on bookings |

### Administrative & Notification
| Pod | Role |
|---|---|
| `ts-admin-basic-info-service` | Admin configuration |
| `ts-admin-order-service` | Admin order management |
| `ts-admin-route-service` | Admin route management |
| `ts-admin-travel-service` | Admin travel management |
| `ts-admin-user-service` | Admin user management |
| `ts-news-service` | News / promotions display |
| `ts-notification-service` | Email / SMS notifications |
| `ts-voucher-service` | Discount vouchers |

---

## Data-Store Pods

Each domain that owns persistent state has its own database pod following the naming pattern `ts-{domain}-mongo` (or `ts-voucher-mysql` for the only MySQL store). These pods receive read/write traffic from their owning application service and emit their own CPU and memory metrics.

```
ts-assurance-mongo         ts-order-mongo            ts-station-mongo
ts-auth-mongo              ts-order-other-mongo      ts-ticket-office-mongo
ts-config-mongo            ts-payment-mongo          ts-train-mongo
ts-consign-mongo           ts-price-mongo            ts-travel-mongo
ts-consign-price-mongo     ts-route-mongo            ts-travel2-mongo
ts-contacts-mongo          ts-security-mongo         ts-user-mongo
ts-food-mongo              ts-food-map-mongo         ts-voucher-mysql
ts-inside-payment-mongo
```

A database pod's CPU or memory anomaly is typically a downstream consequence of fault load on its owning application service. Conversely, an application service's latency anomaly may be caused by an underlying database pod (locking contention, disk pressure, slow query).

---

## Request Topology (Key Paths)

```
User → ts-ui-dashboard → ts-gateway-service
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        ts-auth-service  ts-travel-service  ts-preserve-service
              │               │                │
        ts-user-service  ts-route-service  ts-order-service
              │          ts-seat-service   ts-inside-payment-service
        ts-auth-mongo    ts-train-service  ts-seat-service
        ts-user-mongo    ts-config-service ts-contacts-service
                         ts-travel-mongo   ts-food-service
                         ts-route-mongo    ts-assurance-service
                                           ts-consign-service
                                           ts-order-mongo
                                           ts-inside-payment-mongo
                                           ts-food-mongo
                                           ts-assurance-mongo
                                           ts-consign-mongo
```

Booking is the deepest transactional path: it touches authentication, travel, multiple payment paths, and several add-ons, and each stage in turn reads/writes its data-store pod.

---

## Metric Channel Reference

Each pod exposes a subset of the following metric channels in the historian, named `{pod}_{metric_type}`:

| Suffix | Description | Units | Typical coverage |
|---|---|---|---|
| `_cpu` | CPU usage | millicores (m) | every pod |
| `_mem` | Resident memory usage | MB | every application pod, every database pod |
| `_workload` | Request rate / throughput | RPS | application services that receive HTTP traffic |
| `_latency-50` | P50 request latency | seconds | HTTP-facing application services |
| `_latency-90` | P90 request latency | seconds | HTTP-facing application services |

Coverage notes:
- **HTTP-facing application services** typically expose all five channels (`_cpu`, `_mem`, `_workload`, `_latency-50`, `_latency-90`).
- **Back-office and admin services** that don't serve user-facing traffic generally expose only `_cpu` / `_mem` (sometimes also `_workload`). Examples: `ts-cancel-service`, `ts-rebook-service`, `ts-news-service`, `ts-execute-service`, `ts-route-plan-service`, `ts-travel-plan-service`, `ts-ticket-office-service`, `ts-verification-code-service`, `ts-admin-order-service`, `ts-admin-route-service`, `ts-admin-user-service`, `ts-avatar-service`.
- **Data-store pods** (`ts-*-mongo`, `ts-voucher-mysql`) expose only `_cpu` / `_mem`. They don't serve HTTP traffic, so workload and latency aren't directly instrumented.

**Example channel names:** `ts-travel-service_cpu`, `ts-order-service_latency-90`, `ts-gateway-service_workload`, `ts-travel-mongo_mem`, `ts-voucher-mysql_cpu`.
