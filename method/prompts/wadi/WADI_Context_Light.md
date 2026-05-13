# Water Distribution System (WaDi) — Operational System Documentation

**Facility:** WaDi Water Distribution Plant (commissioned July 2016).
**Document scope:** Process architecture, instrumentation, control topology, and historian channel inventory of the WaDi water distribution system.

---

## Facility Information

The facility is an industry-compliant water distribution plant operating at a nominal throughput of 10 US gallons per minute (~38 l/min). It receives treated water from an upstream reverse-osmosis (RO) treatment facility and distributes it through a structured grid to consumer endpoints; water is then recollected and recirculated through a closed return loop.

All process variables and actuator states are recorded continuously by the plant historian. The historian carries approximately 123–127 channels at a 1-second sampling interval. Channels follow the tag-naming pattern `<stage>_<instrument_type>_<loop_number>_<value_suffix>`, where stage is `1`, `2`, `2A`, `2B`, or `3`; instrument type encodes function (`AIT` analyser, `LT` level, `FIT` flow, `PIT` pressure, `MV` motorised valve, `P` pump); and the suffix is `PV` for analogue process variables (omitted for discrete actuator states).

---

## System Overview

The plant is organised into three sequential process stages — primary storage (P1), domestic distribution (P2), and return collection (P3) — each with distinct hydraulic functions, dedicated control equipment, and instrumentation.

**Workflow:** raw water enters Stage 1 from the municipal mains and/or RO permeate from the upstream treatment plant; it is stored in two raw-water tanks where chemical dosing maintains water quality. Stage 2 pumps or gravity-feeds water from the raw tanks to two elevated reservoirs, which then distribute via the secondary grid to six consumer tanks (CT1–CT6) split into sub-networks 2A and 2B. When consumer tanks reach their upper level setpoint, water drains by gravity into a single Stage 3 return tank, which feeds back to Stage 1 through a metered recirculation line, closing the hydraulic loop.

The plant is designed for continuous, unattended operation. Process control is executed autonomously by PLCs and RTUs; a SCADA workstation provides supervisory monitoring with manual-override capability for any actuator.

---

## Subsystems & Signal Reference

### 1. Stage 1 — Primary Grid (Raw Water Storage)

Two raw-water tanks (~2,500 L each) buffer between upstream supply (municipal mains and/or RO permeate) and the downstream distribution grid. Each tank carries a level transmitter; inline analysers measure pH, conductivity, turbidity, residual chlorine, and dissolved oxygen, and chemical dosing pumps run in closed-loop feedback against those readings. Stage 1 PLC logic governs raw-water inlet valves and dosing-pump activation against tank-level thresholds and quality readings.

| Label / Pattern | Type | Description |
|---|---|---|
| `1_LT_001_PV` | sensor (level) | Raw-water tank level (mm or % of full scale). |
| `1_FIT_001_PV` | sensor (flow) | Inlet flow rate to Stage 1 (m³/h). |
| `1_AIT_001_PV` | sensor (quality) | pH analyser process value. |
| `1_AIT_002_PV` | sensor (quality) | Residual chlorine analyser. |
| `1_AIT_003_PV` | sensor (quality) | Turbidity analyser. |
| `1_AIT_004_PV` | sensor (quality) | Conductivity analyser. |
| `1_AIT_005_PV` | sensor (quality) | Dissolved oxygen analyser. |
| `1_MV_001` … `1_MV_004` | actuator | Motorised inlet/outlet valves (open/closed). |
| `1_P_001` … `1_P_006` | actuator | Booster and chemical-dosing pumps (running/stopped). |

### 2. Stage 2 — Secondary Grid (Elevated Reservoirs and Consumer Distribution)

Water from the Stage 1 raw tanks is pumped or gravity-fed into two elevated reservoirs that act as hydraulic head sources. From there it distributes through the secondary grid to six consumer tanks (CT1–CT6) in sub-networks 2A and 2B. Level transmitters monitor each reservoir; flow transmitters report volumetric flow on each consumer-tank supply line; pressure transmitters monitor key grid nodes. Motorised valves regulate flow to each consumer outlet, and drainage to the return grid is initiated automatically when a consumer tank reaches its upper fill setpoint. Consumer Tank 2 (CT2) is held at 0.00 m³/h by design — this zero-flow state is a deliberate setpoint, not a fault.

| Label / Pattern | Type | Description |
|---|---|---|
| `2_LT_001_PV` … `2_LT_002_PV` | sensor (level) | Elevated reservoir tank levels. |
| `2_FIT_001_PV` … `2_FIT_006_PV` | sensor (flow) | Volumetric flow on consumer-tank supply lines (m³/h). |
| `2_PIT_001_PV` … `2_PIT_003_PV` | sensor (pressure) | Line pressure at key secondary-grid nodes (bar/kPa). |
| `2A_AIT_001_PV` … `2A_AIT_004_PV` | sensor (quality) | Sub-network 2A water-quality analyser values. |
| `2B_AIT_001_PV` … `2B_AIT_004_PV` | sensor (quality) | Sub-network 2B water-quality analyser values. |
| `2_MV_001` … `2_MV_009` | actuator | Motorised valves regulating flow to consumer tanks. |
| `2_P_001` … `2_P_004` | actuator | Booster pumps lifting water to elevated reservoirs. |
| `2_MCV_*` | actuator | Modulating control valves on consumer outlets. |
| `2_LS_001_AL` / `2_LS_101_AL` | state | Discrete level-switch alarms (high/low). |

### 3. Stage 3 — Return Water Grid

A single return tank receives gravity drainage from the consumer tanks and returns water to the Stage 1 raw tanks through a metered recirculation line, closing the hydraulic loop. The return tank is monitored by a level transmitter and water-quality instrumentation; flow through the return line is metered.

| Label / Pattern | Type | Description |
|---|---|---|
| `3_LT_001_PV` | sensor (level) | Return tank level. |
| `3_FIT_001_PV` | sensor (flow) | Return-line flow rate (m³/h). |
| `3_AIT_005_PV` | sensor (quality) | Stage 3 water-quality analyser. |
| `3_MV_001` … `3_MV_003` | actuator | Motorised valves on the return path. |
| `3_P_001` … `3_P_004` | actuator | Recirculation pumps from return tank back to Stage 1. |

### 4. Plant Mode and SCADA State

Discrete plant-wide indicators reported to the historian alongside process measurements.

| Label / Pattern | Type | Description |
|---|---|---|
| `PLANT_START_STOP_LOG` | state | Plant-wide start/stop log. |
| `TOTAL_CONS_REQUIRED_FLOW` | setpoint | Aggregate consumer demand (m³/h). |
| `LEAK_DIFF_PRESSURE` | sensor | Differential-pressure indicator used for leak detection. |
