# Secure Water Treatment (SWaT) — Operational System Documentation

**Facility:** SWaT testbed (commissioned 2015, six-stage water-treatment plant).
**Document scope:** Process architecture, instrumentation, control topology, and historian channel inventory of the SWaT facility.

---

## Facility Information

The facility is a six-stage water-treatment testbed that produces filtered, dechlorinated, and reverse-osmosis-purified water from a raw-water source. End-to-end throughput passes sequentially through Stage 1 (raw-water storage), Stage 2 (chemical dosing), Stage 3 (ultrafiltration, UF), Stage 4 (dechlorination by UV), Stage 5 (reverse osmosis, RO), and Stage 6 (backwash). RO permeate exits the plant; reject and backwash water are recirculated.

All process variables and actuator states are recorded continuously by the plant historian at 1-second resolution. The historian carries 51 channels covering level transmitters, flow transmitters, analyser transmitters, pressure transmitters, motorised valves, pumps, and a UV lamp. Channel names follow the pattern `<type><stage><loop>`, where `type` encodes the instrument function (`LIT` level, `FIT` flow, `AIT` analyser, `PIT` pressure, `DPIT` differential pressure, `MV` motorised valve, `P` pump, `UV` UV lamp), `stage` is the process stage `1`–`6`, and `loop` is a two-digit loop number (e.g., `LIT101` is the level transmitter in stage 1, loop 01).

---

## System Overview

**Process workflow:** raw water enters Stage 1 and is buffered in a raw-water tank. Stage 2 doses chemicals (sodium hypochlorite, hydrochloric acid) under closed-loop pH and ORP control. Stage 3 passes the conditioned water through ultrafiltration membranes; differential pressure across the membranes is monitored to detect fouling and trigger backwash. Stage 4 exposes the UF permeate to a UV lamp to dechlorinate residual hypochlorite. Stage 5 forces the dechlorinated water through RO membranes at high pressure; permeate is the product, reject is recirculated. Stage 6 manages the periodic backwash of the UF and RO membranes.

The plant runs in continuous, unattended operation. Each stage is governed by a dedicated PLC executing process logic against tank-level setpoints, flow setpoints, and quality thresholds. A SCADA workstation provides supervisory monitoring with manual-override capability for any actuator.

---

## Subsystems & Signal Reference

### 1. Stage 1 — Raw Water Supply and Storage

Raw water enters the plant from a public supply line and is buffered in a raw-water tank. Stage 1 PLC opens the inlet motorised valve to fill the tank when the level transmitter falls below a low setpoint, and closes it when the level reaches a high setpoint. Two transfer pumps lift water from the raw-water tank to Stage 2.

| Label | Type | Description |
|---|---|---|
| `LIT101` | sensor (level) | Raw-water tank level (mm). |
| `FIT101` | sensor (flow) | Inlet flow rate to the raw-water tank (m³/h). |
| `MV101` | actuator | Motorised inlet valve to the raw-water tank (open/closed). |
| `P101` | actuator | Raw-water transfer pump 1 (running/stopped). |
| `P102` | actuator | Raw-water transfer pump 2 — standby (running/stopped). |

### 2. Stage 2 — Pre-Treatment (Chemical Dosing)

Stage 2 conditions the raw water by dosing sodium hypochlorite (chlorination) and hydrochloric acid (pH control). Three analyser transmitters monitor pH (`AIT202`), conductivity (`AIT201`), and oxidation–reduction potential (`AIT203`) of the conditioned water. Six dosing pumps (`P201`–`P206`) inject the chemicals in closed-loop control against the analyser readings.

| Label | Type | Description |
|---|---|---|
| `FIT201` | sensor (flow) | Outflow rate from Stage 2 to Stage 3 (m³/h). |
| `AIT201` | sensor (quality) | Conductivity analyser (μS/cm). |
| `AIT202` | sensor (quality) | pH analyser. |
| `AIT203` | sensor (quality) | ORP analyser (mV). |
| `MV201` | actuator | Motorised valve into Stage 2 (open/closed). |
| `P201` … `P206` | actuator | Chemical dosing pumps (running/stopped). |

### 3. Stage 3 — Ultrafiltration (UF)

The UF stage removes suspended solids by forcing water through a hollow-fibre membrane. A differential pressure indicator (`DPIT301`) measures the pressure drop across the membrane; when it exceeds a threshold the PLC triggers a backwash sequence. Two pumps (`P301`, `P302`) feed the UF unit and four motorised valves (`MV301`–`MV304`) route water either through the UF membrane (forward flow) or through a backwash path.

| Label | Type | Description |
|---|---|---|
| `LIT301` | sensor (level) | UF feed-tank level (mm). |
| `FIT301` | sensor (flow) | UF feed flow rate (m³/h). |
| `DPIT301` | sensor (Δ pressure) | Differential pressure across the UF membrane (kPa). High value = membrane fouling. |
| `MV301` … `MV304` | actuator | Motorised valves on UF/backwash routing (open/closed). |
| `P301` | actuator | UF feed pump (running/stopped). |
| `P302` | actuator | UF transfer pump (running/stopped). |

### 4. Stage 4 — Dechlorination (UV)

The UV stage removes residual free chlorine from the UF permeate before the water enters the RO membranes (chlorine damages the RO film). A UV lamp (`UV401`) is energised when feed flow is present; analyser transmitters monitor conductivity (`AIT401`) and ORP (`AIT402`) of the dechlorinated water.

| Label | Type | Description |
|---|---|---|
| `LIT401` | sensor (level) | UF permeate / RO feed-tank level (mm). |
| `FIT401` | sensor (flow) | UF permeate flow rate to UV (m³/h). |
| `AIT401` | sensor (quality) | Hardness / conductivity analyser. |
| `AIT402` | sensor (quality) | ORP analyser of dechlorinated water (mV). |
| `UV401` | actuator | UV lamp on/off. |
| `P401` … `P404` | actuator | Stage-4 transfer pumps (running/stopped). |

### 5. Stage 5 — Reverse Osmosis (RO)

The RO stage forces dechlorinated water through a semi-permeable membrane at high pressure to produce purified product water. Pressure transmitters (`PIT501`–`PIT503`) monitor inlet, membrane, and outlet pressures; analyser transmitters (`AIT501`–`AIT504`) report quality of feed, permeate, reject, and product streams. Flow transmitters (`FIT501`–`FIT504`) measure feed, permeate, reject, and recirculation flows. Two high-pressure pumps (`P501`, `P502`) drive the RO membrane.

| Label | Type | Description |
|---|---|---|
| `PIT501` | sensor (pressure) | RO inlet pressure (kPa). |
| `PIT502` | sensor (pressure) | RO membrane pressure (kPa). |
| `PIT503` | sensor (pressure) | RO outlet / reject pressure (kPa). |
| `FIT501` | sensor (flow) | RO feed flow (m³/h). |
| `FIT502` | sensor (flow) | RO permeate flow (m³/h). |
| `FIT503` | sensor (flow) | RO reject flow (m³/h). |
| `FIT504` | sensor (flow) | RO recirculation flow (m³/h). |
| `AIT501` | sensor (quality) | RO feed water quality. |
| `AIT502` | sensor (quality) | RO permeate water quality (low conductivity = pure). |
| `AIT503` | sensor (quality) | RO reject water quality. |
| `AIT504` | sensor (quality) | RO product water quality. |
| `P501` | actuator | RO high-pressure pump 1 (running/stopped). |
| `P502` | actuator | RO high-pressure pump 2 (running/stopped). |

### 6. Stage 6 — Backwash and Disposal

Stage 6 manages periodic backwash of the UF and RO membranes. A flow transmitter (`FIT601`) measures backwash flow; three pumps (`P601`–`P603`) execute the backwash and drain sequences when scheduled or when triggered by membrane fouling indicators in earlier stages.

| Label | Type | Description |
|---|---|---|
| `FIT601` | sensor (flow) | Backwash flow rate (m³/h). |
| `P601` | actuator | Backwash pump 1 (running/stopped). |
| `P602` | actuator | Backwash pump 2 (running/stopped). |
| `P603` | actuator | Drain pump (running/stopped). |
