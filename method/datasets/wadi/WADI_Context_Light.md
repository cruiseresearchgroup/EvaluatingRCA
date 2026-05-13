# WADI Domain Knowledge Context (Light)
**Source:** iTrust Labs, SUTD — Water Distribution (WaDi) Testbed
**Dataset:** WADI.A1 — 14 days normal + 2 days attack (Oct 9–11, 2017)
**Resolution:** 1 second | **Sensors after preprocessing:** 79 (7-type filter + 2A/2B merge)

---

## System Overview

WaDi is a scaled-down, industry-compliant emulation of a modern water distribution facility. Water flows in a **closed loop** through three stages:

```
[Raw Water Sources] → Stage 1 (Intake & Treatment)
                    → Stage 2 (Distribution to consumers)
                    → Stage 3 (Return & Recirculation)
                    → back to Stage 1
```

---

## Stage Descriptions

### Stage 1 — Water Intake & Chemical Dosing
- Two raw water sources: PUB-supplied mains water + RO permeate from SWaT testbed
- Water flows into **primary raw water storage tanks** (1_LT_001)
- **Chemical dosing pumps** (1_P_001–1_P_006) add treatment chemicals
- **Motorized valves** (1_MV_001–1_MV_004) control inlet/outlet flow
- **Flow transmitter** 1_FIT_001 measures incoming flow rate
- **Analyzer sensors** (1_AIT_001–1_AIT_005) measure water quality
- **Level switches** (1_LS_001, 1_LS_002) provide high/low level alarms

### Stage 2 — Distribution Network
- Water pumped from Stage 1 into **Elevated Reservoir (ER) tanks** (2_LT_001, 2_LT_002)
- Water flows from ER by gravity or **booster pumps** (2_P_003, 2_P_004) to six **consumer tanks**
- Consumer tank levels monitored by **level switches** (2_LS_101–2_LS_601)
- **Manual control valves** (2_MCV_007, 2_MCV_101–2_MCV_601) control flow to each consumer
- **Motorized valves** (2_MV_001–2_MV_009, 2_MV_101–2_MV_601) gate the consumer supply lines
- **Flow transmitters** (2_FIT_001–2_FIT_003) measure flows within distribution network
- **Analyzer sensors** (2_AIT_001–2_AIT_004) monitor quality in distribution

### Stage 3 — Return & Recirculation
- Consumer tanks drain to a **return tank** (3_LT_001)
- **Return pumps** (3_P_001–3_P_004) recirculate water back to Stage 1 raw tanks
- **Motorized valves** (3_MV_001–3_MV_003) control return flow routing
- **Flow transmitter** 3_FIT_001 measures return flow
- **Analyzer sensors** (3_AIT_001–3_AIT_005) monitor return water quality

---

## Sensor Type Legend

| Code | Full Name | Measurement | Typical range |
|------|-----------|-------------|---------------|
| MV   | Motorized Valve | STATUS: 0=closed, 1=open | Binary |
| LS   | Level Switch | AL=low alert, AH=high alert (binary) | Binary |
| LT   | Level Transmitter | PV = process value (% or mm) | 0–100% |
| FIT  | Flow Indicator Transmitter | PV = flow rate (L/min or m³/h) | 0–∞ |
| AIT  | Analyzer Indicator Transmitter | PV = water quality parameter | varies |
| MCV  | Manual Control Valve | CO = control output (% open) | 0–100% |
| P    | Pump | STATUS: 0=off, 1=running; SPEED in RPM | Binary / RPM |

---

## Causal Dependencies (Physical Flow Logic)

Understanding these dependencies is critical for RCA — downstream sensors react to upstream manipulations:

```
1_MV_001 (open) ──→ 1_LT_001 rises ──→ 1_LS_002_AL triggers ──→ overflow
1_FIT_001 (false) ──→ PLC sees "no flow" ──→ 1_P_001–P_004 activate ──→ over-dosing
1_P_005/P_006 (on) ──→ pressure spike ──→ pipe burst risk
2_LT_002 (false) ──→ PLC drains/fills ER ──→ 2_FIT_001/002 change ──→ consumer levels affected
2_MCV_xxx = 0% ──→ consumer tank LS_xxx_AL triggers ──→ no supply
2_MCV_007 (open) ──→ water bypasses consumers ──→ 2_FIT_002 drops ──→ booster never triggers
```

---

## PLC Control Logic (Automated Responses)

| Condition (PLC input) | Automated PLC response | Sensors that change as effects |
|----------------------|------------------------|-------------------------------|
| 1_LT_001 < low setpoint | Open 1_MV_001 to refill | 1_FIT_001 rises, 1_LT_001 rises |
| 1_FIT_001 reads 0 (no flow) | Activate 1_P_001–P_004 (chemical dosing) | Pump STATUS sensors flip ON |
| 1_FIT_001 reads high | Reduce/stop 1_P_001–P_004 dosing | Pump STATUS sensors flip OFF |
| 2_LT_001/002 < low setpoint | Activate booster pumps 2_P_003/P_004 | 2_FIT_003 rises, pump SPEED rises |
| 2_FIT_002 > expected outflow | Booster logic suppressed | 2_P_003/P_004 stay off |
| Consumer LS_xxx_AL triggers | Open corresponding 2_MCV_xxx or 2_MV_xxx | Flow resumes to consumer tank |
| 3_LT_001 < low setpoint | Activate return pumps 3_P_001–P_004 | 3_FIT_001 changes |
