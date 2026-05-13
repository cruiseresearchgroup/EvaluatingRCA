# HVAC Rooftop Unit (RTU) — Operational System Documentation

**Asset:** Packaged Rooftop Unit (RTU) — self-contained commercial HVAC system mounted on the building roof.
**Document scope:** Process architecture, refrigeration cycle, instrumentation, and historian channel inventory of the RTU and associated terminal distribution.

---

## System Overview

A Rooftop Unit conditions outdoor and return air (heating and/or cooling) and delivers it indoors via ductwork. The unit operates autonomously under building-management-system (BMS) supervision; sensor readings, actuator commands, and operating-state indicators are recorded continuously by the historian.

**Airflow path:** outdoor air enters through the OA damper while return air re-enters through the RA damper; the two streams blend in the mixed-air plenum, are drawn across the indoor coil (evaporator) by the supply fan, and the conditioned supply air is delivered to occupied zones. Downstream Variable Air Volume (VAV) terminal boxes modulate airflow per zone and provide electric reheat.

**Refrigeration cycle:** the compressor pressurises refrigerant vapour from the evaporator and pumps it to the outdoor coil (condenser), where heat is rejected to outdoor air via the outdoor fan. The high-pressure liquid throttles through a thermal expansion valve and re-enters the evaporator to absorb heat from the supply-air stream. Compressors may be staged (Stage 1 ≈ 67 % capacity, Stage 2 = 100 % capacity) to modulate cooling output.

Different unit configurations are documented:
- A laboratory unit (Trane YCD150, 12.5 ton, EER 9.6) serving 10 VAV zones with electric resistance reheat in a two-storey office emulator.
- Field-deployed commercial RTUs (a 7.5-ton restaurant unit and a 10-ton distribution-center unit).
- A single-zone configuration (5,500 sq-ft office, ASHRAE Climate Zone 3C) with a two-stage scroll compressor on R410A refrigerant.

---

## Subsystems & Signal Reference

### 1. Air Handling and Distribution

The OA and RA dampers control the fraction of outdoor vs return air mixed into the supply stream; their positions are synchronised so that OA + RA = 1. The mixed air is conditioned by the evaporator and pushed through the duct system by the supply fan.

| Label | Type | Description |
|---|---|---|
| `RTU_OA_DMPR_DM` | actuator | OA damper control signal (0 = fully closed, 1 = fully open). |
| `RTU_RA_DMPR_DM` | actuator | RA damper control signal; complement of OA damper. |
| `RTU_OA_TEMP` | sensor | Outdoor air temperature (°F or °C). Weather, not actuated. |
| `RTU_MA_TEMP` | sensor | Mixed air temperature; blend of OA and RA before the evaporator. |
| `RTU_SA_TEMP` | sensor | Supply air temperature; conditioned air leaving the RTU. |
| `RTU_RA_TEMP` | sensor | Return air temperature; air returning from the zones. |
| `RTU_SA_FLOW` | sensor | Supply air volumetric flow rate (ACFM or cfm). |
| `RTU_RA_FLOW` | sensor | Return air volumetric flow rate (cfm). |
| `RTU_OA_FLOW` | sensor | Outdoor air volumetric flow rate (cfm). |
| `RTU_SA_HUM` / `RTU_RA_HUM` / `RTU_OA_HUM` / `RTU_MA_HUM` | sensor | Relative humidity at each duct location. |
| `RTU_SA_FAN_WATT` | sensor | Supply fan electricity consumption (W). |

### 2. Refrigeration Cycle

Refrigerant flows compressor → condenser (outdoor coil) → expansion valve → evaporator (indoor coil) and back to the compressor. Compressor staging modulates cooling capacity.

| Label | Type | Description |
|---|---|---|
| `RTU_COMP_WATT_1` | sensor | Compressor 1 electricity consumption (W). |
| `RTU_COMP_WATT_2` | sensor | Compressor 2 electricity consumption (W). |
| `RTU_COMP_WATT` | sensor | Single-compressor power (Stage 1) where one compressor is installed. |
| `RTU_STG_STA` | state | Compressor stage status (0 = off, 0.67 = Stage 1, 1 = Stage 2). |
| `RTU_REFG_DISC_PRES` / `RTU_REFG_DISC_PRES_1/2` | sensor | Refrigerant discharge pressure. |
| `RTU_REFG_SUCT_PRES` / `RTU_REFG_SUCT_PRES_1/2` | sensor | Refrigerant suction pressure. |
| `RTU_REFG_COND_PRES` | sensor | Refrigerant condenser-outlet pressure. |
| `RTU_REFG_DISC_TEMP` / `RTU_REFG_DISC_TEMP_1/2` | sensor | Refrigerant discharge line temperature. |
| `RTU_REFG_SUCT_TEMP` / `RTU_REFG_SUCT_TEMP_1/2` | sensor | Refrigerant suction line temperature. |
| `RTU_REFG_COND_TEMP` / `RTU_REFG_COND_TEMP_1/2` | sensor | Refrigerant condenser-outlet temperature. |
| `RTU_LA_COND_TEMP` | sensor | Air temperature leaving the condenser. |
| `RTU_SEN_CAPA` | sensor | Sensible cooling capacity (W). |
| `RTU_TOT_CAPA` | sensor | Total cooling capacity (W). |

### 3. Energy and Operating State

Aggregate electrical/gas consumption and occupancy mode for the RTU and broader HVAC system.

| Label | Type | Description |
|---|---|---|
| `RTU_TOT_WATT` | sensor | RTU total electricity consumption (W). |
| `HVAC_TOT_WATT` | sensor | Total HVAC electricity including VAV reheat (W). |
| `RTU_GAS_CSUM` | sensor | Natural gas consumption (SCFM). |
| `OCCU_MOD` | state | Occupancy mode indicator (1 = occupied, 0 = unoccupied). |

### 4. Zones and Terminal Distribution

Each zone is served by a VAV box that throttles airflow and provides electric reheat. Zone temperature/humidity sensors measure the actual delivered conditions.

| Label | Type | Description |
|---|---|---|
| `TERM_RM_TEMP_HSPT` | setpoint | Heating temperature setpoint (shared across rooms). |
| `TERM_RM_TEMP_CSPT` | setpoint | Cooling temperature setpoint (shared across rooms). |
| `TERM_RM_TEMP_102` … `TERM_RM_TEMP_206` | sensor | Per-room ambient temperature. |
| `TERM_RM_HUMD_102` … `TERM_RM_HUMD_206` | sensor | Per-room ambient relative humidity. |
| `VAV_RM_SAT_102` … `VAV_RM_SAT_206` | sensor | Per-VAV-box supply air temperature. |
| `VAV_RM_WATT_102` … `VAV_RM_WATT_206` | sensor | Per-VAV-box reheat power consumption. |
| `ZA_TEMP` | sensor | Zone air temperature (single-zone configuration). |
| `ZA_HUM` | sensor | Zone air relative humidity (single-zone configuration). |
| `ZA_TEMP_SPT` | setpoint | Zone air temperature setpoint (single-zone configuration). |
