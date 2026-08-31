# Work State Summary

## Status Overview
- Legacy tiers: COMPLETED
- AequilibraE migration: COMPLETED (initial network-demand backend)
- Blocked: None
- Next move: migrate public-transport assignment to AequilibraE transit/GTFS when route schedules or frequencies are available

## AequilibraE migration
- Added `src/aequilibrae_pipeline.py`.
- Added `aequilibrae==1.7.0` to `requirements.txt`.
- Added Streamlit section `AequilibraE` with execution and rebuild controls.
- Added `README_AEQUILIBRAE.md`.

### New computational pipeline
1. Cached OSM road layer → GMNS nodes/links.
2. Existing real/reference transit stops → AequilibraE centroid zones.
3. AequilibraE graph + centroid connectors → network skims.
4. AequilibraE `SyntheticGravityModel` + `GravityApplication` with EXPO deterrence and internal IPF/Furness.
5. AequilibraE BPR/BFW traffic assignment → link loads and convergence report.

### Outputs
- `data/cache/aequilibrae/project`
- `data/cache/aequilibrae/gmns/nodes.csv`
- `data/cache/aequilibrae/gmns/links.csv`
- `data/cache/aequilibrae/link_load.parquet`
- `data/cache/aequilibrae/convergence.parquet`
- `data/cache/aequilibrae/aequilibrae_report.json`
- `data/report/aequilibrae_report.md`

### Important limitation
The current AequilibraE route-system documentation states that public-transport routes are imported from GTFS and that manual/programmatic route creation in the route system is not currently supported. Therefore the first migration stage uses AequilibraE as the network/skimming/distribution/assignment engine while preserving the existing transit-route layer for comparison.
