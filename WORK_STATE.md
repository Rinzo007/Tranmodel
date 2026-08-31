# Work State Summary

## Status Overview
- Tier 1: COMPLETED
- Tier 2: COMPLETED
- Tier 3: COMPLETED
- Blocked: None
- Next move: Await user direction

---

## Tier 1 — COMPLETED
- (Prior functionality baseline established; no outstanding work.)

## Tier 2 — COMPLETED (opt-in geometry-precision profile `dedup_profile`)
- Added opt-in geometry-precision profile `dedup_profile` with choices `"exact"` (default) | `"fast"`.
- Threaded through:
  - `config.py` — new `dedup_profile` config field.
  - `cli_args.py` — `--dedup-profile` argument with choices `exact`/`fast`.
  - `dedup_analyze` — new param `profile: str = "exact"`.
  - `pipeline_dedup_stage.py` — passes profile through the pipeline.
- Fast profile behavior:
  - `_fast_line` helper: simplify with `tol=buffer_r/20` + `set_precision` grid=`buffer_r/100`, with fallback to original geometry if collapsed.
  - Buffer uses `quad_segs=2`, `join_style=2`.
  - `union_all` extended in `geo_utils.py` to accept `grid_size`; fast mode passes `eps=max(0.1, min(2.0, buffer_r/50))` to heavy unions (`route_lines`, `unique_net`, `union_buf`).
- Bug found and fixed: flat/mitre buffer must NOT use `cap_style=2` (flat). Transit corridors share endpoints and flat caps zero the coverage there; round caps are required. Round caps were kept as a result.
- Decision: A separate "unique_net" toggle was NOT added. Semantics were ambiguous from notes, and the unique computation already benefits from `grid_size` acceleration on `union_all`. The global `unique_net` union already feeds `unique_km`/`km_coef`.
- Caching: Cache key includes `profile` so exact vs fast are separate cache entries.
- Tests: 266 total (3 new in `test_dedup_optimizations.py`):
  - fast changes coverage moderately,
  - default exact unchanged,
  - fast cache key differs.
  - All pass; ruff clean.

## Tier 3 — COMPLETED
- (Subsequent functionality established; no outstanding work.)

---

## Key Technical Notes
- IMPORTANT: Flat/mitre buffer must NOT use `cap_style=2` (flat) because transit corridors share endpoints and flat caps zero coverage there; round caps required.
- `union_all` in `geo_utils.py` now supports `grid_size` + fast `eps` for accelerated heavy unions.

## Next Steps
- Await user direction. Nothing is blocked.
