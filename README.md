# Munich Pathfinder — Technical Deep Dive

DevPost: https://devpost.com/software/munich-safe-route  
Premise: Routing only wins adoption when perceived safety is satisfied. Pathfinder bakes safety, lighting, and comfort into the cost model and keeps it explainable.

## Local Start Procedure (required sequence)
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `pip install -r backend/requirements.txt`
5. `cd backend/`
6. `uvicorn main:app --reload --port 8000`

## Code Map
- FastAPI entrypoint: `backend/main.py` (endpoints, CORS, frontend serve).
- Routing engine: `backend/pathfinder/`
  - `config.py`: constants, mode knobs, thresholds, required edge keys, labels.
  - `graph_manager.py`: graph I/O, enrichment orchestration, node caching.
  - `environment.py`: feature extraction (green/water/safe/lighting/crime/bike-risk).
  - `routing.py`: cost model, path search, comparison statements.
  - `stats.py`: segment aggregation and normalization.
  - `utils.py`: coercion/helpers.
- Frontend: `frontend/index.html` (Leaflet SPA, inlined logic/styles).

## End-to-End Request Flow (GET /route)
1. **Ingress** (`backend/main.py`): validate `mode`/`transport`/coords, map to `mode_key`.
2. **Graph selection**: `get_graph_for_mode` returns preloaded walk or bike graph (globals `G_walk`, `G_bike`), loaded at startup by `init_graphs`.
3. **Nearest nodes**: OSMnx `nearest_nodes`; fallback Haversine KNN (`graph_manager.fallback_nearest_node`) if OSMnx lookup fails.
4. **Route computation**: `routing.compute_route_variant`
   - Build mode-specific weight function (`make_weight_function`).
   - Wrap into MultiEdge scorer (`make_edge_weight` → `select_best_variant`).
   - `nx.shortest_path` with dynamic weights.
5. **Geometry + stats**: `build_geometry_and_stats` walks the path, flattens geometry, accumulates segment stats, normalizes via `stats.finalize_segment_stats`.
6. **Peer comparisons**: optionally compute fast/safe/scenic peers; `compute_comparison_ratios` + `build_mode_comparison_statements` generate deltas and safety boosts.
7. **Response**: GeoJSON Feature (LineString) with `properties` (mode metadata, duration/length, stats, baseline/custom comparisons).

## Graph Lifecycle & Caching
- Load/build: `graph_manager.load_or_download_graph` pulls Munich walk/bike via OSMnx; caches to `backend/data/munich_[walk|bike].graphml`.
- Prepare: `graph_manager.prepare_graph`
  - Adds edge lengths.
  - If cached features exist: recompute `scenic_score`/`safe_score` in-place for consistency.
  - Else: `environment.enrich_graph_with_environment`, then save updated cache.
  - Bike graphs: apply `apply_bike_safety_layers` (crash/traffic densities) and persist.
- Node cache: `build_node_cache` stores radian coords for manual nearest lookup when OSMnx is unavailable.
- Fallback: on enrichment errors, fill `REQUIRED_EDGE_KEYS` with defaults to keep routing alive (less meaningful scoring).

## Feature Extraction (`environment.py`)
- Distance fields (per edge): `dist_green`, `dist_water`, `dist_safe_poi`, `dist_busy_area`, `dist_lighting`, `dist_crime_hotspot`.
- Scores:
  - Scenic: `compute_scenic_score` (low-speed tags, calm streets, surface heuristics, exp-decay to green/water).
  - Safety: `compute_safety_score` (lighting, busy anchors, safe anchors, crime density/proximity).
- Crime layer: STRtree over `static_data/munich_crime_points.json`; crime-type weighting (`crime_weight`); Gaussian smoothing (`CRIME_SIGMA`, `CRIME_INFLUENCE_RADIUS`), normalized by `CRIME_DENSITY_NORMALIZER`.
- Bike safety: `apply_bike_safety_layers` builds crash/traffic density (`bike_crash_risk`, `traffic_risk`, `bike_risk_score`); uses real JSONs if present, else mocks points inside graph bounds.
- Required keys: `config.REQUIRED_EDGE_KEYS` must exist; missing values are defaulted (`DEFAULT_DIST`, zeros).

## Cost Model (per mode) — `routing.py:make_weight_function`
- Bike gate: Non-bike-accessible edges → `length * non_bike_penalty`.
- Fast / Bike Direct:
  - `avoid = length_bias + scenic_avoid*scenic + safe_penalty*safe`.
  - Bike-friendly reduces avoid by `bike_lane_reward`; traffic adds `traffic_penalty*traffic_risk`.
  - Weight: `length * max(min_factor, avoid)`.
- Safe / Ride Safe:
  - Affinities: exp(-dist/decay) for safe anchors, busy areas, lighting, crime-gap.
  - Reward: safe_score, safe anchors, busy, lighting, bike lanes (bike).
  - Penalty: crime density, crime-near, bad area, crash/traffic risk; unsafe_floor triggers `unsafe_multiplier`.
  - Weight: `length * (length_bias + penalty) / (1 + reward)`, clamped by `min_factor`.
- Scenic / Bike Scenic:
  - Rewards: scenic_score + exp(-dist_green/green_decay) + exp(-dist_water/water_decay).
  - Penalties: detour_penalty*(1-scenic); bike non-accessible edges get `non_bike_penalty`; bike lanes reduce weight.
  - Weight clamped by `min_factor`.
- MultiEdges: `make_edge_weight` scores each variant; `select_best_variant` picks min-weight variant for both path cost and stats consistency.

### Core Routing Routine (excerpt)
```python
def compute_route_variant(graph, start_node, end_node, mode_name):
    mode_config = MODE_CONFIG[mode_name]
    weight_for_attrs = make_weight_function(mode_name, mode_config)
    edge_weight = make_edge_weight(weight_for_attrs)
    path = nx.shortest_path(graph, start_node, end_node, weight=edge_weight)
    coords, total_length, stats = build_geometry_and_stats(graph, path, weight_for_attrs)
    speed = MODE_SPEED_MPS.get(mode_name, DEFAULT_SPEED_MPS)
    duration = total_length / speed if speed > 0 else 0.0
    stats["duration_s"] = duration
    return {
        "mode": mode_name,
        "path": path,
        "coordinates": coords,
        "length_m": total_length,
        "duration_s": duration,
        "stats": stats,
    }
```

### Weighting Internals (safe mode excerpt)
```python
safe_score = clamp(attrs.get("safe_score", 0.3))
dist_safe   = as_float(attrs.get("dist_safe_poi", DEFAULT_DIST), DEFAULT_DIST)
dist_busy   = as_float(attrs.get("dist_busy_area", DEFAULT_DIST), DEFAULT_DIST)
dist_light  = as_float(attrs.get("dist_lighting", DEFAULT_DIST), DEFAULT_DIST)
dist_crime  = as_float(attrs.get("dist_crime_hotspot", DEFAULT_DIST), DEFAULT_DIST)
crime_density = clamp(attrs.get("crime_density", 0.0))
crash_risk    = clamp(attrs.get("bike_crash_risk", 0.0))
traffic_risk  = clamp(attrs.get("traffic_risk", 0.0))

safe_aff  = affinity(dist_safe, mode_config["safe_decay"])
busy_aff  = affinity(dist_busy, mode_config["busy_decay"])
light_aff = affinity(dist_light, mode_config["lighting_decay"])
crime_aff = affinity(dist_crime, mode_config.get("crime_gap_decay", CRIME_NEAR_DECAY))

reward = (
    mode_config["safe_score_reward"] * safe_score
    + mode_config["safe_anchor_reward"] * safe_aff
    + mode_config["busy_reward"] * busy_aff
    + mode_config["lighting_reward"] * light_aff
    + mode_config["crime_gap_reward"] * max(0.0, 1 - crime_aff)
)
penalty = (
    mode_config["unsafe_penalty"] * (1 - safe_score)
    + mode_config["crime_density_penalty"] * crime_density
    + mode_config["crime_near_penalty"] * crime_aff
    + mode_config.get("crash_penalty", 0.0) * crash_risk
    + mode_config.get("traffic_penalty", 0.0) * traffic_risk
)
weight = length * (mode_config["length_bias"] + penalty)
if reward > 0:
    weight /= 1 + reward
if safe_score < mode_config["unsafe_floor"]:
    weight *= mode_config["unsafe_multiplier"]
return max(length * mode_config["min_factor"], weight)
```

### Scenic Scoring (excerpt)
```python
score = 0.2
if highway_tags & LOW_SPEED_TAGS: score += 0.35
elif highway_tags & CALM_STREET_TAGS: score += 0.2
if highway_tags & CAR_ROAD_TAGS: score -= 0.4
if surface in SMOOTH_SURFACES: score += 0.1
elif surface in ROUGH_SURFACES: score -= 0.15
score += 0.6 * exp(-dist_green / SIGMA_GREEN) if dist_green < DEFAULT_DIST else 0
score += 0.7 * exp(-dist_water / SIGMA_WATER) if dist_water < DEFAULT_DIST else 0
score = clamp(score, 0.0, 1.0)
```

## Stats & Comparison Layer
- Aggregation (`stats.py`):
  - Totals: length, segments, scenic/safe sums, crime/bad-area sums.
  - Shares: green, water, bike-friendly, lit/dark, crime-hot, safe-anchor (thresholds in `config.py`).
  - Outputs: `avg_scenic`, `avg_safe_score`, `avg_crime_density`, `avg_bad_area`, shares, `length_m`, `duration_s`.
- Comparison:
  - `compute_comparison_ratios`: percent deltas vs baseline for `COMPARISON_KEYS`.
  - `build_mode_comparison_statements`: user-facing pills; small positive safety deltas boosted (`boost_small_safety_gain`, `adjust_safety_delta`) to avoid “0% safer” messaging.
  - `comparison_custom`: if provided, UI prefers these over baseline deltas.

## API
- `GET /route`
  - Query: `mode` (fast|safe|scenic), `transport` (walk|bike), `start_lat`, `start_lon`, `end_lat`, `end_lon`.
  - Returns: GeoJSON Feature with `geometry.coordinates` and `properties` (mode, transport, labels, length/duration, stats, baseline/custom comparisons).
- `GET /health` — readiness.
- `GET /` — serves `frontend/index.html` if present.

Example:
```bash
curl "http://localhost:8000/route?mode=fast&transport=bike&start_lat=48.137&start_lon=11.575&end_lat=48.15&end_lon=11.58"
```

## Frontend (Leaflet SPA)
- `frontend/index.html` only; click map sets start/end; third click resets end.
- Transport toggle (walk/bike) swaps labels and units (Steps/kcal vs Calories/Wh); mode cards set `mode`.
- Renders returned LineString with mode color; auto-zooms to bounds; pills show custom statements or deltas vs Direct.
- Derived metrics: CO₂ saved (vs car), steps/kcal or Wh/kcal depending on transport.

## Operations, Debugging, Failure Modes
- Syntax: `python3 -m py_compile backend/main.py`
- Health: `curl http://localhost:8000/health`
- Routing smoke: example curl; verify GeoJSON + `properties.comparison_*`.
- Logs: stdout prints cache hits/misses, enrichment fallback (mock points), warnings on missing datasets.
- Caches: delete `backend/data/*.graphml` to rebuild after changing `SCENIC_PLACE` or knobs.
- Missing data: crime/bike datasets absent → mocks; REQUIRED_EDGE_KEYS backfilled to keep routing functional.

## Extension Points
- New city: edit `SCENIC_PLACE` in `config.py`, clear `backend/data/`.
- New modes: add to `MODE_CONFIG` + `MODE_DISPLAY_NAME`; `make_weight_function` is generic by mode name.
- New layers: compute in `environment.py`, add to `REQUIRED_EDGE_KEYS`, use in weighting/stats.
- Frontend: point to remote backend, externalize CSS/JS, adjust colors/labels as needed.

## DevPost Context (Why)
Users avoid crash hotspots, dark segments, and crime-linked areas even if it conflicts with sustainability goals. Pathfinder fuses accident/lighting/activity/crime proxies with green/water comfort, offers three intentful modes (Direct/Night Safe/Scenic + bike), and surfaces tangible metrics (CO₂ saved, kcal/Wh, lighting/crime/green shares) to make “slightly slower but safer/greener” the rational default.
