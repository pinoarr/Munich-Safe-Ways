The following will contain a technical documentation. If you prefer to hear the personal coming about of this challenge (motivation, learnings and challenges) be sure to visit: 
https://devpost.com/software/munich-safe-route


# Munich Pathfinder (HackaTUM – Landeshauptstadt Challenge)

Built in 24 hours to make moving through Munich feel safer, calmer and more sustainable. The app serves tailored walking **and** biking routes with three personalities (Direct, Night/Ride Safe, Scenic), fast UI feedback, and explainable trade-offs. 

## Why it exists
- Challenge: improve everyday life for Munich citizens (Landeshauptstadt München track).
- Reality: the fastest path is not always the best—lighting, greenery, crash risk and traffic matter.
- Goal: offer transparent routing that shows why a path is chosen and how it compares to the direct baseline.

## Quick start
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
# In another shell, serve the frontend (or open frontend/index.html directly)
python -m http.server 8001 -d frontend
```

Browse to `http://localhost:8000/` (FastAPI serves `frontend/index.html` too). Click start/end on the map. Use the Walk/Bike toggle and mode cards.

## Features at a glance
- Walk & Bike graphs (OSMnx) cached locally (`backend/data/`).
- Modes:
  - **Direct / Bike Direct**: shortest-ish with minimal fuss.
  - **Night Safe / Ride Safe**: lighting, busy anchors, crime history; for bikes also mock crash + traffic risk.
  - **Scenic / Bike Scenic**: parks, water, greener alignments.
- Comparison pills: tailored statements per mode; time deltas shown in minutes vs Direct.
- Impact stats: distance, ETA, CO₂ saved, steps/kcal (walk) or kcal/Wh (bike), “phones charged” equivalence.
- Auto enrichment: greenery, water, lighting, safety anchors, crime; bike adds crash/traffic mock layers.

## API
`GET /route`
- Query: `mode` (fast|safe|scenic), `transport` (walk|bike), `start_lat`, `start_lon`, `end_lat`, `end_lon`.
- Response: GeoJSON Feature with coordinates, stats, and comparison statements.

Example:
```bash
curl "http://localhost:8000/route?mode=fast&transport=bike&start_lat=48.137&start_lon=11.575&end_lat=48.15&end_lon=11.58"
```

Other endpoints:
- `GET /health` – readiness check.
- `GET /` – serves the frontend.

## How routing works (backend/main.py)
- Graphs: `load_or_download_graph` pulls walk and bike networks; caches to `backend/data/munich_*.graphml`.
- Enrichment:
  - Green/water context via OSM features → `dist_green`, `dist_water`, `compute_scenic_score`.
  - Safety context (lighting, busy anchors, crime heat) → `compute_safety_score`.
  - Bike-only layers: mock crash hotspots + traffic points → `bike_crash_risk`, `traffic_risk`.
- Weights:
```python
def make_weight_function(mode_name, mode_config):
    # Bike modes reject non-bike edges early
    if is_bike_mode and not is_bike_accessible(attrs):
        return length * mode_config.get("non_bike_penalty", 80.0)

    if base_mode == "fast":
        avoid = mode_config["length_bias"] + scenic * mode_config["scenic_avoid"]
        if is_bike_mode and is_bike_friendly(attrs):
            avoid = max(0.05, avoid - mode_config["bike_lane_reward"])
        avoid += traffic_risk * mode_config.get("traffic_penalty", 0.0)
        return length * max(mode_config.get("min_factor", 0.5), avoid)

    if base_mode == "safe":
        reward = safe_score * mode_config["safe_score_reward"] + light_aff * mode_config["lighting_reward"]
        penalty = (1 - safe_score) * mode_config["unsafe_penalty"] + crime_density * mode_config["crime_density_penalty"]
        penalty += crash_risk * mode_config.get("crash_penalty", 0.0)  # bike-specific
        if is_bike_mode and is_bike_friendly(attrs):
            reward += mode_config.get("bike_lane_reward", 0.0)
        return max(length * mode_config["min_factor"], length * (mode_config["length_bias"] + penalty) / (1 + reward))

    # scenic: reward scenic/green/water; bike rewards lanes, penalizes non-bike edges
```
- Comparisons:
  - Mode-specific statements (e.g., Bike Scenic: enjoyable gain vs Direct + time delta).
  - Safety deltas floored/boosted to avoid tiny or negative “safer” outputs.

Bike safety layer (mocked if data missing):
```python
def apply_bike_safety_layers(graph):
    crash_points = load_or_mock_points(BIKE_CRASH_DATA_PATH, projected, target_crs, mock_count=220)
    traffic_points = load_or_mock_points(TRAFFIC_DATA_PATH, projected, target_crs, mock_count=180)
    crash_density, _ = density_from_points(center, crash_points, BIKE_CRASH_INFLUENCE_RADIUS, BIKE_CRASH_SIGMA)
    traffic_density, _ = density_from_points(center, traffic_points, BIKE_CRASH_INFLUENCE_RADIUS, TRAFFIC_SIGMA)
    data["bike_crash_risk"] = min(1.0, crash_density)
    data["traffic_risk"] = min(1.0, traffic_density)
    data["bike_risk_score"] = min(1.0, 0.7 * crash_density + 0.6 * traffic_density)
```

## Frontend highlights (frontend/index.html)
- Leaflet map with click-to-start/end, auto-fit route.
- Walk/Bike toggle with animated slider and color cues (walk: amber, bike: blue); panel shadows follow transport.
- Stats grid (2×2) and comparison pills with minute deltas for time.
- Bike metric swap: “Steps” → “Calories”, kcal note → Wh effort; CO₂ stays.

## Data & placeholders
- Crime: `backend/static_data/munich_crime_points.json` (included). 
- Bike crashes & traffic: looks for `munich_bike_crashes.json` and `munich_traffic_hotspots.json`; if missing, generates mock hotspots within graph bounds so the Ride Safe weighting still works.
- Tile data: OpenStreetMap via Leaflet; OSMnx for graph building.

## Extending
- Drop real bike crash data into `backend/static_data/munich_bike_crashes.json` (lng/lat fields) and real traffic feeds into `munich_traffic_hotspots.json`.
- Tune mode knobs in `MODE_CONFIG` (green decay, crash penalties, bike lane rewards).
- Add more transports by duplicating mode triplets and pointing them to new graphs.

## Known limits
- Mock crash/traffic data until real feeds are provided.
- No persistence of UI selections across reloads.
- NO MOBILE VERSION YET!!!

## Testing
- Backend syntax check: `python3 -m py_compile backend/main.py`
- Quick API smoke (requires backend running): use the curl example above and verify a GeoJSON Feature returns.

## Inspiration
Built for HackaTUM to make Munich’s everyday routes feel safer, greener and clearer—so judges and citizens can see not just where to go, but why that path is the right one tonight.
