from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
import math
import osmnx as ox
import networkx as nx
import numpy as np
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
G_walk = None  # global walking graph
CACHE_PATH = BASE_DIR / "data" / "munich_walk.graphml"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
STATIC_DATA_DIR = BASE_DIR / "static_data"
STATIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
NODE_CACHE = {}
LOW_SPEED_TAGS = {"footway", "path", "cycleway", "pedestrian", "living_street", "track"}
CALM_STREET_TAGS = {"residential", "service", "unclassified"}
CAR_ROAD_TAGS = {"primary", "primary_link", "secondary", "secondary_link", "trunk", "trunk_link"}
GREEN_LEISURE = {"park", "garden", "playground", "nature_reserve"}
GREEN_LANDUSE = {"forest", "grass", "meadow", "recreation_ground"}
SMOOTH_SURFACES = {"asphalt", "paved", "paving_stones", "concrete"}
ROUGH_SURFACES = {"gravel", "dirt", "ground", "unpaved"}
DEFAULT_DIST = 9999.0
SIGMA_GREEN = 220.0
SIGMA_WATER = 320.0
SIGMA_SAFE_POI = 140.0
SIGMA_BUSY = 200.0
SIGMA_LIGHT = 160.0
SCENIC_PLACE = "Munich, Germany"
# Historic incident list adapted from https://github.com/fuchsvomwalde/munichwatcher (crimeDataJSON.js)
CRIME_DATA_PATH = STATIC_DATA_DIR / "munich_crime_points.json"
CRIME_INFLUENCE_RADIUS = 220.0
CRIME_SIGMA = 95.0
CRIME_DENSITY_NORMALIZER = 3.4
CRIME_NEAR_DECAY = 140.0
SAFE_POI_TAGS = {
    "amenity": ["police", "fire_station", "ranger_station", "embassy", "hospital", "clinic"],
    "emergency": ["ambulance_station"],
    "building": ["police"],
}
BUSY_PUBLIC_TAGS = {
    "amenity": [
        "bus_station",
        "subway_entrance",
        "tram_stop",
        "ferry_terminal",
        "marketplace",
        "community_centre",
        "townhall",
        "library",
        "cinema",
        "theatre",
        "arts_centre",
        "university",
        "college",
        "school",
        "restaurant",
        "cafe",
    ],
    "public_transport": ["stop_position", "station", "platform"],
    "railway": ["station", "halt", "tram_stop", "light_rail"],
}
LIGHTING_TAGS = {
    "highway": "street_lamp",
    "man_made": "street_lamp",
}
VIOLENT_CRIME_TYPES = {"raub", "k\xf6rperverletzung", "vergewaltigung", "mord", "totschlag"}
PROPERTY_CRIME_TYPES = {"diebstahl", "einbruch", "taschendiebstahl"}

MODE_CONFIG = {
    "fast": {
        "length_bias": 1.05,
        "min_factor": 1.0,
        "scenic_avoid": 0.65,
        "safe_penalty": 0.8,
        "bike_penalty": 0.2,
        "lit_penalty": 0.15,
        "dark_reward": 0.4,
    },
    "scenic": {  # go all in on rivers/parks
        "length_bias": 0.7,
        "scenic_reward": 1.6,
        "water_reward": 2.8,
        "green_reward": 1.2,
        "detour_penalty": 0.05,
        "min_factor": 0.06,
        "water_decay": 75.0,
        "green_decay": 170.0,
    },
    "safe": {
        "length_bias": 1.05,
        "min_factor": 0.35,
        "safe_score_reward": 2.8,
        "safe_anchor_reward": 1.9,
        "busy_reward": 1.3,
        "lighting_reward": 1.0,
        "crime_density_penalty": 2.2,
        "bad_area_penalty": 3.4,
        "crime_near_penalty": 1.9,
        "crime_gap_reward": 1.2,
        "unsafe_penalty": 1.4,
        "unsafe_floor": 0.35,
        "unsafe_multiplier": 3.2,
        "safe_decay": 110.0,
        "busy_decay": 180.0,
        "lighting_decay": 150.0,
        "crime_gap_decay": 200.0,
    },
}

# Backwards-compatible aliases for earlier beta modes
MODE_CONFIG["scenic_plus"] = MODE_CONFIG["scenic"]
MODE_CONFIG["scenic_river"] = MODE_CONFIG["scenic"]

MODE_DISPLAY_NAME = {
    "fast": "Direct",
    "safe": "Night Safe",
    "scenic": "Scenic",
}
MODE_SPEED_MPS = {
    "fast": 1.5,
    "safe": 1.35,
    "scenic": 1.3,
}
DEFAULT_SPEED_MPS = 1.35
GREEN_PROX_THRESHOLD = 80.0
WATER_PROX_THRESHOLD = 80.0
SAFE_PROX_THRESHOLD = 120.0
CRIME_PROX_THRESHOLD = 80.0
CRIME_DENSITY_THRESHOLD = 0.35
COMPARISON_KEYS = (
    "length_m",
    "duration_s",
    "avg_scenic",
    "avg_safe_score",
    "green_share",
    "water_share",
    "bike_share",
    "lit_share",
    "crime_hot_share",
    "safe_anchor_share",
)
LIT_POSITIVE = {"yes", "true", "1"}
LIT_NEGATIVE = {"no", "false", "0"}

REQUIRED_EDGE_KEYS = (
    "scenic_score",
    "dist_green",
    "dist_water",
    "safe_score",
    "dist_safe_poi",
    "dist_busy_area",
    "dist_lighting",
    "crime_density",
    "dist_crime_hotspot",
    "bad_area_score",
    "crime_count_close",
)


def as_int(value, default):
    try:
        return int(str(value).split()[0])
    except Exception:
        return default


def as_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def truthy(value):
    return str(value).lower() in {"1", "yes", "true"}


def to_tags(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tags = []
        for v in value:
            if v is None:
                continue
            tags.extend(str(v).split(";"))
        return [tag.strip() for tag in tags if tag.strip()]
    return [tag.strip() for tag in str(value).split(";") if tag.strip()]


def tree_geometry_from_candidate(tree_info, candidate):
    if candidate is None or tree_info is None:
        return None
    if isinstance(candidate, (int, np.integer)):
        idx = int(candidate)
        if 0 <= idx < len(tree_info["geoms"]):
            return tree_info["geoms"][idx]
        return None
    if isinstance(candidate, BaseGeometry):
        return candidate
    return None


def resolve_tree_geometry(tree_info, target_geom):
    if tree_info is None:
        return None
    candidate = tree_info["tree"].nearest(target_geom)
    return tree_geometry_from_candidate(tree_info, candidate)


def crime_weight(crime_type: str) -> float:
    label = str(crime_type or "").strip().lower()
    if not label:
        return 1.0
    if label in VIOLENT_CRIME_TYPES:
        return 2.4
    if label in PROPERTY_CRIME_TYPES:
        return 1.3
    if "droge" in label or "waffe" in label:
        return 1.6
    if "brand" in label:
        return 1.2
    return 1.0


def load_crime_points(target_crs):
    if not CRIME_DATA_PATH.exists():
        print(f"Warning: crime dataset missing at {CRIME_DATA_PATH}")
        return None
    try:
        entries = json.loads(CRIME_DATA_PATH.read_text())
    except Exception as exc:
        print(f"Warning: failed to parse crime dataset ({exc})")
        return None

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    geoms = []
    weights = []

    for entry in entries:
        try:
            lon = float(entry.get("lng"))
            lat = float(entry.get("lat"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(lon) or not math.isfinite(lat):
            continue
        x, y = transformer.transform(lon, lat)
        geoms.append(Point(x, y))
        weights.append(crime_weight(entry.get("type")))

    if not geoms:
        print("Warning: crime dataset had no usable points")
        return None

    tree = STRtree(geoms)
    geom_weights = {id(geom): weight for geom, weight in zip(geoms, weights)}
    return {"tree": tree, "geoms": geoms, "weights": geom_weights}


def compute_scenic_score(edge_attrs):
    """
    Returns a scenic score in [0, 1].
    Higher = calmer / more pleasant / less car-oriented.
    """

    highway_tags = {tag for tag in to_tags(edge_attrs.get("highway"))}
    maxspeed = as_int(edge_attrs.get("maxspeed", 50), 50)
    lanes = as_int(edge_attrs.get("lanes", 1), 1)
    surface = str(edge_attrs.get("surface", "")).lower()
    landuse = str(edge_attrs.get("landuse", "")).lower()
    leisure = str(edge_attrs.get("leisure", "")).lower()
    dist_green = as_float(edge_attrs.get("dist_green", DEFAULT_DIST), DEFAULT_DIST)
    dist_water = as_float(edge_attrs.get("dist_water", DEFAULT_DIST), DEFAULT_DIST)

    score = 0.2  # base bias toward neutral

    # Prefer small / human-scale streets and paths
    if highway_tags & LOW_SPEED_TAGS:
        score += 0.35
    elif highway_tags & CALM_STREET_TAGS:
        score += 0.2

    # Penalise big car roads
    if highway_tags & CAR_ROAD_TAGS:
        score -= 0.4

    # Speed heuristics
    if maxspeed <= 30:
        score += 0.2
    elif maxspeed <= 50:
        score += 0.1
    elif maxspeed >= 80:
        score -= 0.25
    elif maxspeed >= 60:
        score -= 0.15

    # Busy roads with many lanes feel less safe
    if lanes >= 4:
        score -= 0.25
    elif lanes >= 3:
        score -= 0.15

    # Reward green context (parks, gardens, forests)
    if landuse in GREEN_LANDUSE or leisure in GREEN_LEISURE:
        score += 0.25

    # Gentle reward for bridges (views) but penalise tunnels
    if truthy(edge_attrs.get("bridge")):
        score += 0.05
    if truthy(edge_attrs.get("tunnel")):
        score -= 0.2

    # Encourage well-lit, paved walkways
    if surface in SMOOTH_SURFACES:
        score += 0.05
    if surface in ROUGH_SURFACES:
        score -= 0.1
    if str(edge_attrs.get("lit", "yes")).lower() in {"no", "0", "false"}:
        score -= 0.05

    # Dedicated cycle infra counts as pleasant fallback
    if edge_attrs.get("cycleway") or edge_attrs.get("segregated"):
        score += 0.1

    # Smooth bonus for proximity to greenery/water (values in meters)
    if dist_green < DEFAULT_DIST:
        score += 0.25 * math.exp(-dist_green / SIGMA_GREEN)
    if dist_water < DEFAULT_DIST:
        score += 0.2 * math.exp(-dist_water / SIGMA_WATER)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def compute_safety_score(edge_attrs):
    """Estimate personal safety (lighting, proximity to crowd/support) for an edge."""

    highway_tags = {tag for tag in to_tags(edge_attrs.get("highway"))}
    surface = str(edge_attrs.get("surface", "")).lower()
    service = str(edge_attrs.get("service", "")).lower()
    sidewalk = str(edge_attrs.get("sidewalk", "")).lower()
    lit_value = str(edge_attrs.get("lit", "")).lower()
    width = as_float(edge_attrs.get("width", 0.0), 0.0)

    dist_safe = as_float(edge_attrs.get("dist_safe_poi", DEFAULT_DIST), DEFAULT_DIST)
    dist_busy = as_float(edge_attrs.get("dist_busy_area", DEFAULT_DIST), DEFAULT_DIST)
    dist_light = as_float(edge_attrs.get("dist_lighting", DEFAULT_DIST), DEFAULT_DIST)
    dist_crime = as_float(edge_attrs.get("dist_crime_hotspot", DEFAULT_DIST), DEFAULT_DIST)
    crime_density = max(0.0, min(1.0, float(edge_attrs.get("crime_density", 0.0))))
    bad_area = max(0.0, min(1.0, float(edge_attrs.get("bad_area_score", 0.0))))

    score = 0.35

    if highway_tags & {"pedestrian", "living_street"}:
        score += 0.25
    elif "residential" in highway_tags or "residential" == edge_attrs.get("highway"):
        score += 0.15
    elif highway_tags & {"primary", "secondary", "tertiary"}:
        score += 0.05

    if highway_tags & {"track", "path", "bridleway"}:
        score -= 0.15

    if service == "alley":
        score -= 0.45
    elif service in {"driveway", "siding"}:
        score -= 0.15

    if truthy(edge_attrs.get("tunnel")):
        score -= 0.3

    if lit_value in {"yes", "true", "1"}:
        score += 0.25
    elif lit_value in {"no", "false", "0"}:
        score -= 0.35

    if sidewalk and sidewalk not in {"no", "none", "0"}:
        score += 0.1

    if width >= 6:
        score += 0.05
    elif 0 < width <= 2.0:
        score -= 0.05

    if surface in SMOOTH_SURFACES:
        score += 0.05
    elif surface in ROUGH_SURFACES:
        score -= 0.05

    safe_aff = 0.0 if dist_safe >= DEFAULT_DIST else math.exp(-dist_safe / SIGMA_SAFE_POI)
    busy_aff = 0.0 if dist_busy >= DEFAULT_DIST else math.exp(-dist_busy / SIGMA_BUSY)
    light_aff = 0.0 if dist_light >= DEFAULT_DIST else math.exp(-dist_light / SIGMA_LIGHT)
    crime_aff = 0.0 if dist_crime >= DEFAULT_DIST else math.exp(-dist_crime / CRIME_NEAR_DECAY)

    score += 0.45 * safe_aff + 0.35 * busy_aff + 0.25 * light_aff
    if crime_density > 0:
        score -= 0.35 * crime_density
    if bad_area > 0:
        score -= 0.45 * bad_area
    if crime_aff > 0:
        score -= 0.25 * crime_aff

    return max(0.0, min(1.0, score))


def edge_variants(data):
    if isinstance(data, dict) and "length" in data:
        return [data]
    return list(data.values())


def is_bike_friendly(attrs):
    if attrs.get("cycleway") or attrs.get("segregated"):
        return True
    highway = str(attrs.get("highway", "")).lower()
    if highway == "cycleway":
        return True
    bicycle = str(attrs.get("bicycle", "")).lower()
    return bicycle in {"designated", "yes"}


def init_segment_stats():
    return {
        "length": 0.0,
        "segments": 0,
        "scenic_sum": 0.0,
        "safe_sum": 0.0,
        "crime_density_sum": 0.0,
        "bad_area_sum": 0.0,
        "green_length": 0.0,
        "water_length": 0.0,
        "bike_length": 0.0,
        "lit_length": 0.0,
        "dark_length": 0.0,
        "safe_anchor_length": 0.0,
        "crime_hot_length": 0.0,
    }


def accumulate_segment_stats(stats, attrs, segment_length):
    scenic = max(0.0, min(1.0, float(attrs.get("scenic_score", 0.0))))
    safe = max(0.0, min(1.0, float(attrs.get("safe_score", 0.0))))
    dist_green = as_float(attrs.get("dist_green", DEFAULT_DIST), DEFAULT_DIST)
    dist_water = as_float(attrs.get("dist_water", DEFAULT_DIST), DEFAULT_DIST)
    dist_safe = as_float(attrs.get("dist_safe_poi", DEFAULT_DIST), DEFAULT_DIST)
    dist_crime = as_float(attrs.get("dist_crime_hotspot", DEFAULT_DIST), DEFAULT_DIST)
    crime_density = max(0.0, float(attrs.get("crime_density", 0.0)))
    bad_area = max(0.0, float(attrs.get("bad_area_score", 0.0)))
    lit_value = str(attrs.get("lit", "")).lower()

    stats["length"] += segment_length
    stats["segments"] += 1
    stats["scenic_sum"] += scenic * segment_length
    stats["safe_sum"] += safe * segment_length
    stats["crime_density_sum"] += crime_density * segment_length
    stats["bad_area_sum"] += bad_area * segment_length

    if dist_green < GREEN_PROX_THRESHOLD:
        stats["green_length"] += segment_length
    if dist_water < WATER_PROX_THRESHOLD:
        stats["water_length"] += segment_length
    if dist_safe < SAFE_PROX_THRESHOLD:
        stats["safe_anchor_length"] += segment_length
    if dist_crime < CRIME_PROX_THRESHOLD or crime_density >= CRIME_DENSITY_THRESHOLD:
        stats["crime_hot_length"] += segment_length
    if is_bike_friendly(attrs):
        stats["bike_length"] += segment_length
    if lit_value in LIT_POSITIVE:
        stats["lit_length"] += segment_length
    elif lit_value in LIT_NEGATIVE:
        stats["dark_length"] += segment_length


def finalize_segment_stats(raw_stats):
    total = raw_stats["length"]
    if total <= 0:
        return {
            "length_m": 0.0,
            "segments": 0,
            "avg_scenic": 0.0,
            "avg_safe_score": 0.0,
            "avg_crime_density": 0.0,
            "avg_bad_area": 0.0,
            "green_share": 0.0,
            "water_share": 0.0,
            "bike_share": 0.0,
            "lit_share": 0.0,
            "dark_share": 0.0,
            "crime_hot_share": 0.0,
            "safe_anchor_share": 0.0,
        }

    def share(value):
        return value / total if total > 0 else 0.0

    return {
        "length_m": total,
        "segments": raw_stats["segments"],
        "avg_scenic": raw_stats["scenic_sum"] / total,
        "avg_safe_score": raw_stats["safe_sum"] / total,
        "avg_crime_density": raw_stats["crime_density_sum"] / total,
        "avg_bad_area": raw_stats["bad_area_sum"] / total,
        "green_share": share(raw_stats["green_length"]),
        "water_share": share(raw_stats["water_length"]),
        "bike_share": share(raw_stats["bike_length"]),
        "lit_share": share(raw_stats["lit_length"]),
        "dark_share": share(raw_stats["dark_length"]),
        "crime_hot_share": share(raw_stats["crime_hot_length"]),
        "safe_anchor_share": share(raw_stats["safe_anchor_length"]),
    }


def select_best_variant(edge_data, weight_for_attrs):
    variants = edge_variants(edge_data)
    if not variants:
        return None
    return min(variants, key=weight_for_attrs)


def make_weight_function(mode_name: str, mode_config: dict):
    def affinity(dist_value: float, decay: float) -> float:
        if dist_value >= DEFAULT_DIST:
            return 0.0
        return math.exp(-dist_value / decay)

    def weight_for_attrs(attrs: dict) -> float:
        length = float(attrs.get("length", 1.0))
        scenic = max(0.0, min(1.0, float(attrs.get("scenic_score", 0.5))))

        if mode_name == "fast":
            safe = max(0.0, min(1.0, float(attrs.get("safe_score", 0.3))))
            avoid = mode_config.get("length_bias", 1.0)
            avoid += mode_config.get("scenic_avoid", 0.0) * scenic
            avoid += mode_config.get("safe_penalty", 0.0) * safe

            lit_value = str(attrs.get("lit", "")).lower()
            if attrs.get("cycleway") or attrs.get("segregated"):
                avoid += mode_config.get("bike_penalty", 0.0)
            elif str(attrs.get("bicycle", "")).lower() in {"designated", "yes"}:
                avoid += mode_config.get("bike_penalty", 0.0)

            if lit_value in LIT_POSITIVE:
                avoid += mode_config.get("lit_penalty", 0.0)
            elif lit_value in LIT_NEGATIVE:
                avoid = max(0.1, avoid - mode_config.get("dark_reward", 0.0))

            return length * max(mode_config.get("min_factor", 0.5), avoid)

        if mode_name == "safe":
            safe_score = max(0.0, min(1.0, float(attrs.get("safe_score", 0.3))))
            dist_safe = as_float(attrs.get("dist_safe_poi", DEFAULT_DIST), DEFAULT_DIST)
            dist_busy = as_float(attrs.get("dist_busy_area", DEFAULT_DIST), DEFAULT_DIST)
            dist_light = as_float(attrs.get("dist_lighting", DEFAULT_DIST), DEFAULT_DIST)
            dist_crime = as_float(attrs.get("dist_crime_hotspot", DEFAULT_DIST), DEFAULT_DIST)
            crime_density = max(0.0, min(1.0, float(attrs.get("crime_density", 0.0))))
            bad_area = max(0.0, min(1.0, float(attrs.get("bad_area_score", 0.0))))

            safe_aff = affinity(dist_safe, mode_config["safe_decay"])
            busy_aff = affinity(dist_busy, mode_config["busy_decay"])
            light_aff = affinity(dist_light, mode_config["lighting_decay"])
            crime_aff = affinity(dist_crime, mode_config.get("crime_gap_decay", CRIME_NEAR_DECAY))

            reward = (
                mode_config["safe_score_reward"] * safe_score
                + mode_config["safe_anchor_reward"] * safe_aff
                + mode_config["busy_reward"] * busy_aff
                + mode_config["lighting_reward"] * light_aff
                + mode_config["crime_gap_reward"] * max(0.0, 1 - crime_aff)
            )
            penalty = mode_config["unsafe_penalty"] * (1 - safe_score)
            penalty += mode_config["crime_density_penalty"] * crime_density
            penalty += mode_config["bad_area_penalty"] * bad_area
            penalty += mode_config["crime_near_penalty"] * crime_aff

            weight = length * (mode_config["length_bias"] + penalty)
            if reward > 0:
                weight /= 1 + reward
            if safe_score < mode_config["unsafe_floor"]:
                weight *= mode_config["unsafe_multiplier"]

            min_weight = length * mode_config["min_factor"]
            return max(min_weight, weight)

        dist_green = as_float(attrs.get("dist_green", DEFAULT_DIST), DEFAULT_DIST)
        dist_water = as_float(attrs.get("dist_water", DEFAULT_DIST), DEFAULT_DIST)
        green_aff = affinity(dist_green, mode_config.get("green_decay", 200.0))
        water_aff = affinity(dist_water, mode_config.get("water_decay", 200.0))

        reward = (
            mode_config.get("scenic_reward", 1.0) * scenic
            + mode_config.get("green_reward", 0.0) * green_aff
            + mode_config.get("water_reward", 0.0) * water_aff
        )
        penalty = mode_config.get("detour_penalty", 0.0) * (1 - scenic)

        weight = length * (mode_config.get("length_bias", 1.0) + penalty)
        if reward > 0:
            weight /= 1 + reward
        min_weight = length * mode_config.get("min_factor", 0.1)
        return max(min_weight, weight)

    return weight_for_attrs


def make_edge_weight(weight_for_attrs):
    def edge_weight(u, v, data):
        variants = edge_variants(data)
        if not variants:
            return float("inf")
        weights = [weight_for_attrs(attrs) for attrs in variants]
        return min(weights)

    return edge_weight


def build_geometry_and_stats(graph: nx.MultiDiGraph, path, weight_for_attrs):
    coords = []
    raw_stats = init_segment_stats()
    total_length = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        if edge_data is None:
            continue
        best_edge = select_best_variant(edge_data, weight_for_attrs)
        if best_edge is None:
            continue

        segment_length = float(best_edge.get("length", 0.0))
        total_length += segment_length
        accumulate_segment_stats(raw_stats, best_edge, segment_length)

        if "geometry" in best_edge and best_edge["geometry"] is not None:
            xs, ys = best_edge["geometry"].xy
            seg_coords = list(zip(xs, ys))
        else:
            y1 = graph.nodes[u]["y"]
            x1 = graph.nodes[u]["x"]
            y2 = graph.nodes[v]["y"]
            x2 = graph.nodes[v]["x"]
            seg_coords = [(x1, y1), (x2, y2)]

        if coords and coords[-1] == seg_coords[0]:
            coords.extend(seg_coords[1:])
        else:
            coords.extend(seg_coords)

    stats = finalize_segment_stats(raw_stats)
    stats["length_m"] = total_length
    return coords, total_length, stats


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


def compute_comparison_ratios(current_stats, baseline_stats):
    if not baseline_stats:
        return {}
    ratios = {}
    for key in COMPARISON_KEYS:
        cur = current_stats.get(key)
        base = baseline_stats.get(key)
        if base is None or not isinstance(base, (int, float)) or base == 0:
            ratios[key] = None
            continue
        ratios[key] = (cur - base) / base
    return ratios


def build_linestring(graph, u, v):
    y1 = graph.nodes[u]["y"]
    x1 = graph.nodes[u]["x"]
    y2 = graph.nodes[v]["y"]
    x2 = graph.nodes[v]["x"]
    return LineString([(x1, y1), (x2, y2)])


def load_context_tree(place: str, tags: dict, target_crs):
    try:
        gdf = ox.features_from_place(place, tags=tags)
    except Exception as exc:
        print(f"Warning: could not download {tags} shapes: {exc}")
        return None

    if gdf.empty:
        return None

    try:
        gdf = gdf.to_crs(target_crs)
    except Exception as exc:
        print(f"Warning: could not project context geometries: {exc}")
        return None

    geoms = [
        geom
        for geom in gdf.geometry
        if isinstance(geom, BaseGeometry) and geom is not None and not geom.is_empty
    ]
    if not geoms:
        return None
    return {"tree": STRtree(geoms), "geoms": geoms}


def enrich_graph_with_environment(graph: nx.MultiDiGraph):
    projected = ox.project_graph(graph)
    target_crs = projected.graph.get("crs")

    print("Loading green areas...")
    green_tags = {
        "leisure": ["park", "garden", "playground"],
        "landuse": ["recreation_ground", "meadow", "grass"],
        "natural": ["wood", "grassland"],
    }
    green_tree = load_context_tree(SCENIC_PLACE, green_tags, target_crs)

    print("Loading water bodies...")
    water_tags = {"natural": "water", "waterway": ["river", "stream", "canal"]}
    water_tree = load_context_tree(SCENIC_PLACE, water_tags, target_crs)

    print("Loading safety anchors (police / emergency)...")
    safety_tree = load_context_tree(SCENIC_PLACE, SAFE_POI_TAGS, target_crs)

    print("Loading busy public amenities...")
    busy_tree = load_context_tree(SCENIC_PLACE, BUSY_PUBLIC_TAGS, target_crs)

    print("Loading lighting references...")
    lighting_tree = load_context_tree(SCENIC_PLACE, LIGHTING_TAGS, target_crs)

    print("Loading historic crime incidents...")
    crime_points = load_crime_points(target_crs)

    print("Scoring every edge with environmental context...")
    for u, v, k, data_proj in projected.edges(keys=True, data=True):
        geom = data_proj.get("geometry")
        if geom is None or geom.is_empty:
            geom = build_linestring(projected, u, v)

        center = geom.interpolate(0.5, normalized=True)

        nearest_green = resolve_tree_geometry(green_tree, center)
        dist_green = center.distance(nearest_green) if nearest_green is not None else DEFAULT_DIST

        nearest_water = resolve_tree_geometry(water_tree, center)
        dist_water = center.distance(nearest_water) if nearest_water is not None else DEFAULT_DIST

        nearest_safe = resolve_tree_geometry(safety_tree, center)
        dist_safe = center.distance(nearest_safe) if nearest_safe is not None else DEFAULT_DIST

        nearest_busy = resolve_tree_geometry(busy_tree, center)
        dist_busy = center.distance(nearest_busy) if nearest_busy is not None else DEFAULT_DIST

        nearest_light = resolve_tree_geometry(lighting_tree, center)
        dist_light = center.distance(nearest_light) if nearest_light is not None else DEFAULT_DIST

        dist_crime = DEFAULT_DIST
        crime_density = 0.0
        crime_count = 0
        bad_area = 0.0
        if crime_points is not None:
            nearest_crime = resolve_tree_geometry(crime_points, center)
            if nearest_crime is not None:
                dist_crime = center.distance(nearest_crime)

            buffer_geom = center.buffer(CRIME_INFLUENCE_RADIUS)
            nearby_crimes = crime_points["tree"].query(buffer_geom)
            for incident_candidate in nearby_crimes:
                incident = tree_geometry_from_candidate(crime_points, incident_candidate)
                if incident is None:
                    continue
                dist = center.distance(incident)
                if dist > CRIME_INFLUENCE_RADIUS:
                    continue
                weight = crime_points["weights"].get(id(incident), 1.0)
                influence = weight * math.exp(-max(dist, 1.0) / CRIME_SIGMA)
                crime_density += influence
                crime_count += 1

            if crime_density > 0:
                bad_area = min(1.0, crime_density / CRIME_DENSITY_NORMALIZER)

        data = graph[u][v][k]
        data["dist_green"] = float(dist_green)
        data["dist_water"] = float(dist_water)
        data["scenic_score"] = compute_scenic_score(data)
        data["dist_safe_poi"] = float(dist_safe)
        data["dist_busy_area"] = float(dist_busy)
        data["dist_lighting"] = float(dist_light)
        data["safe_score"] = compute_safety_score(data)
        data["dist_crime_hotspot"] = float(dist_crime)
        data["crime_density"] = float(min(1.0, crime_density))
        data["crime_count_close"] = int(crime_count)
        data["bad_area_score"] = float(bad_area)


def graph_has_environmental_data(graph: nx.MultiDiGraph) -> bool:
    for _, _, data in graph.edges(data=True):
        if not all(key in data for key in REQUIRED_EDGE_KEYS):
            return False
    return True


def build_node_cache(graph: nx.MultiDiGraph):
    """Precompute node ids and radian coordinates for manual nearest lookup."""
    node_ids = list(graph.nodes)
    lons = np.array([graph.nodes[n]["x"] for n in node_ids], dtype=float)
    lats = np.array([graph.nodes[n]["y"] for n in node_ids], dtype=float)
    return {
        "ids": node_ids,
        "lon_rad": np.deg2rad(lons),
        "lat_rad": np.deg2rad(lats),
    }


def get_node_cache(graph: nx.MultiDiGraph):
    key = id(graph)
    if key not in NODE_CACHE:
        NODE_CACHE[key] = build_node_cache(graph)
    return NODE_CACHE[key]


def fallback_nearest_node(graph: nx.MultiDiGraph, lon: float, lat: float):
    cache = get_node_cache(graph)
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    dlon = cache["lon_rad"] - lon_rad
    dlat = cache["lat_rad"] - lat_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad) * np.cos(cache["lat_rad"]) * np.sin(dlon / 2) ** 2
    idx = int(np.argmin(a))
    return cache["ids"][idx]


def get_nearest_nodes(
    graph: nx.MultiDiGraph,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
):
    try:
        start_node = ox.distance.nearest_nodes(graph, start_lon, start_lat)
        end_node = ox.distance.nearest_nodes(graph, end_lon, end_lat)
        return start_node, end_node
    except ImportError:
        # Fall back to manual haversine search if optional deps missing
        return (
            fallback_nearest_node(graph, start_lon, start_lat),
            fallback_nearest_node(graph, end_lon, end_lat),
        )


@app.on_event("startup")
def load_graph():
    global G_walk

    if CACHE_PATH.exists():
        print(f"Loading walking graph from cache: {CACHE_PATH}")
        G_walk = ox.load_graphml(CACHE_PATH)
    else:
        print("Down  walking graph for Munich...")
        G_walk = ox.graph_from_place("Munich, Germany", network_type="walk", simplify=True)
        ox.save_graphml(G_walk, CACHE_PATH)
        print(f"Graph cached at {CACHE_PATH}")

    G_walk = ox.distance.add_edge_lengths(G_walk)

    if graph_has_environmental_data(G_walk):
        print("Refreshing scenic scores from cached environmental data...")
        for _, _, _, data in G_walk.edges(keys=True, data=True):
            data["scenic_score"] = compute_scenic_score(data)
    else:
        print("Computing environmental context (green/water) ...")
        try:
            enrich_graph_with_environment(G_walk)
            ox.save_graphml(G_walk, CACHE_PATH)
            print("Updated graph cache with scenic context.")
        except Exception as exc:
            print(f"Warning: scenic enrichment failed, falling back to heuristics ({exc})")
            for _, _, _, data in G_walk.edges(keys=True, data=True):
                data.setdefault("dist_green", DEFAULT_DIST)
                data.setdefault("dist_water", DEFAULT_DIST)
                data.setdefault("dist_safe_poi", DEFAULT_DIST)
                data.setdefault("dist_busy_area", DEFAULT_DIST)
                data.setdefault("dist_lighting", DEFAULT_DIST)
                data.setdefault("dist_crime_hotspot", DEFAULT_DIST)
                data.setdefault("crime_density", 0.0)
                data.setdefault("crime_count_close", 0)
                data.setdefault("bad_area_score", 0.0)
                data["scenic_score"] = compute_scenic_score(data)
                data["safe_score"] = compute_safety_score(data)

    print("Graph ready.")
    # rebuild nearest-node cache for fallback lookups
    NODE_CACHE.clear()
    NODE_CACHE[id(G_walk)] = build_node_cache(G_walk)


def get_graph_for_mode(mode: str):
    if mode in MODE_CONFIG:
        return G_walk
    raise ValueError(f"Unknown mode: {mode}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    if not FRONTEND_DIR.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html missing")
    return FileResponse(index_file)


@app.get("/route")
def route(
    mode: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
):
    global G_walk

    try:
        G = get_graph_for_mode(mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if G is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")

    # Find nearest nodes in the graph
    try:
        start_node, end_node = get_nearest_nodes(G, start_lon, start_lat, end_lon, end_lat)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not find nearest nodes: {exc}")

    try:
        result = compute_route_variant(G, start_node, end_node, mode)
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found")

    baseline_result = None
    if mode != "fast":
        try:
            baseline_result = compute_route_variant(G, start_node, end_node, "fast")
        except nx.NetworkXNoPath:
            baseline_result = None

    comparison_info = None
    if baseline_result is not None:
        ratios = compute_comparison_ratios(result["stats"], baseline_result["stats"])
        comparison_info = {
            "baseline_mode": "fast",
            "baseline_label": MODE_DISPLAY_NAME.get("fast", "Fast"),
            "ratios": ratios,
        }

    properties = {
        "mode": mode,
        "mode_label": MODE_DISPLAY_NAME.get(mode, mode.title()),
        "length_m": result["length_m"],
        "duration_s": result["duration_s"],
        "stats": result["stats"],
    }

    if baseline_result is not None:
        properties["baseline_stats"] = baseline_result["stats"]
        properties["comparison_to_fast"] = comparison_info

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": result["coordinates"],
        },
        "properties": properties,
    }
