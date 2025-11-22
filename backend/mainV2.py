# actual implementation of features like green spaces parks trees water bridges etc
# -> After getting "prototyp.py" running we can start on trying with this. 
# if this approach works we can actually start on implementing the other 2 navigation 
# modes because its basically the same just with different parameters

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import math
import osmnx as ox
import networkx as nx

from shapely.geometry import LineString
from shapely.strtree import STRtree

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects
G_walk = None
GREEN_TREE = None
WATER_TREE = None


def as_int(value, default):
    try:
        return int(str(value).split()[0])
    except Exception:
        return default


def compute_base_scenic(edge_attrs):
    """
    Base scenic score from road type and speed only.
    Returns value in [0,1].
    """
    h = edge_attrs.get("highway", "")
    maxspeed = as_int(edge_attrs.get("maxspeed", 50), 50)

    score = 0.5  # neutral baseline

    # Prefer small / human oriented streets and paths
    if h in ["footway", "path", "cycleway", "pedestrian", "living_street"]:
        score += 0.3
    elif h in ["residential", "service"]:
        score += 0.15

    # Penalise big car roads
    if h in ["primary", "secondary", "trunk"]:
        score -= 0.3

    # Speed: slower usually feels nicer / calmer
    if maxspeed <= 30:
        score += 0.1
    elif maxspeed >= 70:
        score -= 0.1

    return max(0.0, min(1.0, score))


def compute_scenic_score(edge_attrs):
    """
    Scenic score v2.
    Combines:
      - base scenic from road type + speed
      - distance to green areas
      - distance to water
    Returns value in [0,1].
    """
    base = compute_base_scenic(edge_attrs)

    dist_green = float(edge_attrs.get("dist_green", 9999.0))
    dist_water = float(edge_attrs.get("dist_water", 9999.0))

    # Convert distance to a smooth [0,1] bonus via exponential decay
    sigma_green = 200.0  # ~200 m influence radius
    sigma_water = 300.0  # water feels scenic a bit further out

    if dist_green < 9999:
        g = math.exp(-dist_green / sigma_green)
    else:
        g = 0.0

    if dist_water < 9999:
        w = math.exp(-dist_water / sigma_water)
    else:
        w = 0.0

    score = base + 0.3 * g + 0.2 * w
    return max(0.0, min(1.0, score))


@app.on_event("startup")
def load_graph_and_context():
    global G_walk, GREEN_TREE, WATER_TREE

    print("Loading walking graph for Munich...")
    G_walk = ox.graph_from_place(
        "Munich, Germany", network_type="walk", simplify=True
    )
    G_walk = ox.add_edge_lengths(G_walk)

    # --- Load green spaces (parks, forests, meadows, etc.) ---
    print("Loading green areas from OSM...")
    tags_green = {
        "leisure": ["park", "garden", "playground"],
        "landuse": ["recreation_ground", "meadow", "grass"],
        "natural": ["wood", "grassland"],
    }
    green_gdf = ox.geometries_from_place("Munich, Germany", tags_green)
    green_geoms = [
        geom for geom in green_gdf.geometry if geom is not None
    ]
    GREEN_TREE = STRtree(green_geoms) if green_geoms else None

    # --- Load water (rivers, lakes, canals) ---
    print("Loading water areas from OSM...")
    tags_water = {
        "natural": "water",
        "waterway": ["river", "stream", "canal"],
    }
    water_gdf = ox.geometries_from_place("Munich, Germany", tags_water)
    water_geoms = [
        geom for geom in water_gdf.geometry if geom is not None
    ]
    WATER_TREE = STRtree(water_geoms) if water_geoms else None

    # --- Compute per-edge distances and scenic score ---
    print("Computing distances to green/water and scenic scores...")
    for u, v, k, data in G_walk.edges(keys=True, data=True):
        geom = data.get("geometry")

        # If edge has no geometry, build a simple line between nodes
        if geom is None:
            y1 = G_walk.nodes[u]["y"]
            x1 = G_walk.nodes[u]["x"]
            y2 = G_walk.nodes[v]["y"]
            x2 = G_walk.nodes[v]["x"]
            geom = LineString([(x1, y1), (x2, y2)])

        # Use mid point of the segment as representative
        center = geom.interpolate(0.5, normalized=True)

        # Distance to nearest green area
        dist_green = 9999.0
        if GREEN_TREE is not None:
            nearest_green = GREEN_TREE.nearest(center)
            dist_green = center.distance(nearest_green)

        # Distance to nearest water
        dist_water = 9999.0
        if WATER_TREE is not None:
            nearest_water = WATER_TREE.nearest(center)
            dist_water = center.distance(nearest_water)

        data["dist_green"] = float(dist_green)
        data["dist_water"] = float(dist_water)
        data["scenic_score"] = compute_scenic_score(data)

    print("Graph loaded and scored.")


def get_graph_for_mode(mode: str):
    if mode in ["fast", "scenic"]:
        return G_walk
    raise ValueError(f"Unknown mode: {mode}")


@app.get("/health")
def health():
    return {"status": "ok"}


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
        start_node = ox.nearest_nodes(G, start_lon, start_lat)
        end_node = ox.nearest_nodes(G, end_lon, end_lat)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not find nearest nodes")

    # Weight function for routing
    def edge_weight(u, v, k, data):
        length = data.get("length", 1.0)

        if mode == "fast":
            return length

        if mode == "scenic":
            scenic = data.get("scenic_score", 0.5)
            beta = 0.6  # how strongly scenic-ness shortens edges
            return length * (1.0 - beta * scenic)

        return length

    # Compute shortest path
    try:
        path = nx.shortest_path(G, start_node, end_node, weight=edge_weight)
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found")

    # Build geometry from path
    coords = []
    total_length = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            continue

        # pick first edge variant
        edge = list(edge_data.values())[0]

        total_length += edge.get("length", 0.0)

        if "geometry" in edge:
            xs, ys = edge["geometry"].xy
            seg_coords = list(zip(xs, ys))  # lon, lat
        else:
            y1 = G.nodes[u]["y"]
            x1 = G.nodes[u]["x"]
            y2 = G.nodes[v]["y"]
            x2 = G.nodes[v]["x"]
            seg_coords = [(x1, y1), (x2, y2)]

        # Avoid duplicating vertices
        if coords and coords[-1] == seg_coords[0]:
            coords.extend(seg_coords[1:])
        else:
            coords.extend(seg_coords)

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,  # lon, lat
        },
        "properties": {
            "mode": mode,
            "length_m": total_length,
        },
    }