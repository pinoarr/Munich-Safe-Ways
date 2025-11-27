import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import networkx as nx

# Enable running via `python backend/main.py` or `uvicorn main:app` from inside backend/
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "backend"

from .pathfinder.config import FRONTEND_DIR, MODE_DISPLAY_NAME
from .pathfinder.graph_manager import get_graph_for_mode, get_nearest_nodes, init_graphs
from .pathfinder.routing import (
    build_mode_comparison_statements,
    compute_comparison_ratios,
    compute_route_variant,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_graphs():
    init_graphs()


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
    transport: str = "walk",
):
    transport = (transport or "walk").lower()
    if transport not in {"walk", "bike"}:
        raise HTTPException(status_code=400, detail=f"Unsupported transport: {transport}")

    base_mode = mode
    mode_key = f"bike_{base_mode}" if transport == "bike" else base_mode

    try:
        graph = get_graph_for_mode(mode_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")

    try:
        start_node, end_node = get_nearest_nodes(graph, start_lon, start_lat, end_lon, end_lat)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not find nearest nodes: {exc}")

    try:
        result = compute_route_variant(graph, start_node, end_node, mode_key)
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="No path found")

    def try_variant(name: str):
        variant_key = f"bike_{name}" if transport == "bike" else name
        try:
            return compute_route_variant(graph, start_node, end_node, variant_key)
        except nx.NetworkXNoPath:
            return None

    baseline_result = try_variant("fast") if base_mode != "fast" else result
    safe_result = try_variant("safe") if base_mode != "safe" else result
    scenic_result = try_variant("scenic") if base_mode != "scenic" else result

    comparison_info = None
    if base_mode != "fast" and baseline_result is not None:
        ratios = compute_comparison_ratios(result["stats"], baseline_result["stats"])
        baseline_label = MODE_DISPLAY_NAME.get(
            "bike_fast" if transport == "bike" else "fast",
            MODE_DISPLAY_NAME.get("fast", "Fast"),
        )
        comparison_info = {
            "baseline_mode": "fast",
            "baseline_label": baseline_label,
            "ratios": ratios,
        }

    properties = {
        "mode": base_mode,
        "mode_key": mode_key,
        "transport": transport,
        "mode_label": MODE_DISPLAY_NAME.get(mode_key, MODE_DISPLAY_NAME.get(base_mode, base_mode.title())),
        "length_m": result["length_m"],
        "duration_s": result["duration_s"],
        "stats": result["stats"],
    }

    if base_mode != "fast" and baseline_result is not None:
        properties["baseline_stats"] = baseline_result["stats"]
        properties["comparison_to_fast"] = comparison_info

    peer_results = {
        "fast": baseline_result,
        "safe": safe_result,
        "scenic": scenic_result,
    }
    custom_statements = build_mode_comparison_statements(mode, result, peer_results)
    if custom_statements:
        properties["comparison_custom"] = {"statements": custom_statements}

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": result["coordinates"],
        },
        "properties": properties,
    }

# -