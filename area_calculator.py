#!/usr/bin/env python3

import json
import math
import os
from typing import Dict, List, Tuple, Optional

FEET_PER_METER = 3.28084


def _normalize_contour(contour_raw) -> List[Tuple[float, float]]:
    """
    Normalize contour into a flat list of (x, y) tuples.
    Supports [[[x,y]], ...] and [[x,y], ...] shapes.
    """
    points: List[Tuple[float, float]] = []
    for item in contour_raw or []:
        if (
            isinstance(item, (list, tuple))
            and len(item) == 1
            and isinstance(item[0], (list, tuple))
            and len(item[0]) == 2
        ):
            x, y = item[0]
            points.append((float(x), float(y)))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            x, y = item
            points.append((float(x), float(y)))
    return points


def _polygon_area_pixels(points: List[Tuple[float, float]]) -> float:
    """
    Shoelace formula for polygon area in pixel^2. Returns absolute value.
    """
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _read_gsd(gserve_json_path: str) -> float:
    """Read GSD (meters per pixel) from gserve_response.json. Fallback to 1.0."""
    try:
        with open(gserve_json_path, 'r') as f:
            data = json.load(f)
            gsd = float(data.get('gsd', 1.0))
            return gsd
    except Exception:
        return 1.0


def compute_facet_areas(
    response_json_path: str,
    gserve_json_path: Optional[str] = None,
    score_threshold: float = 0.70,
) -> Dict[str, float]:
    """
    Compute per-facet areas (sq ft) and total area from response JSON.
    - Filters facets by confidence score >= score_threshold
    - Uses GSD from gserve_response.json (meters/pixel) if provided or discovered
    - Converts pixel area to square feet using (gsd * FEET_PER_METER)^2

    Returns a dict like { "facet1": 123.4, ..., "total_sqft": 999.9 }
    """
    with open(response_json_path, 'r') as f:
        response = json.load(f)

    if gserve_json_path is None:
        gserve_json_path = os.path.join(os.path.dirname(response_json_path), 'gserve_response.json')

    gsd_m_per_px = _read_gsd(gserve_json_path)
    feet_per_pixel = gsd_m_per_px * FEET_PER_METER
    area_factor = feet_per_pixel * feet_per_pixel  # (ft/px)^2

    results: Dict[str, float] = {}
    total = 0.0

    predictions = response.get('predictions', [])
    facet_index = 1
    for pred in predictions:
        score = float(pred.get('score', 0.0))
        if score < score_threshold:
            continue
        pts = _normalize_contour(pred.get('contour'))
        if len(pts) < 3:
            continue
        area_px2 = _polygon_area_pixels(pts)
        area_sqft = area_px2 * area_factor
        key = f"facet{facet_index}"
        results[key] = area_sqft
        total += area_sqft
        facet_index += 1

    results['total_sqft'] = total
    return results


def save_area_json(response_json_path: str, output_path: str, gserve_json_path: Optional[str] = None, score_threshold: float = 0.70) -> str:
    """Compute areas and save as JSON to output_path. Returns the output path."""
    data = compute_facet_areas(response_json_path, gserve_json_path, score_threshold)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    return output_path
