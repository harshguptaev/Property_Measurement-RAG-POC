#!/usr/bin/env python3

import json
import os
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
import math


def _load_predictions(response_json_path: str) -> dict:
    with open(response_json_path, 'r') as f:
        data = json.load(f)
    return data


def _normalize_contour(contour_raw) -> List[Tuple[int, int]]:
    """
    Normalize the contour format into a flat list of (x, y) tuples.
    Handles both [[x,y], [x,y], ...] and [[[x,y]], [[x,y]], ...] variants.
    """
    points: List[Tuple[int, int]] = []
    for item in contour_raw:
        # Case 1: [[[x, y]]] -> item[0] is [x, y]
        if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (list, tuple)) and len(item[0]) == 2:
            x, y = item[0]
            points.append((int(x), int(y)))
        # Case 2: [[x, y]] -> item is [x, y]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            x, y = item
            points.append((int(x), int(y)))
        # Ignore any malformed entries silently
    return points


def _iter_facets(response: dict):
    for pred in response.get('predictions', []):
        contour_raw = pred.get('contour') or []
        bbox = pred.get('bbox')
        klass = pred.get('class')
        score = pred.get('score')
        points = _normalize_contour(contour_raw)
        yield {
            'points': points,
            'bbox': bbox,
            'class': klass,
            'score': score,
        }


def _color_for_index(index: int) -> Tuple[int, int, int, int]:
    # Simple distinct color palette (RGBA with some alpha)
    palette = [
        (255, 0, 0, 96),
        (0, 255, 0, 96),
        (0, 128, 255, 96),
        (255, 165, 0, 96),
        (255, 0, 255, 96),
        (0, 255, 255, 96),
        (160, 32, 240, 96),
        (255, 215, 0, 96),
    ]
    return palette[index % len(palette)]


def draw_facets_canvas(response_json_path: str, output_path: str, background: Tuple[int, int, int] = (255, 255, 255)) -> str:
    """
    Create a new image using the response's image dimensions and draw all facet polygons.
    Saves to output_path (PNG) and returns the output path.
    """
    response = _load_predictions(response_json_path)
    width = int(response.get('image_width') or response.get('width') or 640)
    height = int(response.get('image_height') or response.get('height') or 640)

    # Use RGBA to allow semi-transparent fills
    img = Image.new('RGBA', (width, height), color=(*background, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    for i, facet in enumerate(_iter_facets(response)):
        pts = facet['points']
        if len(pts) >= 3:
            fill = _color_for_index(i)
            outline = (0, 0, 0, 200)
            draw.polygon(pts, fill=fill, outline=outline)

    # Save as PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return output_path


def overlay_facets_on_image(image_path: str, response_json_path: str, output_path: str, outline_width: int = 2, score_threshold: float = 0.70) -> str:
    """
    Overlay facet polygons on top of the original image with translucent fills and outlines.
    Saves to output_path (PNG) and returns the output path.
    """
    base = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')

    response = _load_predictions(response_json_path)

    for i, facet in enumerate(_iter_facets(response)):
        pts = facet['points']
        if len(pts) >= 3:
            # Transparent fill, outline color by confidence score
            score = facet.get('score') or 0.0
            outline = (255, 255, 255, 255) if score >= score_threshold else (0, 0, 0, 255)
            # polygon with no fill
            draw.polygon(pts, fill=None, outline=outline)
            # Draw thicker outline if requested
            if outline_width > 1:
                draw.line(pts + [pts[0]], fill=outline, width=outline_width)

    composited = Image.alpha_composite(base, overlay)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composited.save(output_path)
    return output_path


def overlay_facets_with_lengths_on_image(
    image_path: str,
    response_json_path: str,
    gserve_json_path: str | None,
    output_path: str,
    outline_width: int = 2,
    feet_per_meter: float = 3.28084,
) -> str:
    """
    Overlay facet borders on the original image and label each polygon edge with
    its length in feet, computed using GSD (meters per pixel) from the provided
    gserve_response.json file located in the same folder as the inputs.

    If gserve_json_path is None, the function will look for a file named
    'gserve_response.json' in the same directory as response_json_path.
    """
    base = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')

    # Load response facets
    response = _load_predictions(response_json_path)

    # Load GSD (meters per pixel)
    if gserve_json_path is None:
        gserve_json_path = os.path.join(os.path.dirname(response_json_path), 'gserve_response.json')
    gsd_m_per_px = 1.0
    try:
        with open(gserve_json_path, 'r') as gf:
            gserve = json.load(gf)
            gsd_m_per_px = float(gserve.get('gsd', 1.0))
    except Exception:
        # Fallback to 1.0 m/px if unavailable
        gsd_m_per_px = 1.0

    # Font for labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    outline = (255, 255, 255, 255)

    for facet in _iter_facets(response):
        pts = facet['points']
        if len(pts) < 2:
            continue

        # Draw polygon border only
        draw.line(pts + [pts[0]], fill=outline, width=outline_width)

        # Label each edge with its length
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]

            dx = x2 - x1
            dy = y2 - y1
            px_len = math.hypot(dx, dy)
            meters = px_len * gsd_m_per_px
            feet = meters * feet_per_meter

            # Midpoint and a small normal offset so text doesn't sit on the line
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            norm = math.hypot(dx, dy) or 1.0
            nx = -dy / norm
            ny = dx / norm
            offset = 8.0
            tx = mx + nx * offset
            ty = my + ny * offset

            label = f"{feet:.1f} ft"
            # Draw text with a dark stroke for contrast
            try:
                draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 200))
            except TypeError:
                # Older Pillow without stroke support
                draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    composited = Image.alpha_composite(base, overlay)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composited.save(output_path)
    return output_path
