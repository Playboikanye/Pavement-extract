from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import atan2, degrees, pi
from pathlib import Path
from typing import Iterator

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from affine import Affine
from pyproj import CRS
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union
from skimage.morphology import skeletonize
from tqdm import tqdm


@dataclass(frozen=True)
class ExtractConfig:
    tile_size: int = 1024
    overlap: int = 64
    min_area_px: int = 200
    sieve_px: int = 64
    write_skeleton: bool = True
    simplify_tolerance: float = 0.0
    min_line_length: float = 0.0
    line_simplify_tolerance: float = 0.0
    # Mask improvement knobs
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8
    tophat_kernel: int = 15


def _iter_windows(width: int, height: int, tile: int, overlap: int) -> Iterator[rasterio.windows.Window]:
    step = max(1, tile - overlap)
    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            w = min(tile, width - col_off)
            h = min(tile, height - row_off)
            yield rasterio.windows.Window(col_off=col_off, row_off=row_off, width=w, height=h)


def _read_rgb(ds: rasterio.io.DatasetReader, window: rasterio.windows.Window) -> np.ndarray:
    """
    Returns uint8 RGB image with shape (H, W, 3).
    Assumes bands 1..3 are RGB (typical for NAIP/orthos). If not, adjust here.
    """
    arr = ds.read(indexes=[1, 2, 3], window=window, boundless=False)  # (3, H, W)
    arr = np.moveaxis(arr, 0, -1)  # (H, W, 3)

    # Robust scaling to 8-bit for imagery that isn't already uint8.
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        lo = np.percentile(arr_f, 1)
        hi = np.percentile(arr_f, 99)
        if hi <= lo:
            hi = lo + 1.0
        arr_f = np.clip((arr_f - lo) * (255.0 / (hi - lo)), 0, 255)
        arr = arr_f.astype(np.uint8)
    return arr


def _read_rgb_image_file(path: Path) -> np.ndarray:
    """
    Read a normal image (PNG/JPG/...) from disk and return uint8 RGB (H,W,3).
    """
    bgr_or_gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr_or_gray is None:
        raise ValueError(f"Could not read image: {path}")

    if bgr_or_gray.ndim == 2:
        rgb = cv2.cvtColor(bgr_or_gray, cv2.COLOR_GRAY2RGB)
        return rgb

    # If there's alpha, drop it
    if bgr_or_gray.ndim == 3 and bgr_or_gray.shape[2] >= 3:
        bgr = bgr_or_gray[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    raise ValueError(f"Unsupported image shape: {bgr_or_gray.shape}")


def _as_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Accepts (H,W,3) or (3,H,W) and returns uint8 (H,W,3).
    """
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB array, got shape={arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    lo = np.percentile(arr_f, 1)
    hi = np.percentile(arr_f, 99)
    if hi <= lo:
        hi = lo + 1.0
    arr_f = np.clip((arr_f - lo) * (255.0 / (hi - lo)), 0, 255)
    return arr_f.astype(np.uint8)


def transform_from_extent(extent: tuple[float, float, float, float], width: int, height: int) -> Affine:
    """
    extent: (xmin, ymin, xmax, ymax) in the output CRS.
    Returns an Affine transform mapping pixel coords -> CRS coords (north-up).
    """
    xmin, ymin, xmax, ymax = extent
    return from_bounds(xmin, ymin, xmax, ymax, width=width, height=height)


def smart_marking_mask(rgb_u8: np.ndarray, cfg: ExtractConfig) -> dict[str, np.ndarray]:
    """
    "Smarter" baseline with classification:
    - Enhance contrast on L channel (LAB + CLAHE)
    - Highlight bright thin strokes (top-hat)
    - Highlight bright fat strokes (adaptive threshold)
    - Combine with HSV color rules (white, yellow) separately.

    Returns: dict {'White': mask, 'Yellow': mask}
    """
    rgb_u8 = _as_uint8_rgb(rgb_u8)

    # Contrast enhancement (LAB)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=(int(cfg.clahe_grid), int(cfg.clahe_grid)))
    l2 = clahe.apply(l)

    # Top-hat to isolate bright thin markings
    k = max(3, int(cfg.tophat_kernel))
    if k % 2 == 0:
        k += 1
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(l2, cv2.MORPH_TOPHAT, tophat_kernel, iterations=1)
    _, tophat_bin = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive Thresholding for larger markings ("fat" features)
    block_size = 51
    if block_size % 2 == 0:
        block_size += 1
    adaptive_bin = cv2.adaptiveThreshold(
        l2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, -5
    )
    
    # Combined Structure components
    structure_mask = cv2.bitwise_or(tophat_bin, adaptive_bin)

    # Color rules (HSV)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # White-ish: high V, low S
    white_hsv = (v >= 190) & (s <= 70)

    # Yellow-ish: moderate/high V, moderate S, hue around ~20-40 in OpenCV scale [0..179]
    yellow_hsv = (h >= 12) & (h <= 50) & (s >= 60) & (v >= 140)

    # Vegetation Filter (Green areas)
    # Hue 35-85 captures most green vegetation.
    # Saturation > 25 to avoid removing gray pavement.
    green_mask = cv2.inRange(hsv, (35, 25, 0), (85, 255, 255))
    
    # Dilate green mask slightly to cover edges of branches/leaves
    k_green = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, k_green, iterations=2)
    
    # Valid areas are NOT green
    valid_area_inv = cv2.bitwise_not(green_mask)

    # Process each color independently to avoid merging
    results = {}
    
    for name, hsv_mask_bool in [("White", white_hsv), ("Yellow", yellow_hsv)]:
        color_mask = hsv_mask_bool.astype(np.uint8) * 255
        
        # Selection Rule:
        # - White: Only use Top-Hat (thin lines) to avoid selecting road surfaces/glare.
        # - Yellow: Use Combined Structure (thin + fat) as yellow is more distinctive.
        if name == "White":
            base_structure = tophat_bin
        else:
            base_structure = structure_mask

        # Intersection of Structure AND Color AND Non-Vegetation
        mask = cv2.bitwise_and(color_mask, base_structure)
        mask = cv2.bitwise_and(mask, valid_area_inv)

        # Remove small noise and connect strokes
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))

        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
        # directional closing
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_h, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_v, iterations=1)
        
        results[name] = mask

    return results


def _filter_components(mask255: np.ndarray, min_area_px: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask255 > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(num, dtype=bool)
    keep[0] = False
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            keep[i] = True
    out = keep[labels]
    return (out.astype(np.uint8) * 255)


def _polygon_aspect_ratio(p: Polygon) -> float | None:
    try:
        mrr = p.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        if len(coords) < 5:
            return None
        # 4 edges, last==first
        edges = []
        for i in range(4):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            edges.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        edges = sorted(edges)
        if edges[0] == 0:
            return None
        return float(edges[-1] / edges[0])
    except Exception:
        return None


def _vectorize_polygons(mask255: np.ndarray, transform, crs: CRS | None) -> gpd.GeoDataFrame:
    shapes = rasterio.features.shapes(
        (mask255 > 0).astype(np.uint8),
        mask=(mask255 > 0),
        transform=transform,
        connectivity=8,
    )
    geoms: list[Polygon] = []
    for geom, val in shapes:
        if int(val) != 1:
            continue
        g = shape(geom)
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            geoms.append(g)
        elif g.geom_type == "MultiPolygon":
            geoms.extend(list(g.geoms))
    gdf = gpd.GeoDataFrame({"class": ["marking"] * len(geoms)}, geometry=geoms, crs=crs)
    if len(gdf) == 0:
        return gdf
    gdf["area_units2"] = gdf.geometry.area
    gdf["perimeter_units"] = gdf.geometry.length
    gdf["compactness"] = gdf.apply(
        lambda r: float(4 * pi * r["area_units2"] / (r["perimeter_units"] ** 2)) if r["perimeter_units"] > 0 else None,
        axis=1,
    )
    gdf["aspect_ratio"] = [(_polygon_aspect_ratio(p) if isinstance(p, Polygon) else None) for p in gdf.geometry]
    return gdf


def _skeleton_centerlines(mask255: np.ndarray) -> np.ndarray:
    sk = skeletonize((mask255 > 0))
    return (sk.astype(np.uint8) * 255)


def _line_azimuth_deg(line: LineString) -> float | None:
    try:
        coords = list(line.coords)
        if len(coords) < 2:
            return None
        x1, y1 = coords[0]
        x2, y2 = coords[-1]
        ang = degrees(atan2((x2 - x1), (y2 - y1)))  # azimuth: 0=north, 90=east
        if ang < 0:
            ang += 360.0
        return float(ang)
    except Exception:
        return None


def _vectorize_skeleton_lines(skel255: np.ndarray, transform, crs: CRS | None) -> gpd.GeoDataFrame:
    """
    Simple centerline extraction:
    - find contours on skeleton pixels
    - convert each contour polyline to LineString in map coords

    Note: This is a baseline; for production use, you'd want graph-based tracing.
    """
    skel = (skel255 > 0).astype(np.uint8)
    if skel.sum() == 0:
        return gpd.GeoDataFrame({"class": [], "length_units": [], "azimuth_deg": []}, geometry=[], crs=crs)

    contours, _hier = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lines: list[LineString] = []

    # transform maps pixel centers to coordinates
    for cnt in contours:
        if cnt.shape[0] < 10:
            continue
        pts = cnt[:, 0, :]  # (N,2) x,y in pixel coords
        coords = []
        for x, y in pts:
            X, Y = rasterio.transform.xy(transform, y, x, offset="center")
            coords.append((X, Y))
        line = LineString(coords)
        if line.is_empty or line.length == 0:
            continue
        lines.append(line)

    gdf = gpd.GeoDataFrame({"class": ["marking_centerline"] * len(lines)}, geometry=lines, crs=crs)
    if len(gdf) == 0:
        return gdf
    gdf["length_units"] = gdf.geometry.length
    gdf["azimuth_deg"] = [_line_azimuth_deg(ls) for ls in gdf.geometry]
    return gdf


def extract_to_vectors(input_tif: Path, out_gpkg: Path, cfg: ExtractConfig) -> None:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_tif) as ds:
        crs = ds.crs
        if crs is None:
            raise ValueError("Input raster has no CRS. Please use a georeferenced GeoTIFF.")

        all_polys: list[gpd.GeoDataFrame] = []
        all_lines: list[gpd.GeoDataFrame] = []

        windows = list(_iter_windows(ds.width, ds.height, cfg.tile_size, cfg.overlap))
        for w in tqdm(windows, desc="Processing tiles"):
            rgb = _read_rgb(ds, w)
            masks = smart_marking_mask(rgb, cfg)
            
            for color_name, mask in masks.items():
                mask = _filter_components(mask, cfg.min_area_px)
                if cfg.sieve_px > 0:
                    mask = _filter_components(mask, cfg.sieve_px)

                transform = rasterio.windows.transform(w, ds.transform)
                poly_gdf = _vectorize_polygons(mask, transform=transform, crs=crs)
                if len(poly_gdf) > 0:
                    poly_gdf["color"] = color_name
                    all_polys.append(poly_gdf)

                if cfg.write_skeleton:
                    sk = _skeleton_centerlines(mask)
                    line_gdf = _vectorize_skeleton_lines(sk, transform=transform, crs=crs)
                    if len(line_gdf) > 0:
                        line_gdf["color"] = color_name
                        all_lines.append(line_gdf)

    # Merge + dissolve overlaps
    if len(all_polys) > 0:
        polys = pd_concat(all_polys)
        # dissolve into single multipart then explode for cleaner features
        dissolved_list = []
        # We need to dissolve by color to avoid merging distinctive colors
        for color_name in polys["color"].unique():
            sub = polys[polys["color"] == color_name]
            d = unary_union(sub.geometry.values)
            # Create a 1-row GDF, then explode
            g = gpd.GeoDataFrame({"class": ["marking"], "color": [color_name]}, geometry=[d], crs=all_polys[0].crs)
            dissolved_list.append(g.explode(index_parts=False))
        
        polys = pd_concat(dissolved_list)
        if cfg.simplify_tolerance > 0:
            polys["geometry"] = polys.geometry.simplify(cfg.simplify_tolerance, preserve_topology=True)
        polys["area_units2"] = polys.geometry.area
        polys["perimeter_units"] = polys.geometry.length
        polys["compactness"] = polys.apply(
            lambda r: float(4 * pi * r["area_units2"] / (r["perimeter_units"] ** 2)) if r["perimeter_units"] > 0 else None,
            axis=1,
        )
        polys["aspect_ratio"] = [(_polygon_aspect_ratio(p) if isinstance(p, Polygon) else None) for p in polys.geometry]
        polys.to_file(out_gpkg, layer="markings_polygons", driver="GPKG")
    else:
        # write empty layer
        gpd.GeoDataFrame(
            {"class": [], "color": [], "area_units2": [], "perimeter_units": [], "compactness": [], "aspect_ratio": []},
            geometry=[],
            crs=crs,
        ).to_file(
            out_gpkg, layer="markings_polygons", driver="GPKG"
        )

    if cfg.write_skeleton:
        if len(all_lines) > 0:
            lines = pd_concat(all_lines)
            if cfg.min_line_length > 0:
                lines = lines[lines.geometry.length >= cfg.min_line_length].copy()
            if cfg.line_simplify_tolerance > 0 and len(lines) > 0:
                lines["geometry"] = lines.geometry.simplify(cfg.line_simplify_tolerance, preserve_topology=False)
            lines["length_units"] = lines.geometry.length
            if "azimuth_deg" not in lines.columns:
                lines["azimuth_deg"] = [_line_azimuth_deg(ls) for ls in lines.geometry]
            lines.to_file(out_gpkg, layer="markings_centerlines", driver="GPKG")
        else:
            gpd.GeoDataFrame({"class": [], "color": [], "length_units": [], "azimuth_deg": []}, geometry=[], crs=crs).to_file(
                out_gpkg, layer="markings_centerlines", driver="GPKG"
            )


def extract_rgb_array_to_vectors(
    rgb: np.ndarray,
    out_gpkg: Path,
    cfg: ExtractConfig,
    *,
    crs: CRS | None = None,
    transform: Affine | None = None,
) -> None:
    """
    Extract markings from a normal RGB image array and write a GeoPackage.

    If you don't provide CRS/transform, output coordinates are in pixel units.
    """
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    rgb_u8 = _as_uint8_rgb(rgb)
    height, width = rgb_u8.shape[:2]
    base_transform = transform if transform is not None else Affine.identity()

    all_polys: list[gpd.GeoDataFrame] = []
    all_lines: list[gpd.GeoDataFrame] = []

    windows = list(_iter_windows(width, height, cfg.tile_size, cfg.overlap))
    for w in tqdm(windows, desc="Processing tiles"):
        r0 = int(w.row_off)
        c0 = int(w.col_off)
        r1 = r0 + int(w.height)
        c1 = c0 + int(w.width)

        tile_rgb = rgb_u8[r0:r1, c0:c1]
        tile_transform = rasterio.windows.transform(w, base_transform)
        
        masks = smart_marking_mask(tile_rgb, cfg)
        for color_name, mask in masks.items():
            mask = _filter_components(mask, cfg.min_area_px)
            if cfg.sieve_px > 0:
                mask = _filter_components(mask, cfg.sieve_px)

            poly_gdf = _vectorize_polygons(mask, transform=tile_transform, crs=crs)
            if len(poly_gdf) > 0:
                poly_gdf["color"] = color_name
                all_polys.append(poly_gdf)

            if cfg.write_skeleton:
                sk = _skeleton_centerlines(mask)
                line_gdf = _vectorize_skeleton_lines(sk, transform=tile_transform, crs=crs)
                if len(line_gdf) > 0:
                    line_gdf["color"] = color_name
                    all_lines.append(line_gdf)

    if len(all_polys) > 0:
        polys = pd_concat(all_polys)
        
        dissolved_list = []
        for color_name in polys["color"].unique():
            sub = polys[polys["color"] == color_name]
            d = unary_union(sub.geometry.values)
            g = gpd.GeoDataFrame({"class": ["marking"], "color": [color_name]}, geometry=[d], crs=crs)
            dissolved_list.append(g.explode(index_parts=False))
        
        polys = pd_concat(dissolved_list)
        if cfg.simplify_tolerance > 0:
            polys["geometry"] = polys.geometry.simplify(cfg.simplify_tolerance, preserve_topology=True)
        polys["area_units2"] = polys.geometry.area
        polys["perimeter_units"] = polys.geometry.length
        polys["compactness"] = polys.apply(
            lambda r: float(4 * pi * r["area_units2"] / (r["perimeter_units"] ** 2)) if r["perimeter_units"] > 0 else None,
            axis=1,
        )
        polys["aspect_ratio"] = [(_polygon_aspect_ratio(p) if isinstance(p, Polygon) else None) for p in polys.geometry]
        polys.to_file(out_gpkg, layer="markings_polygons", driver="GPKG")
    else:
        gpd.GeoDataFrame(
            {"class": [], "color": [], "area_units2": [], "perimeter_units": [], "compactness": [], "aspect_ratio": []},
            geometry=[],
            crs=crs,
        ).to_file(
            out_gpkg, layer="markings_polygons", driver="GPKG"
        )

    if cfg.write_skeleton:
        if len(all_lines) > 0:
            lines = pd_concat(all_lines)
            if cfg.min_line_length > 0:
                lines = lines[lines.geometry.length >= cfg.min_line_length].copy()
            if cfg.line_simplify_tolerance > 0 and len(lines) > 0:
                lines["geometry"] = lines.geometry.simplify(cfg.line_simplify_tolerance, preserve_topology=False)
            lines["length_units"] = lines.geometry.length
            if "azimuth_deg" not in lines.columns:
                lines["azimuth_deg"] = [_line_azimuth_deg(ls) for ls in lines.geometry]
            lines.to_file(out_gpkg, layer="markings_centerlines", driver="GPKG")
        else:
            gpd.GeoDataFrame({"class": [], "color": [], "length_units": [], "azimuth_deg": []}, geometry=[], crs=crs).to_file(
                out_gpkg, layer="markings_centerlines", driver="GPKG"
            )


def pd_concat(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    # Local helper to avoid hard pandas import in the module namespace until needed.
    import pandas as pd

    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline pavement marking extraction to GeoPackage.")
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input image: GeoTIFF (preferred) or PNG/JPG. PNG/JPG outputs pixel coords unless --extent/--crs are given.",
    )
    p.add_argument("--out", required=True, type=Path, help="Output GeoPackage path.")
    p.add_argument("--tile", type=int, default=1024, help="Tile size in pixels for processing.")
    p.add_argument("--overlap", type=int, default=64, help="Tile overlap in pixels.")
    p.add_argument("--min_area", type=int, default=200, help="Min connected component size (px) to keep.")
    p.add_argument("--sieve", type=int, default=64, help="Extra sieve threshold (px) to remove speckle.")
    p.add_argument("--no_skeleton", action="store_true", help="Disable centerline output.")
    p.add_argument("--simplify", type=float, default=0.0, help="Polygon simplify tolerance (CRS units or pixels).")
    p.add_argument("--min_line_length", type=float, default=0.0, help="Drop centerlines shorter than this (units/pixels).")
    p.add_argument("--line_simplify", type=float, default=0.0, help="Centerline simplify tolerance (units/pixels).")
    p.add_argument(
        "--crs",
        type=str,
        default="",
        help="Optional output CRS (e.g., 'EPSG:32614'). For PNG/JPG this enables georeferenced output (use with --extent).",
    )
    p.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="Optional extent for PNG/JPG: xmin ymin xmax ymax in the output CRS. Used to georeference output vectors.",
    )
    args = p.parse_args()

    cfg = ExtractConfig(
        tile_size=args.tile,
        overlap=args.overlap,
        min_area_px=args.min_area,
        sieve_px=args.sieve,
        write_skeleton=(not args.no_skeleton),
        simplify_tolerance=float(args.simplify),
        min_line_length=float(args.min_line_length),
        line_simplify_tolerance=float(args.line_simplify),
    )

    suf = args.input.suffix.lower()
    is_tif = suf in {".tif", ".tiff"}

    if is_tif and not args.extent and not args.crs:
        # Use raster georeferencing as-is.
        extract_to_vectors(args.input, args.out, cfg)
        return

    crs: CRS | None = CRS.from_user_input(args.crs) if args.crs else None

    if is_tif:
        # If user wants to override CRS/extent (or the GeoTIFF lacks CRS), read into an array.
        with rasterio.open(args.input) as ds:
            rgb = ds.read(indexes=[1, 2, 3])
        rgb_u8 = _as_uint8_rgb(rgb)
    else:
        rgb_u8 = _read_rgb_image_file(args.input)

    transform = None
    if args.extent is not None:
        extent = (float(args.extent[0]), float(args.extent[1]), float(args.extent[2]), float(args.extent[3]))
        transform = transform_from_extent(extent, width=rgb_u8.shape[1], height=rgb_u8.shape[0])

    extract_rgb_array_to_vectors(rgb_u8, args.out, cfg, crs=crs, transform=transform)


if __name__ == "__main__":
    main()
