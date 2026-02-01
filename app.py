from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

import cv2
import rasterio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from pyproj import CRS
from src.extract_markings import ExtractConfig, extract_rgb_array_to_vectors, extract_to_vectors


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """
    arr: (3,H,W) or (H,W,3). Returns (H,W,3) uint8.
    """
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32)
    lo = np.percentile(arr_f, 1)
    hi = np.percentile(arr_f, 99)
    if hi <= lo:
        hi = lo + 1.0
    arr_f = np.clip((arr_f - lo) * (255.0 / (hi - lo)), 0, 255)
    return arr_f.astype(np.uint8)


def _decode_uploaded_image_bytes(buf: bytes) -> np.ndarray:
    """
    Decode PNG/JPG bytes to uint8 RGB (H,W,3).
    """
    data = np.frombuffer(buf, dtype=np.uint8)
    bgr_or_gray = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if bgr_or_gray is None:
        raise ValueError("Could not decode image bytes.")

    if bgr_or_gray.ndim == 2:
        rgb = cv2.cvtColor(bgr_or_gray, cv2.COLOR_GRAY2RGB)
        return rgb

    if bgr_or_gray.ndim == 3 and bgr_or_gray.shape[2] >= 3:
        bgr = bgr_or_gray[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    raise ValueError(f"Unsupported image shape: {bgr_or_gray.shape}")


def _draw_vectors_on_image(
    image_rgb: np.ndarray,
    gpkg_path: Path,
    transform: rasterio.transform.Affine,
    scale_factor: float = 1.0,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw polygons/lines from GeoPackage onto the image.
    transform: maps original pixels -> map coordinates.
    scale_factor: original pixels / image pixels (typically >= 1.0).
    """
    vis = image_rgb.copy()
    
    # Helper to convert map xy -> display xy
    inv_tf = ~transform
    def map_to_disp(x, y):
        # map -> original pixel
        c, r = inv_tf * (x, y)
        # original pixel -> display pixel
        return int(c / scale_factor), int(r / scale_factor)

    # 1. Polygons
    try:
        gdf = gpd.read_file(gpkg_path, layer="markings_polygons")
        if not gdf.empty:
            # Check for 'color' column
            has_color = "color" in gdf.columns
            
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom.is_empty:
                    continue
                
                # Determine color (default red, but use predicted color if available)
                draw_color = color
                if has_color:
                    c_name = row["color"]
                    if c_name == "Yellow":
                        draw_color = (255, 255, 0) # RGB Yellow
                    elif c_name == "White":
                        draw_color = (0, 255, 255) # RGB Cyan (to contrast with white paint)
                
                if geom.geom_type == "Polygon":
                    geoms = [geom]
                elif geom.geom_type == "MultiPolygon":
                    geoms = list(geom.geoms)
                else:
                    geoms = []

                for poly in geoms:
                    ext_coords = list(poly.exterior.coords)
                    pts = [map_to_disp(x, y) for x, y in ext_coords]
                    cv2.polylines(vis, [np.array(pts)], isClosed=True, color=draw_color, thickness=thickness)
                    
                    # Interiors
                    for interior in poly.interiors:
                        int_coords = list(interior.coords)
                        pts = [map_to_disp(x, y) for x, y in int_coords]
                        cv2.polylines(vis, [np.array(pts)], isClosed=True, color=draw_color, thickness=thickness)
    except Exception:
        pass

    # 2. Centerlines (Underlining)
    try:
        ln_gdf = gpd.read_file(gpkg_path, layer="markings_centerlines")
        if not ln_gdf.empty:
            has_color = "color" in ln_gdf.columns
            
            for idx, row in ln_gdf.iterrows():
                geom = row.geometry
                if geom.is_empty:
                    continue
                
                # Centerline colors
                line_color = (0, 255, 255) # Default Cyan
                if has_color:
                    c_name = row["color"]
                    if c_name == "Yellow":
                        line_color = (255, 215, 0) # Gold/Yellow
                    elif c_name == "White":
                        line_color = (0, 0, 255) # Blue? Or Magenta? Let's trying distinct Blue for "underline" effect.
                        # Actually user asked for "underlined".
                        # To make it visible on white paint, use Blue.
                        line_color = (0, 0, 255)

                if geom.geom_type == "LineString":
                    coords = list(geom.coords)
                    pts = [map_to_disp(x, y) for x, y in coords]
                    cv2.polylines(vis, [np.array(pts)], isClosed=False, color=line_color, thickness=2)
                elif geom.geom_type == "MultiLineString":
                     for ls in geom.geoms:
                        coords = list(ls.coords)
                        pts = [map_to_disp(x, y) for x, y in coords]
                        cv2.polylines(vis, [np.array(pts)], isClosed=False, color=line_color, thickness=2)

    except Exception:
        pass

    return vis


def main() -> None:
    st.set_page_config(page_title="Pavement Marking Extractor", layout="wide")
    st.title("Pavement Marking Extractor")

    st.markdown(
        "- Upload a **GeoTIFF / PNG / JPEG**\n"
        "- Click **Run extraction**\n"
        "- Download a **GeoPackage** with polygons + optional centerlines"
    )

    uploaded = st.file_uploader("Input image", type=["tif", "tiff", "png", "jpg", "jpeg"])

    with st.sidebar:
        st.subheader("Parameters")
        tile = st.number_input("Tile size (px)", min_value=256, max_value=4096, value=1024, step=256)
        overlap = st.number_input("Tile overlap (px)", min_value=0, max_value=512, value=64, step=16)
        min_area = st.number_input("Min component area (px)", min_value=0, max_value=1_000_000, value=200, step=50)
        sieve = st.number_input("Sieve speckle (px)", min_value=0, max_value=1_000_000, value=64, step=50)
        write_centerlines = st.checkbox("Write centerlines", value=True)
        simplify = st.number_input("Polygon simplify (units/pixels)", min_value=0.0, value=0.0, step=0.1, format="%.3f")
        min_line_len = st.number_input("Min line length (units/pixels)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
        line_simplify = st.number_input("Line simplify (units/pixels)", min_value=0.0, value=0.0, step=0.1, format="%.3f")

    if not uploaded:
        st.info("Upload an image to begin.")
        return

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        in_path = tmpdir / uploaded.name
        in_path.write_bytes(uploaded.getbuffer())

        suffix = in_path.suffix.lower()
        is_tif = suffix in {".tif", ".tiff"}

        # Read + preview
        c1, c2 = st.columns([1, 1])

        # State to keep track of for visualization
        preview_img = None
        preview_scale = 1.0
        # This transform maps "Original Pixels" -> "Map Coords"
        # If we are in pixel-mode, it might be Identity (or pixel->pixel).
        active_transform = rasterio.transform.Affine.identity()
        
        width = None
        height = None
        rgb_for_extract = None

        if is_tif:
            try:
                with rasterio.open(in_path) as ds:
                    width, height = ds.width, ds.height
                    active_transform = ds.transform # Map pixels -> CRS

                    with c1:
                        st.subheader("Raster info")
                        st.write({"CRS": str(ds.crs) if ds.crs else None, "Size": f"{ds.width} x {ds.height}", "Bands": ds.count})
                        st.write({"Pixel size": ds.res, "Driver": ds.driver})

                    # Downsample preview to keep UI fast
                    max_dim = 900
                    preview_scale = max(ds.width / max_dim, ds.height / max_dim, 1.0)
                    out_w = int(ds.width / preview_scale)
                    out_h = int(ds.height / preview_scale)
                    rgb_preview = ds.read(
                        indexes=[1, 2, 3],
                        out_shape=(3, out_h, out_w),
                        resampling=Resampling.bilinear,
                    )
                    preview_img = _to_uint8_rgb(rgb_preview)
                    with c2:
                        st.subheader("Preview")
                        st.image(preview_img, caption="Downsampled RGB preview", use_container_width=True)
            except Exception as e:
                st.error(f"Could not read this file as a GeoTIFF: {e}")
                return
        else:
            try:
                preview_img = _decode_uploaded_image_bytes(uploaded.getbuffer())
                height, width = preview_img.shape[:2]
                preview_scale = 1.0
                with c1:
                    st.subheader("Image info")
                    st.write({"Size": f"{width} x {height}", "Format": suffix})
                with c2:
                    st.subheader("Preview")
                    st.image(preview_img, caption="RGB preview", use_container_width=True)
                rgb_for_extract = preview_img
            except Exception as e:
                st.error(f"Could not read this file as a PNG/JPEG: {e}")
                return

        with st.sidebar:
            st.subheader("Georeferencing")
            if is_tif:
                georef_mode = st.radio(
                    "Output coordinates",
                    options=["Use GeoTIFF georeferencing", "Override with extent + EPSG", "Pixel coordinates (no CRS)"],
                    index=0,
                )
            else:
                georef_mode = st.radio(
                    "Output coordinates",
                    options=["Pixel coordinates (no CRS)", "Use extent + EPSG"],
                    index=0,
                )

            out_crs_text = ""
            extent = None

            if georef_mode == "Use extent + EPSG" or georef_mode == "Override with extent + EPSG":
                out_crs_text = st.text_input("CRS (example: EPSG:3857)", value="EPSG:3857")
                xmin = st.number_input("Xmin", value=0.0, format="%.6f")
                ymin = st.number_input("Ymin", value=0.0, format="%.6f")
                xmax = st.number_input("Xmax", value=float(width if width else 0), format="%.6f")
                ymax = st.number_input("Ymax", value=float(height if height else 0), format="%.6f")
                extent = (float(xmin), float(ymin), float(xmax), float(ymax))

                st.caption("Tip: Use your image footprint bounds in the chosen CRS.")

        run = st.button("Run extraction", type="primary")
        if not run:
            return

        out_dir = tmpdir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_gpkg = out_dir / "markings.gpkg"

        cfg = ExtractConfig(
            tile_size=int(tile),
            overlap=int(overlap),
            min_area_px=int(min_area),
            sieve_px=int(sieve),
            write_skeleton=bool(write_centerlines),
            simplify_tolerance=float(simplify),
            min_line_length=float(min_line_len),
            line_simplify_tolerance=float(line_simplify),
        )

        # To support overwrite/pixel modes, determine the final transform used for extraction
        final_transform = active_transform
        
        with st.spinner("Extracting markings and writing vectors..."):
            try:
                if is_tif and georef_mode == "Use GeoTIFF georeferencing":
                    # Uses ds.transform internally
                    extract_to_vectors(in_path, out_gpkg, cfg)
                    # final_transform is aleady active_transform
                else:
                    if is_tif:
                        # Read full raster into memory (override/pixel mode).
                        with rasterio.open(in_path) as ds:
                            rgb = ds.read(indexes=[1, 2, 3])
                        rgb_for_extract = _to_uint8_rgb(rgb)
                        # When reading as array, the implicit transform is identity unless we supply one

                    crs = CRS.from_user_input(out_crs_text) if out_crs_text else None
                    transform_override = None
                    if extent is not None and width is not None and height is not None:
                        transform_override = from_bounds(extent[0], extent[1], extent[2], extent[3], width=width, height=height)
                        final_transform = transform_override
                    else:
                        # Pixel mode or just no extent provided for array-based extraction
                        final_transform = rasterio.transform.Affine.identity()

                    extract_rgb_array_to_vectors(
                        rgb_for_extract,
                        out_gpkg,
                        cfg,
                        crs=crs,
                        transform=transform_override,
                    )
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                return

        st.success("Extraction complete!")
        
        # --- Visualisation ---
        # We visualize the results on the preview image (preview_img).
        # We need to map the output coordinates (in GPKG) back to the preview pixels.
        # transform: Original Pixel -> Map Coord
        # scale: Original Pixel -> Preview Pixel
        # So: Map Coord -> (inv transform) -> Original Pixel -> ( / scale ) -> Preview Pixel
        
        st.subheader("Results Preview")
        with st.spinner("Rendering result preview..."):
            try:
                if preview_img is not None:
                    vis_img = _draw_vectors_on_image(
                        preview_img,
                        out_gpkg,
                        transform=final_transform,
                        scale_factor=preview_scale,
                        color=(255, 0, 0), # Red polygons
                        thickness=2
                    )
                    st.image(vis_img, caption="Extracted Markings (Red=Polygons, Yellow=Centerlines)", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render preview: {e}")

        gpkg_bytes = out_gpkg.read_bytes()
        st.download_button(
            "Download GeoPackage (markings.gpkg)",
            data=gpkg_bytes,
            file_name="markings.gpkg",
            mime="application/geopackage+sqlite3",
        )


if __name__ == "__main__":
    main()

