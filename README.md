## Pavement Marking Extraction (Starter)

This repo is a **starter** for a Mapathon-style challenge:
- **Detect / Extract / Classify** pavement markings from aerial/satellite imagery
- Convert results into **GIS-ready vector layers** (GeoPackage: polygons + centerlines)

It includes a baseline that works without training data (OpenCV + rasterio), and is designed to be upgraded to a deep-learning segmentation model (U-Net, etc.).

### What you get
- Read a **GeoTIFF** (preferred) or a normal **PNG/JPEG**
- Create a **binary mask** of likely pavement markings (baseline)
- **Vectorize** the mask to:
  - polygons (`markings_polygons` layer)
  - centerlines (`markings_centerlines` layer)
- Output as a single **GeoPackage** (`.gpkg`)

### Setup
Install Python 3.10+.

```bash
pip install -r requirements.txt
```

### Run (baseline)
Put an image at `data/input.tif` (GeoTIFF) or `data/input.png` / `data/input.jpg`.

```bash
python -m src.extract_markings --input data/input.tif --out outputs/markings.gpkg
```

Useful knobs:
- `--min_area`: filter tiny blobs
- `--sieve`: remove speckle in the raster mask
- centerlines are written by default; use `--no_skeleton` to disable

### Using PNG/JPEG (non-georeferenced)
If you run on a normal PNG/JPEG **without** georeferencing, output coordinates are in **pixel units**.

If you know the map bounds, you can georeference the output by providing:
- `--crs` (example: `EPSG:3857`)
- `--extent XMIN YMIN XMAX YMAX` (your image footprint in that CRS)

Example:

```bash
python -m src.extract_markings --input data/input.png --out outputs/markings.gpkg --crs EPSG:3857 --extent 0 0 1000 1000
```

### Making the output cleaner ("smart" output)
These options make the vectors easier to read in GIS (less jagged / less clutter):
- `--simplify`: simplify polygon boundaries
- `--line_simplify`: simplify centerlines
- `--min_line_length`: drop very short line fragments

Example:

```bash
python -m src.extract_markings --input data/input.png --out outputs/markings.gpkg --simplify 0.8 --line_simplify 0.8 --min_line_length 20
```

### Run (web uploader)
Start the local web app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload your **GeoTIFF / PNG / JPEG**, choose an output coordinate mode, and download the output GeoPackage.

### Notes / recommended next upgrades
- Restrict search area to **road corridors** (buffer from road centerlines) to reduce false positives.
- Replace baseline thresholding with a **segmentation model** (U-Net) and keep the same post-processing/vectorization.
- Add a second stage classifier to label marking types (e.g., crosswalk vs lane line) using shape features + CNN.

