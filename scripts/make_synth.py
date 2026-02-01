from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main() -> None:
    Path("data").mkdir(exist_ok=True)

    # Synthetic RGB tile with a few "markings"
    h, w = 512, 512
    img = np.zeros((3, h, w), dtype=np.uint8)
    img[:] = 60  # gray asphalt

    # white lines
    img[:, 120:130, 50:460] = 245
    img[:, 250:260, 80:480] = 245

    # yellow line
    img[0, 360:370, 40:500] = 240  # R
    img[1, 360:370, 40:500] = 220  # G
    img[2, 360:370, 40:500] = 40  # B

    transform = from_origin(500000, 4100000, 0.5, 0.5)  # 0.5m pixel size
    crs = "EPSG:32614"

    out = Path("data/input.tif")
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(img)

    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()

