# build_records.py — fase2/src/build_records.py
import re
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from collections import defaultdict
from dataset import SKIP_TIFS, HALF, N_BANDS, PATCH_SIZE

_BASE = Path(__file__).resolve().parent.parent

AREAS_CFG = {
    'area1': {
        'img_dir': _BASE / 'track2_download_link_5/area1/area1_images',
        'turb':    _BASE / 'track2_download_link_5/area1/track2_turb_train_point_area1.csv',
        'cha':     _BASE / 'track2_download_link_5/area1/track2_cha_train_point_area1.csv',
    },
    'area2': {
        'img_dir': _BASE / 'track2_download_link_4/area2/area2_images',
        'turb':    _BASE / 'track2_download_link_4/area2/track2_turb_train_point_area2.csv',
        'cha':     None,
    },
    'area3': {
        'img_dir': _BASE / 'track2_download_link_3/area3/area3_images',
        'turb':    _BASE / 'track2_download_link_3/area3/track2_turb_train_point_area3.csv',
        'cha':     None,
    },
    'area5': {
        'img_dir': _BASE / 'track2_download_link_3/area5/area5_images',
        'turb':    _BASE / 'track2_download_link_3/area5/track2_turb_train_point_area5.csv',
        'cha':     _BASE / 'track2_download_link_3/area5/track2_cha_train_point_area5.csv',
    },
    'area6': {
        'img_dir': _BASE / 'track2_download_link_2/area6/area6_images',
        'turb':    _BASE / 'track2_download_link_2/area6/track2_turb_train_point_area6.csv',
        'cha':     _BASE / 'track2_download_link_2/area6/track2_cha_train_point_area6.csv',
    },
    'area7': {
        'img_dir': _BASE / 'track2_download_link_2/area7/area7_images',
        'turb':    _BASE / 'track2_download_link_2/area7/track2_turb_train_point_area7.csv',
        'cha':     _BASE / 'track2_download_link_2/area7/track2_cha_train_point_area7.csv',
    },
}


def build_records(target: str) -> list[dict]:
    col          = 'turb_value' if target == 'turb' else 'cha_value'
    records      = []
    skipped_tif  = 0
    skipped_oob  = 0
    skipped_null = 0

    for area, cfg in AREAS_CFG.items():
        csv_path = cfg[target]
        if csv_path is None:
            continue

        df      = pd.read_csv(csv_path)
        img_dir = Path(cfg['img_dir'])

        rows_by_file = defaultdict(list)
        for _, row in df.iterrows():
            rows_by_file[row['filename']].append(row)

        for fname, rows in rows_by_file.items():
            if fname in SKIP_TIFS:
                skipped_tif += len(rows)
                continue

            tif_path = img_dir / fname
            try:
                with rasterio.open(tif_path) as src:
                    h, w = src.shape
                    for row in rows:
                        label = row[col]
                        if pd.isna(label) or label <= 0:
                            skipped_null += 1
                            continue

                        py, px = src.index(row['Lon'], row['Lat'])
                        if py < HALF or py >= h - HALF or px < HALF or px >= w - HALF:
                            skipped_oob += 1
                            continue

                        window = rasterio.windows.Window(
                            px - HALF, py - HALF, PATCH_SIZE, PATCH_SIZE)
                        patch = src.read(window=window).astype(np.float32)

                        if patch.shape != (N_BANDS, PATCH_SIZE, PATCH_SIZE):
                            skipped_oob += 1
                            continue
                        if np.isnan(patch).any() or (patch == 0).all():
                            skipped_oob += 1
                            continue

                        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', fname)
                        month = int(match.group(2)) if match else 6

                        records.append({
                            'patch':    patch,
                            'label':    float(label),
                            'area':     area,
                            'filename': fname,
                            'month':    month,
                            'lon':      float(row['Lon']),
                            'lat':      float(row['Lat']),
                        })
            except Exception as e:
                print(f"  erro ao abrir {fname}: {e}")
                continue

    print(f"[{target}] total válidos : {len(records)}")
    print(f"[{target}] skip tif peq : {skipped_tif}")
    print(f"[{target}] skip oob     : {skipped_oob}")
    print(f"[{target}] skip null    : {skipped_null}")
    return records