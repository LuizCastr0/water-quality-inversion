# build_records.py — fase2/src/build_records.py
import re
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from collections import defaultdict
from dataset import SKIP_TIFS, HALF, N_BANDS, PATCH_SIZE
from qaa import compute_iops_from_patch   # função QAA que você implementou

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

# Mapeamento dos índices das bandas Sentinel-2 (ajuste conforme seus dados)
# Assumindo ordem comum: B2=Blue(490nm), B3=Green(560nm), B4=Red(665nm)
# O patch tem shape (12, 11, 11) e as bandas estão na posição 1,2,3? 
# Verifique seu dataset.py para confirmar. Exemplo: se as primeiras bandas são B2,B3,B4:
BAND_IDX = {'blue': 1, 'green': 2, 'red': 3}   # índice zero-based

def build_records(target: str) -> list[dict]:
    col = 'turb_value' if target == 'turb' else 'cha_value'
    records = []
    skipped_tif = 0
    skipped_oob = 0
    skipped_null = 0
    skipped_qaa = 0

    for area, cfg in AREAS_CFG.items():
        csv_path = cfg[target]
        if csv_path is None:
            continue

        df = pd.read_csv(csv_path)
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

                        # --- Extração das IOPs via QAA ---
                        try:
                            iops = compute_iops_from_patch(patch, BAND_IDX)
                        except Exception as e:
                            print(f"  QAA falhou para {fname} ponto ({row['Lon']},{row['Lat']}): {e}")
                            skipped_qaa += 1
                            continue

                        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', fname)
                        month = int(match.group(2)) if match else 6

                        # Registro com todas as informações + IOPs
                        record = {
                            'patch':    patch,
                            'label':    float(label),
                            'area':     area,
                            'filename': fname,
                            'month':    month,
                            'lon':      float(row['Lon']),
                            'lat':      float(row['Lat']),
                            # IOPs (adicionadas como chaves separadas)
                            'a_blue': iops['a_blue'],
                            'a_green': iops['a_green'],
                            'a_red': iops['a_red'],
                            'bb_blue': iops['bb_blue'],
                            'bb_green': iops['bb_green'],
                            'bb_red': iops['bb_red'],
                            'ratio_bb_a_blue': iops['ratio_bb_a_blue'],
                            'ratio_bb_a_green': iops['ratio_bb_a_green'],
                            'slope_gamma': iops['slope_gamma'],
                        }
                        records.append(record)

            except Exception as e:
                print(f"  erro ao abrir {fname}: {e}")
                continue

    print(f"[{target}] total válidos (com QAA): {len(records)}")
    print(f"[{target}] skip tif pequeno : {skipped_tif}")
    print(f"[{target}] skip out-of-bounds: {skipped_oob}")
    print(f"[{target}] skip null/zero    : {skipped_null}")
    print(f"[{target}] skip QAA falhou   : {skipped_qaa}")
    return records