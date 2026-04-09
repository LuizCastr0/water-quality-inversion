# infer.py — fase2/src/infer.py
import json
import re
import numpy as np
import pandas as pd
import joblib
import rasterio
from pathlib import Path
from collections import defaultdict
from dataset import compute_indices, HALF, N_BANDS, PATCH_SIZE

INPUT_DIR  = Path('/input')
OUTPUT_DIR = Path('/output')
MODEL_DIR  = Path('/workspace/models')


def extract_features(patch: np.ndarray, month: int) -> np.ndarray:
    b   = patch[:, HALF, HALF]
    eps = 1e-6

    indices = compute_indices(patch)

    ratios = np.array([
        b[0] / (b[1] + eps),
        b[2] / (b[1] + eps),
        b[3] / (b[2] + eps),
        b[8] / (b[1] + eps),
        b[8] / (b[3] + eps),
        b[4] / (b[2] + eps),
    ], dtype=np.float32)

    spatial      = patch.reshape(N_BANDS, -1).std(-1)
    spatial_mean = patch.reshape(N_BANDS, -1).mean(-1)

    month_sin = np.float32(np.sin(2 * np.pi * month / 12))
    month_cos = np.float32(np.cos(2 * np.pi * month / 12))

    return np.concatenate([b, indices, ratios, spatial, spatial_mean,
                           [month_sin, month_cos]])

# substitui o topo e a função predict_csv no infer.py

TURB_MEDIAN_BY_MONTH = {
    1: 7.10, 2: 6.20, 3: 12.60, 4: 31.60, 5: 32.60,
    6: 17.40, 7: 10.30, 8: 16.45, 9: 9.00,
}
TURB_GLOBAL_MEDIAN = 15.40

CHA_MEDIAN_BY_MONTH = {
    1: 4.94, 2: 6.50, 3: 13.30, 4: 14.70, 5: 5.30,
    6: 6.35, 7: 7.56, 8: 11.14, 9: 10.49,
}
CHA_GLOBAL_MEDIAN = 8.30


def get_fallback(target: str, month: int) -> float:
    if target == 'turb':
        return TURB_MEDIAN_BY_MONTH.get(month, TURB_GLOBAL_MEDIAN)
    return CHA_MEDIAN_BY_MONTH.get(month, CHA_GLOBAL_MEDIAN)


def extract_month(fname: str) -> int:
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', fname)
    return int(match.group(2)) if match else 6


def make_key(row) -> str:
    return f"{row['filename']}_{row['Lon']}_{row['Lat']}"


def apply_fallback(result: dict, rows, target: str, month: int):
    value = round(get_fallback(target, month), 4)
    for row in rows:
        result[make_key(row)] = [value]


def process_row(src, row, model, month, h, w, target):
    key = make_key(row)
    py, px = src.index(row['Lon'], row['Lat'])

    if py < HALF or py >= h - HALF or px < HALF or px >= w - HALF:
        return key, get_fallback(target, month)

    window = rasterio.windows.Window(px - HALF, py - HALF, PATCH_SIZE, PATCH_SIZE)
    patch = src.read(window=window).astype(np.float32)

    if patch.shape != (N_BANDS, PATCH_SIZE, PATCH_SIZE):
        return key, get_fallback(target, month)

    feats = extract_features(patch, month)
    log_pred = model.predict(feats.reshape(1, -1))[0]

    return key, float(np.expm1(log_pred))


def process_file(tif: Path, rows, model, target: str, month: int, result: dict):
    if not tif.exists():
        apply_fallback(result, rows, target, month)
        return

    try:
        with rasterio.open(tif) as src:
            h, w = src.shape

            for row in rows:
                key, value = process_row(src, row, model, month, h, w, target)
                result[key] = [round(value, 4)]

    except Exception as e:
        print(f"  erro em {tif.name}: {e}")
        apply_fallback(result, rows, target, month)


def predict_csv(csv_path: Path, img_dir: Path, model, target: str) -> dict:
    df = pd.read_csv(csv_path)
    result = {}

    rows_by_file = defaultdict(list)
    for _, row in df.iterrows():
        rows_by_file[row['filename']].append(row)

    for fname, rows in rows_by_file.items():
        tif = img_dir / fname
        month = extract_month(fname)

        process_file(tif, rows, model, target, month, result)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir    = INPUT_DIR / 'area8_images'
    model_turb = joblib.load(MODEL_DIR / 'model_turb.joblib')
    model_cha  = joblib.load(MODEL_DIR / 'model_cha.joblib')

    turb = predict_csv(INPUT_DIR / 'track2_turb_test_point.csv',
                       img_dir, model_turb, 'turb')
    cha  = predict_csv(INPUT_DIR / 'track2_cha_test_point.csv',
                       img_dir, model_cha,  'cha')

    with open(OUTPUT_DIR / 'result_turbidity.json', 'w') as f:
        json.dump(turb, f, indent=2)
    with open(OUTPUT_DIR / 'result_chla.json', 'w') as f:
        json.dump(cha, f, indent=2)

    print(f"turbidez : {len(turb)} pontos")
    print(f"chl-a    : {len(cha)} pontos")


if __name__ == '__main__':
    main()