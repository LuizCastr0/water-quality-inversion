# infer.py
import json
import os
import re
import numpy as np
import pandas as pd
import joblib
import rasterio
from pathlib import Path
from collections import defaultdict
from dataset import compute_indices, HALF, N_BANDS, PATCH_SIZE
from qaa import compute_iops_from_patch   # necessário para o modelo híbrido

INPUT_DIR  = Path('/input')
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/output'))
MODEL_DIR  = Path('/workspace/models')    # ajuste conforme seu Dockerfile

# ========== Fallback values (em escala original) ==========
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

def get_fallback_log(target: str, month: int) -> float:
    """Retorna o valor de fallback em escala log1p."""
    if target == 'turb':
        raw = TURB_MEDIAN_BY_MONTH.get(month, TURB_GLOBAL_MEDIAN)
    else:
        raw = CHA_MEDIAN_BY_MONTH.get(month, CHA_GLOBAL_MEDIAN)
    return np.log1p(raw)

# ========== Extração de features para o modelo original (65 features) ==========
def extract_features_original(patch: np.ndarray, month: int) -> np.ndarray:
    b = patch[:, HALF, HALF]
    eps = 1e-6
    indices = compute_indices(patch)

    ratios_orig = np.array([
        b[0] / (b[1] + eps), b[2] / (b[1] + eps), b[3] / (b[2] + eps),
        b[8] / (b[1] + eps), b[8] / (b[3] + eps), b[4] / (b[2] + eps),
    ], dtype=np.float32)

    ratios_new = np.array([
        b[8] / (b[1] + eps), b[9] / (b[1] + eps),
        b[4] / (b[3] + eps), b[8] / (b[2] + eps),
    ], dtype=np.float32)

    flat = patch.reshape(N_BANDS, -1)
    spatial_std = flat.std(-1)
    spatial_mean = flat.mean(-1)
    spatial_cv = np.clip(spatial_std / (spatial_mean + eps), 0, 5).astype(np.float32)

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    feats = np.concatenate([
        b, indices, ratios_orig, ratios_new,
        spatial_std, spatial_mean, spatial_cv,
        [month_sin, month_cos]
    ])
    return feats.astype(np.float32)

# ========== Extração de features para o modelo híbrido (QAA) ==========
# Mapeamento das bandas Sentinel-2 (ajuste conforme seus dados)
BAND_IDX = {'blue': 1, 'green': 2, 'red': 3}   # exemplo: B2, B3, B4 nas posições 1,2,3

def extract_features_hybrid(patch: np.ndarray, month: int) -> np.ndarray:
    iops = compute_iops_from_patch(patch, BAND_IDX)
    feats = [
        iops['a_blue'], iops['a_green'], iops['a_red'],
        iops['bb_blue'], iops['bb_green'], iops['bb_red'],
        iops['ratio_bb_a_blue'], iops['ratio_bb_a_green'], iops['slope_gamma'],
    ]
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    feats.extend([month_sin, month_cos])
    return np.array(feats, dtype=np.float32)

# ========== Carregamento do bundle ==========
def load_bundle(path: Path):
    bundle = joblib.load(path)
    model = bundle['model']
    selected_idx = bundle.get('selected_idx', list(range(bundle.get('n_features_in', 1))))
    is_hybrid = bundle.get('hybrid', False)
    return model, selected_idx, is_hybrid

# ========== Predição para um arquivo CSV ==========
def predict_csv(csv_path: Path, img_dir: Path, model, selected_idx: list[int],
                target: str, is_hybrid: bool) -> dict:
    df = pd.read_csv(csv_path)
    result = {}

    rows_by_file = defaultdict(list)
    for _, row in df.iterrows():
        rows_by_file[row['filename']].append(row)

    for fname, rows in rows_by_file.items():
        tif = img_dir / fname
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', fname)
        month = int(match.group(2)) if match else 6

        # Fallback se o arquivo não existir
        if not tif.exists():
            for row in rows:
                key = f"{row['filename']}_{row['Lon']}_{row['Lat']}"
                log_pred = get_fallback_log(target, month)
                result[key] = [round(float(np.expm1(log_pred)), 4)]
            continue

        try:
            with rasterio.open(tif) as src:
                h, w = src.shape
                for row in rows:
                    key = f"{row['filename']}_{row['Lon']}_{row['Lat']}"
                    py, px = src.index(row['Lon'], row['Lat'])

                    if py < HALF or py >= h - HALF or px < HALF or px >= w - HALF:
                        log_pred = get_fallback_log(target, month)
                        result[key] = [round(float(np.expm1(log_pred)), 4)]
                        continue

                    window = rasterio.windows.Window(
                        px - HALF, py - HALF, PATCH_SIZE, PATCH_SIZE)
                    patch = src.read(window=window).astype(np.float32)

                    if patch.shape != (N_BANDS, PATCH_SIZE, PATCH_SIZE):
                        log_pred = get_fallback_log(target, month)
                        result[key] = [round(float(np.expm1(log_pred)), 4)]
                        continue

                    # Escolhe a função de extração conforme o tipo de modelo
                    if is_hybrid:
                        feats_full = extract_features_hybrid(patch, month)
                    else:
                        feats_full = extract_features_original(patch, month)

                    feats_sel = feats_full[selected_idx]
                    log_pred = model.predict(feats_sel.reshape(1, -1))[0]
                    result[key] = [round(float(np.expm1(log_pred)), 4)]

        except Exception as e:
            print(f"  erro em {fname}: {e}")
            for row in rows:
                key = f"{row['filename']}_{row['Lon']}_{row['Lat']}"
                log_pred = get_fallback_log(target, month)
                result[key] = [round(float(np.expm1(log_pred)), 4)]

    return result

# ========== Main ==========
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    img_dir = INPUT_DIR / 'area8_images'

    # Carrega os modelos (assumindo que ambos são do mesmo tipo)
    model_turb, idx_turb, hybrid_turb = load_bundle(MODEL_DIR / 'model_turb.joblib')
    model_cha,  idx_cha,  hybrid_cha  = load_bundle(MODEL_DIR / 'model_cha.joblib')

    print(f"Turbidez - modelo híbrido: {hybrid_turb}, features: {len(idx_turb)}")
    print(f"Clorofila - modelo híbrido: {hybrid_cha}, features: {len(idx_cha)}")

    # Para segurança, usa o tipo do modelo de turbidez (ambos devem ser iguais)
    is_hybrid = hybrid_turb

    turb = predict_csv(INPUT_DIR / 'track2_turb_test_point.csv',
                       img_dir, model_turb, idx_turb, 'turb', is_hybrid)
    cha  = predict_csv(INPUT_DIR / 'track2_cha_test_point.csv',
                       img_dir, model_cha, idx_cha, 'cha', is_hybrid)

    with open(OUTPUT_DIR / 'result_turbidity.json', 'w') as f:
        json.dump(turb, f, indent=2)
    with open(OUTPUT_DIR / 'result_chla.json', 'w') as f:
        json.dump(cha, f, indent=2)

    print(f"turbidez : {len(turb)} pontos")
    print(f"chl-a    : {len(cha)} pontos")
    print(f"arquivos escritos em: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()