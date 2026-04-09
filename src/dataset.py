# dataset.py  — salvar em fase2/src/dataset.py
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from pathlib import Path

PATCH_SIZE = 11          # pixels de cada lado — justificativa: 11x11 a ~30m/pixel cobre ~330m,
                         # escala típica de variação de turbidez em rios/lagos
HALF       = PATCH_SIZE // 2
N_BANDS    = 12

# TIFs pequenos demais pra extrair patch — excluir do treino
SKIP_TIFS  = {
    'area5_2024-02-29.tif', 'area5_2024-07-13.tif',
    'area6_2024-02-09.tif', 'area6_2024-07-03.tif', 'area6_2024-09-01.tif',
}

def load_patch(tif_path: Path, lon: float, lat: float) -> np.ndarray | None:
    """
    Extrai patch PATCH_SIZExPATCH_SIZE centrado no ponto (lon, lat).
    Retorna array (N_BANDS, PATCH_SIZE, PATCH_SIZE) float32 ou None se inválido.
    """
    with rasterio.open(tif_path) as src:
        py, px = src.index(lon, lat)
        # checa se o ponto está dentro da imagem com margem para o patch
        h, w = src.shape
        if py < HALF or py >= h - HALF or px < HALF or px >= w - HALF:
            return None
        window = rasterio.windows.Window(px - HALF, py - HALF, PATCH_SIZE, PATCH_SIZE)
        patch  = src.read(window=window).astype(np.float32)  # (12, 11, 11)
    # rejeita patches com valores inválidos
    if patch.shape != (N_BANDS, PATCH_SIZE, PATCH_SIZE):
        return None
    if np.isnan(patch).any() or (patch == 0).all():
        return None
    return patch


def compute_indices(patch: np.ndarray) -> np.ndarray:
    eps = 1e-6
    c   = HALF
    b   = patch[:, c, c]
    green, red, nir, swir1 = b[1], b[2], b[3], b[8]
    ndwi  = (green - nir)   / (green + nir   + eps)
    ndti  = (red   - green) / (red   + green + eps)
    ndci  = (b[4]  - red)   / (b[4]  + red   + eps)
    mndwi = (green - swir1) / (green + swir1 + eps)
    # ratio NIR/Red — sensível a sedimento em suspensão
    nir_red = nir / (red + eps)
    return np.array([ndwi, ndti, ndci, mndwi, nir_red], dtype=np.float32)



class WaterQualityDataset(Dataset):
    def __init__(self, records: list[dict], target: str, augment: bool = False):
        """
        records: lista de dicts com chaves 'patch' (np array) e 'label' (float)
        target:  'turb' ou 'cha' — controla a transformação do label
        augment: flips horizontais/verticais aleatórios
        """
        self.records = records
        self.target  = target
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        patch = rec['patch'].copy()

        if self.augment:
            if np.random.rand() > 0.5:
                patch = patch[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                patch = patch[:, ::-1, :].copy()

        indices = compute_indices(patch)              # (5,) agora
        center  = patch[:, HALF, HALF]                # (12,)
        spatial = patch.reshape(N_BANDS, -1).std(-1)  # (12,)

        # features temporais — seno/cosseno do mês capturam ciclicidade
        month     = rec.get('month', 6)
        month_sin = np.float32(np.sin(2 * np.pi * month / 12))
        month_cos = np.float32(np.cos(2 * np.pi * month / 12))

        # coordenadas normalizadas para [-1, 1] (range aprox dos dados: lat 40-45, lon -90 a -74)
        lat_norm = np.float32((rec.get('lat', 42) - 42.5) / 5.0)
        lon_norm = np.float32((rec.get('lon', -82) + 82)  / 10.0)

        # total: 12 + 5 + 12 + 4 = 33 features
        features = np.concatenate([center, indices, spatial,
                                   [month_sin, month_cos, lat_norm, lon_norm]])

        label = np.log1p(rec['label'])
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)