# dataset.py
import numpy as np
import rasterio
from pathlib import Path

PATCH_SIZE = 11
HALF       = PATCH_SIZE // 2
N_BANDS    = 12

SKIP_TIFS = {
    'area5_2024-02-29.tif', 'area5_2024-07-13.tif',
    'area6_2024-02-09.tif', 'area6_2024-07-03.tif', 'area6_2024-09-01.tif',
}

def load_patch(tif_path: Path, lon: float, lat: float):
    with rasterio.open(tif_path) as src:
        py, px = src.index(lon, lat)
        h, w = src.shape
        if py < HALF or py >= h - HALF or px < HALF or px >= w - HALF:
            return None
        window = rasterio.windows.Window(px - HALF, py - HALF, PATCH_SIZE, PATCH_SIZE)
        patch  = src.read(window=window).astype(np.float32)
    if patch.shape != (N_BANDS, PATCH_SIZE, PATCH_SIZE):
        return None
    if np.isnan(patch).any() or (patch == 0).all():
        return None
    return patch


def compute_indices(patch: np.ndarray) -> np.ndarray:
    eps = 1e-6
    b   = patch[:, HALF, HALF]
    green, red, nir, swir1 = b[1], b[2], b[3], b[8]
    ndwi    = (green - nir)   / (green + nir   + eps)
    ndti    = (red   - green) / (red   + green + eps)
    ndci    = (b[4]  - red)   / (b[4]  + red   + eps)
    mndwi   = (green - swir1) / (green + swir1 + eps)
    nir_red = nir / (red + eps)
    return np.array([ndwi, ndti, ndci, mndwi, nir_red], dtype=np.float32)


class WaterQualityDataset:
    """
    Dataset para treino local — importa torch só quando necessário,
    evitando falha no container de inferência que não tem PyTorch.
    """
    def __init__(self, records, target, augment=False):
        import torch
        from torch.utils.data import Dataset
        self.records = records
        self.target  = target
        self.augment = augment
        self._torch  = torch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import torch
        rec   = self.records[idx]
        patch = rec['patch'].copy()

        if self.augment:
            if np.random.rand() > 0.5:
                patch = patch[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                patch = patch[:, ::-1, :].copy()

        indices   = compute_indices(patch)
        center    = patch[:, HALF, HALF]
        spatial   = patch.reshape(N_BANDS, -1).std(-1)

        month     = rec.get('month', 6)
        month_sin = np.float32(np.sin(2 * np.pi * month / 12))
        month_cos = np.float32(np.cos(2 * np.pi * month / 12))

        lat_norm  = np.float32((rec.get('lat', 42) - 42.5) / 5.0)
        lon_norm  = np.float32((rec.get('lon', -82) + 82)  / 10.0)

        features  = np.concatenate([center, indices, spatial,
                                    [month_sin, month_cos, lat_norm, lon_norm]])
        label     = np.log1p(rec['label'])
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)