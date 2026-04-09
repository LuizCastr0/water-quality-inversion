# train.py — fase2/src/train.py
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from dataset import compute_indices, HALF, N_BANDS


# em train.py, substitui build_features por isso
def build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in records:
        patch = r['patch']
        b     = patch[:, HALF, HALF]
        eps   = 1e-6

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

        month     = r.get('month', 6)
        month_sin = np.float32(np.sin(2 * np.pi * month / 12))
        month_cos = np.float32(np.cos(2 * np.pi * month / 12))

        # lat/lon removidos — o modelo estava aprendendo geografia,
        # não óptica da água, e área8 tem coordenadas fora do range de treino
        feats = np.concatenate([b, indices, ratios, spatial, spatial_mean,
                                [month_sin, month_cos]])
        X.append(feats)
        y.append(np.log1p(r['label']))

    return np.array(X), np.array(y)


def train_target(records: list[dict], target: str, save_path: str) -> float:
    print(f"\n=== {target} ===")

    # remove outliers p99
    vals = np.array([r['label'] for r in records])
    cap  = np.quantile(vals, 0.99)
    records = [r for r in records if r['label'] <= cap]
    print(f"  p99 cap={cap:.1f} — {len(records)} pontos restantes")

    X, y = build_features(records)
    print(f"  features: {X.shape}")

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    scores = cross_val_score(gbr, X, y, cv=5, scoring='r2')
    print(f"  CV R² (5-fold): {scores.mean():.3f} ± {scores.std():.3f}")

    # treina no dataset completo para salvar
    gbr.fit(X, y)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbr, save_path)
    print(f"  salvo: {save_path}")

    # importâncias top 10
    nomes = ([f'b{i}' for i in range(12)] +
             ['ndwi','ndti','ndci','mndwi','nir_red'] +
             [f'ratio{i}' for i in range(6)] +
             [f'std_b{i}' for i in range(12)] +
             [f'mean_b{i}' for i in range(12)] +
             ['month_sin','month_cos','lat','lon'])
    imp = sorted(zip(gbr.feature_importances_, nomes), reverse=True)[:10]
    print("  top 10 importâncias:")
    for v, n in imp:
        print(f"    {n:>12}: {v:.3f}")

    return scores.mean()