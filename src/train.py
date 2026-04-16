# train.py
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from dataset import compute_indices, HALF, N_BANDS

# ============================================================
#  FLAG: escolha entre o modelo híbrido (QAA) ou o original (65-features)
# ============================================================
USE_HYBRID = True   # Mude para False se quiser usar o pipeline antigo

# ============================================================
#  Funções para o modelo HÍBRIDO (poucas features físicas)
# ============================================================
def build_features_hybrid(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói features usando IOPs do QAA + mês (seno/cosseno).
    Retorna X com 11 features (9 IOPs + 2 sazonalidade).
    """
    X, y = [], []
    for r in records:
        feats = [
            r['a_blue'], r['a_green'], r['a_red'],
            r['bb_blue'], r['bb_green'], r['bb_red'],
            r['ratio_bb_a_blue'],
            r['ratio_bb_a_green'],
            r['slope_gamma'],
        ]
        month = r.get('month', 6)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        feats.extend([month_sin, month_cos])
        
        X.append(np.array(feats, dtype=np.float32))
        y.append(np.log1p(r['label']))
    
    return np.array(X), np.array(y)

def train_target_hybrid(records: list[dict], target: str, save_path: str) -> dict:
    """Pipeline híbrido: QAA + poucas features, sem seleção SHAP."""
    print(f"\n{'='*52}")
    print(f"  TARGET: {target.upper()} (HÍBRIDO - QAA)")
    print(f"{'='*52}")

    # Outlier clipping (p99)
    vals = np.array([r['label'] for r in records])
    cap = np.quantile(vals, 0.99)
    records = [r for r in records if r['label'] <= cap]
    print(f"  p99 cap={cap:.1f} -> {len(records)} amostras")

    X, y = build_features_hybrid(records)
    print(f"  features: {X.shape[1]}")

    # Validação cruzada com 5 folds
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Gradient Boosting
    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    )
    gbr_scores = cross_val_score(gbr, X, y, cv=cv, scoring='r2')
    print(f"  GBR  CV R2: {gbr_scores.mean():+.3f} +/- {gbr_scores.std():.3f}")

    # XGBoost (regularizado)
    xgb = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring='r2')
    print(f"  XGB  CV R2: {xgb_scores.mean():+.3f} +/- {xgb_scores.std():.3f}")

    # Escolhe o melhor
    if xgb_scores.mean() >= gbr_scores.mean():
        winner_name, winner_model = 'XGBoost', xgb
    else:
        winner_name, winner_model = 'GBR', gbr

    print(f"\n  -> vencedor: {winner_name} (R2={winner_scores.mean():+.3f})")

    winner_model.fit(X, y)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # No modelo híbrido, não há seleção de features, então selected_idx = todas
    bundle = {
        'model': winner_model,
        'selected_idx': list(range(X.shape[1])),
        'selected_names': [f'feat_{i}' for i in range(X.shape[1])],
        'n_features_in': X.shape[1],
        'model_type': winner_name,
        'hybrid': True
    }
    joblib.dump(bundle, save_path)
    print(f"  salvo: {save_path}")
    print(f"  features usadas: {X.shape[1]}")

    return {
        'gbr_r2': gbr_scores.mean(), 'gbr_std': gbr_scores.std(),
        'xgb_r2': xgb_scores.mean(), 'xgb_std': xgb_scores.std(),
        'winner': winner_name,
        'n_sel': X.shape[1], 'n_total': X.shape[1],
        'n_samples': len(records),
    }

# ============================================================
#  Funções para o modelo ORIGINAL (65 features + SHAP)
#  (mantidas como estavam, sem alterações)
# ============================================================
def build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Original: 65 features (bandas, índices, estatísticas espaciais, etc.)"""
    X, y = [], []
    for r in records:
        patch = r['patch']
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

        month = r.get('month', 6)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        feats = np.concatenate([
            b, indices, ratios_orig, ratios_new,
            spatial_std, spatial_mean, spatial_cv,
            [month_sin, month_cos]
        ])
        X.append(feats)
        y.append(np.log1p(r['label']))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def _feature_names() -> list[str]:
    return (
        [f'b{i}' for i in range(12)] +
        ['ndwi', 'ndti', 'ndci', 'mndwi', 'nir_red'] +
        [f'ratio_orig{i}' for i in range(6)] +
        ['r865_r560', 'r945_r560', 'redge_nir', 'swir1_red'] +
        [f'std_b{i}' for i in range(12)] +
        [f'mean_b{i}' for i in range(12)] +
        [f'cv_b{i}' for i in range(12)] +
        ['month_sin', 'month_cos']
    )

def select_features_shap(model, X, feature_names, threshold=0.01, verbose=True):
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap não instalado — pulando seleção")
        return X, list(range(X.shape[1]))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    cutoff = threshold * mean_abs.max()
    selected = np.where(mean_abs >= cutoff)[0]
    if verbose:
        print(f"\n  SHAP selecionadas: {len(selected)}/{X.shape[1]}")
    return X[:, selected], list(selected)

def train_target_original(records: list[dict], target: str, save_path: str) -> dict:
    """Pipeline original (65 features + SHAP) - igual ao seu código."""
    print(f"\n{'='*52}")
    print(f"  TARGET: {target.upper()} (ORIGINAL - 65 features)")
    print(f"{'='*52}")

    vals = np.array([r['label'] for r in records])
    cap = np.quantile(vals, 0.99)
    records = [r for r in records if r['label'] <= cap]
    print(f"  p99 cap={cap:.1f} -> {len(records)} amostras")

    X_full, y = build_features(records)
    names = _feature_names()
    print(f"  features iniciais: {X_full.shape[1]}")

    xgb_pre = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    xgb_pre.fit(X_full, y)
    X_sel, selected_idx = select_features_shap(xgb_pre, X_full, names)
    selected_names = [names[i] for i in selected_idx]

    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    gbr_scores = cross_val_score(gbr, X_sel, y, cv=5, scoring='r2')
    print(f"\n  GBR  CV R2: {gbr_scores.mean():+.3f} +/- {gbr_scores.std():.3f}")

    xgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )
    xgb_scores = cross_val_score(xgb, X_sel, y, cv=5, scoring='r2')
    print(f"  XGB  CV R2: {xgb_scores.mean():+.3f} +/- {xgb_scores.std():.3f}")

    if xgb_scores.mean() >= gbr_scores.mean():
        winner_name, winner_model = 'XGBoost', xgb
        winner_scores = xgb_scores
    else:
        winner_name, winner_model = 'GBR', gbr
        winner_scores = gbr_scores

    print(f"\n  -> vencedor: {winner_name} (R2={winner_scores.mean():+.3f})")
    winner_model.fit(X_sel, y)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        'model': winner_model,
        'selected_idx': selected_idx,
        'selected_names': selected_names,
        'n_features_in': X_full.shape[1],
        'model_type': winner_name,
        'hybrid': False
    }
    joblib.dump(bundle, save_path)
    print(f"  salvo: {save_path} features usadas: {len(selected_idx)}/{X_full.shape[1]}")
    return {
        'gbr_r2': gbr_scores.mean(), 'gbr_std': gbr_scores.std(),
        'xgb_r2': xgb_scores.mean(), 'xgb_std': xgb_scores.std(),
        'winner': winner_name,
        'n_sel': len(selected_idx), 'n_total': X_full.shape[1],
        'n_samples': len(records),
    }

# ============================================================
#  PONTO DE ENTRADA
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_records import build_records

    # Constrói os registros (já com IOPs)
    records_turb = build_records('turb')
    records_cha  = build_records('cha')

    if USE_HYBRID:
        r_turb = train_target_hybrid(records_turb, 'turb', 'models/model_turb.joblib')
        r_cha  = train_target_hybrid(records_cha,  'cha',  'models/model_cha.joblib')
    else:
        r_turb = train_target_original(records_turb, 'turb', 'models/model_turb.joblib')
        r_cha  = train_target_original(records_cha,  'cha',  'models/model_cha.joblib')

    print("\n" + "="*52)
    print("  RESUMO FINAL")
    print("="*52)
    for t, r in [('turb', r_turb), ('cha', r_cha)]:
        print(f"\n  {t}:")
        print(f"    GBR : R2={r['gbr_r2']:+.3f} +/- {r['gbr_std']:.3f}")
        print(f"    XGB : R2={r['xgb_r2']:+.3f} +/- {r['xgb_std']:.3f}")
        print(f"    -> {r['winner']} venceu")
        print(f"    features: {r['n_sel']}/{r['n_total']}  amostras: {r['n_samples']}")