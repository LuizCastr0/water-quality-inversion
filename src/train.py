# train.py
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from dataset import compute_indices, HALF, N_BANDS


def build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói matriz X (n_samples, 65) e vetor y em escala log1p.

    Estrutura das features:
      [0:12]  12 bandas do pixel central
      [12:17]  5 índices espectrais (NDWI, NDTI, NDCI, MNDWI, NIR/Red)
      [17:23]  6 razões espectrais originais
      [23:27]  4 razões físicas novas (motivadas por literatura)
      [27:39] 12 desvios padrão espaciais do patch 11x11
      [39:51] 12 médias espaciais do patch 11x11
      [51:63] 12 coeficientes de variação espacial (std/mean)
      [63:65]  2 features temporais (sin/cos do mês)
    """
    X, y = [], []
    for r in records:
        patch = r['patch']
        b     = patch[:, HALF, HALF]
        eps   = 1e-6

        indices = compute_indices(patch)

        ratios_orig = np.array([
            b[0] / (b[1] + eps),
            b[2] / (b[1] + eps),
            b[3] / (b[2] + eps),
            b[8] / (b[1] + eps),
            b[8] / (b[3] + eps),
            b[4] / (b[2] + eps),
        ], dtype=np.float32)

        # razões com motivação física da literatura:
        # Rrs865/Rrs560: preditor dominante de turbidez global (SHAP analysis,
        #   Pham et al. 2025) — NIR-distante absorvido por água limpa,
        #   razão alta indica sedimento em suspensão
        # Rrs945/Rrs560: complementa b865 quando turbidez satura essa banda
        # red-edge/NIR: sensível a clorofila em águas opticamente complexas
        # SWIR1/red: proxy de tamanho de partícula de sedimento
        ratios_new = np.array([
            b[8] / (b[1] + eps),
            b[9] / (b[1] + eps),
            b[4] / (b[3] + eps),
            b[8] / (b[2] + eps),
        ], dtype=np.float32)

        flat         = patch.reshape(N_BANDS, -1)
        spatial_std  = flat.std(-1)
        spatial_mean = flat.mean(-1)
        # coeficiente de variação: heterogeneidade relativa independente
        # da magnitude — distingue borda de rio vs água aberta
        spatial_cv   = np.clip(
            spatial_std / (spatial_mean + eps), 0, 5
        ).astype(np.float32)

        month     = r.get('month', 6)
        month_sin = np.float32(np.sin(2 * np.pi * month / 12))
        month_cos = np.float32(np.cos(2 * np.pi * month / 12))

        feats = np.concatenate([
            b, indices, ratios_orig, ratios_new,
            spatial_std, spatial_mean, spatial_cv,
            [month_sin, month_cos],
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
        [f'std_b{i}'  for i in range(12)] +
        [f'mean_b{i}' for i in range(12)] +
        [f'cv_b{i}'   for i in range(12)] +
        ['month_sin', 'month_cos']
    )


def select_features_shap(
    model: XGBRegressor,
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.01,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Seleciona features usando SHAP (SHapley Additive exPlanations).

    Por que SHAP e não feature_importances_?
    Feature importance por ganho superestima features com muitos valores
    únicos e é instável entre runs. SHAP calcula a contribuição marginal
    de cada feature para cada predição individual, baseada na teoria dos
    jogos cooperativos de Shapley:

        phi_i = sum_{S in F\{i}} [|S|!(|F|-|S|-1)! / |F|!] * [f(S+i) - f(S)]

    onde F é o conjunto de todas as features e f(S) é a predição usando
    apenas as features em S. Propriedades garantidas: eficiência (soma
    dos SHAP = predição), simetria e dummy (features sem contribuição = 0).

    Usamos TreeSHAP (Lundberg & Lee, NeurIPS 2017) que calcula valores
    exatos em O(TLD^2) aproveitando a estrutura das árvores, em vez de
    O(2^|F|) da abordagem ingênua.

    Critério: mantém features com mean(|SHAP|) >= threshold * max(mean(|SHAP|))
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap nao instalado — pulando selecao")
        return X, list(range(X.shape[1]))

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    cutoff   = threshold * mean_abs.max()
    selected = np.where(mean_abs >= cutoff)[0]

    if verbose:
        print(f"\n  SHAP feature selection (threshold={threshold*100:.0f}% do max):")
        print(f"  features originais   : {X.shape[1]}")
        print(f"  features selecionadas: {len(selected)}")
        print(f"\n  top 15 por SHAP medio absoluto:")
        order = np.argsort(mean_abs)[::-1][:15]
        for rank, idx in enumerate(order, 1):
            marker = "OK" if idx in selected else "--"
            print(f"    {rank:2}. [{marker}] {feature_names[idx]:>14}: "
                  f"{mean_abs[idx]:.4f}")

    return X[:, selected], list(selected)


def train_target(records: list[dict], target: str, save_path: str) -> dict:
    """
    Pipeline completo:
      1. Remove outliers p99
      2. Constrói 65 features
      3. Treina XGBoost preliminar para calcular SHAP values
      4. Seleciona features via SHAP (threshold 1% do maximo)
      5. Retreina GBR e XGBoost final nas features selecionadas
      6. Compara CV R², salva bundle {modelo + indices selecionados}
    """
    print(f"\n{'='*52}")
    print(f"  TARGET: {target.upper()}")
    print(f"{'='*52}")

    vals    = np.array([r['label'] for r in records])
    cap     = np.quantile(vals, 0.99)
    records = [r for r in records if r['label'] <= cap]
    print(f"  p99 cap={cap:.1f} -> {len(records)} amostras")

    X_full, y = build_features(records)
    names     = _feature_names()
    print(f"  features iniciais: {X_full.shape[1]}")

    # XGBoost preliminar rapido so para SHAP
    xgb_pre = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    xgb_pre.fit(X_full, y)
    X_sel, selected_idx = select_features_shap(xgb_pre, X_full, names)
    selected_names = [names[i] for i in selected_idx]

    # GBR nas features selecionadas
    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    gbr_scores = cross_val_score(gbr, X_sel, y, cv=5, scoring='r2')
    print(f"\n  GBR  CV R2: {gbr_scores.mean():+.3f} +/- {gbr_scores.std():.3f}")

    # XGBoost final nas features selecionadas
    # reg_alpha=0.1 (L1) + reg_lambda=1.0 (L2): regularizacao dupla
    # importante para domain shift — forca o modelo a ser conservador
    # em extrapolacao para fora da distribuicao de treino
    xgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        objective='reg:squarederror',
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

    # salva bundle: modelo + indices das features selecionadas
    # o infer.py vai carregar esse bundle e aplicar a mesma selecao
    bundle = {
        'model':          winner_model,
        'selected_idx':   selected_idx,
        'selected_names': selected_names,
        'n_features_in':  X_full.shape[1],
        'model_type':     winner_name,
    }
    joblib.dump(bundle, save_path)
    print(f"  salvo: {save_path}")
    print(f"  features usadas: {len(selected_idx)}/{X_full.shape[1]}")

    return {
        'gbr_r2': gbr_scores.mean(), 'gbr_std': gbr_scores.std(),
        'xgb_r2': xgb_scores.mean(), 'xgb_std': xgb_scores.std(),
        'winner': winner_name,
        'n_sel':  len(selected_idx), 'n_total': X_full.shape[1],
        'n_samples': len(records),
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_records import build_records

    records_turb = build_records('turb')
    records_cha  = build_records('cha')

    r_turb = train_target(records_turb, 'turb', 'models/model_turb.joblib')
    r_cha  = train_target(records_cha,  'cha',  'models/model_cha.joblib')

    print("\n" + "="*52)
    print("  RESUMO FINAL")
    print("="*52)
    for t, r in [('turb', r_turb), ('cha', r_cha)]:
        print(f"\n  {t}:")
        print(f"    GBR : R2={r['gbr_r2']:+.3f} +/- {r['gbr_std']:.3f}")
        print(f"    XGB : R2={r['xgb_r2']:+.3f} +/- {r['xgb_std']:.3f}")
        print(f"    -> {r['winner']} venceu")
        print(f"    features: {r['n_sel']}/{r['n_total']}  "
              f"amostras: {r['n_samples']}")
