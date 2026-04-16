# src/external_data/diagnose_usgs.py
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# force UTF-8 no Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

TURB_CSV  = Path("area8_turbidity_usgs.csv")
CHA_CSV   = Path("area8_chlorophyll_usgs.csv")
IMG_DIR   = Path("track2_download_link_1/Guide to the Second Round_track2"
                 "/test_input_sample/area8_images")
BBOX      = (-123.18, 43.90, -122.18, 45.58)

TURB_MIN, TURB_MAX = 0.1, 1000.0
CHA_MIN,  CHA_MAX  = 0.01, 200.0


def load_and_validate(csv_path, val_min, val_max, label):
    if not csv_path.exists():
        print(f"  [ERRO] arquivo nao encontrado: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["lat"]   = pd.to_numeric(df["lat"],   errors="coerce")
    df["lon"]   = pd.to_numeric(df["lon"],   errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "lat", "lon"]).copy()

    print(f"\n{'='*55}")
    print(f"  {label.upper()}")
    print(f"{'='*55}")
    print(f"  registros totais    : {len(df)}")
    print(f"  estacoes unicas     : {df['site_no'].nunique()}")

    out_range = df[(df["value"] < val_min) | (df["value"] > val_max)]
    df_clean  = df[(df["value"] >= val_min) & (df["value"] <= val_max)].copy()
    print(f"  fora do range       : {len(out_range)} ({len(out_range)/len(df)*100:.1f}%)")
    print(f"  registros validos   : {len(df_clean)}")

    if df_clean.empty:
        return df_clean

    v = df_clean["value"]
    print(f"\n  distribuicao de valores:")
    print(f"    min    = {v.min():.2f}")
    print(f"    p5     = {v.quantile(0.05):.2f}")
    print(f"    median = {v.median():.2f}")
    print(f"    p95    = {v.quantile(0.95):.2f}")
    print(f"    max    = {v.max():.2f}")
    print(f"    std    = {v.std():.2f}")

    df_clean["date_only"] = df_clean["date"].dt.date
    print(f"\n  cobertura temporal:")
    print(f"    primeiro registro : {df_clean['date'].min().date()}")
    print(f"    ultimo registro   : {df_clean['date'].max().date()}")
    print(f"    dias com dados    : {df_clean['date_only'].nunique()}")

    lon_min, lat_min, lon_max, lat_max = BBOX
    in_bbox = df_clean[
        (df_clean["lat"] >= lat_min) & (df_clean["lat"] <= lat_max) &
        (df_clean["lon"] >= lon_min) & (df_clean["lon"] <= lon_max)
    ]
    out_bbox = df_clean[~df_clean.index.isin(in_bbox.index)]
    print(f"\n  cobertura espacial:")
    print(f"    dentro do bbox    : {len(in_bbox)} registros ({in_bbox['site_no'].nunique()} estacoes)")
    print(f"    fora do bbox      : {len(out_bbox)} registros ({out_bbox['site_no'].nunique()} estacoes)")

    print(f"\n  tabela de estacoes (ordenado por dias com dados):")
    print(f"  {'site_no':>15} {'n_dias':>7} {'median':>8} {'lat':>8} {'lon':>10} {'bbox':>5}  nome")
    print(f"  {'-'*90}")

    per_site = (df_clean
                .groupby("site_no")
                .agg(
                    n_dias=("date_only", "nunique"),
                    median=("value", "median"),
                    lat=("lat", "first"),
                    lon=("lon", "first"),
                    nome=("station_nm", "first"),
                )
                .sort_values("n_dias", ascending=False))

    for sid, row in per_site.iterrows():
        in_b = "OK" if (lon_min <= row["lon"] <= lon_max and
                        lat_min <= row["lat"] <= lat_max) else "--"
        nome = str(row["nome"])[:40] if pd.notna(row["nome"]) else ""
        print(f"  {str(sid):>15} {int(row['n_dias']):>7} "
              f"{row['median']:>8.2f} {row['lat']:>8.4f} "
              f"{row['lon']:>10.4f} {in_b:>5}  {nome}")

    return df_clean


def check_tif_coverage(df_turb, df_cha):
    if not IMG_DIR.exists():
        print(f"\n  [AVISO] img_dir nao encontrado: {IMG_DIR}")
        print(f"  Para verificar cobertura de TIFs, ajuste IMG_DIR no script.")
        return

    tif_files = sorted(IMG_DIR.glob("*.tif"))
    if not tif_files:
        print(f"\n  [AVISO] Nenhum TIF em {IMG_DIR}")
        return

    tif_dates = []
    for tif in tif_files:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", tif.name)
        if m:
            tif_dates.append(pd.Timestamp(
                int(m.group(1)), int(m.group(2)), int(m.group(3))
            ).date())

    print(f"\n{'='*55}")
    print(f"  COBERTURA TIF x MEDICAO")
    print(f"{'='*55}")
    print(f"  TIFs disponiveis: {len(tif_dates)}")
    if tif_dates:
        print(f"  periodo: {min(tif_dates)} -> {max(tif_dates)}")

    lon_min, lat_min, lon_max, lat_max = BBOX

    for label, df in [("turbidez", df_turb), ("clorofila", df_cha)]:
        if df.empty:
            print(f"\n  {label}: sem dados")
            continue

        df = df.copy()
        df["date_only"] = df["date"].dt.date
        usgs_dates = set(df["date_only"])

        matches_0 = sum(1 for d in tif_dates if d in usgs_dates)
        matches_1 = sum(1 for d in tif_dates
                        if any(abs((d - u).days) <= 1 for u in usgs_dates))

        in_bbox = df[
            (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
            (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
        ]
        in_bbox_dates = set(in_bbox["date_only"]) if not in_bbox.empty else set()
        m0_bbox = sum(1 for d in tif_dates if d in in_bbox_dates)

        print(f"\n  {label}:")
        print(f"    pares tolerance=0 (todos)  : {matches_0}/{len(tif_dates)} TIFs")
        print(f"    pares tolerance=1 (todos)  : {matches_1}/{len(tif_dates)} TIFs")
        print(f"    pares tolerance=0 (so bbox): {m0_bbox}/{len(tif_dates)} TIFs"
              f"  ({in_bbox['site_no'].nunique() if not in_bbox.empty else 0} estacoes)")


if __name__ == "__main__":
    print("DIAGNOSTICO DOS DADOS USGS - AREA8")
    print("=" * 55)

    df_turb = load_and_validate(TURB_CSV,  TURB_MIN, TURB_MAX, "turbidez")
    df_cha  = load_and_validate(CHA_CSV,   CHA_MIN,  CHA_MAX,  "clorofila-a")

    if not df_turb.empty and not df_cha.empty:
        sites_turb = set(df_turb["site_no"].unique())
        sites_cha  = set(df_cha["site_no"].unique())
        both       = sites_turb & sites_cha
        print(f"\n{'='*55}")
        print(f"  OVERLAP")
        print(f"{'='*55}")
        print(f"  estacoes so turbidez  : {len(sites_turb - sites_cha)}")
        print(f"  estacoes so clorofila : {len(sites_cha - sites_turb)}")
        print(f"  estacoes com os dois  : {len(both)}")
        if both:
            print(f"  site_nos com os dois  : {sorted(both)}")

    check_tif_coverage(df_turb, df_cha)

    print("\n" + "=" * 55)
    print("  DIAGNOSTICO CONCLUIDO")
    print("=" * 55)
