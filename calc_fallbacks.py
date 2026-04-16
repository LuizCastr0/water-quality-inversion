# rode na raiz do projeto: python calc_fallbacks.py
import pandas as pd

for label, csv, col in [
    ("turb", "area8_turbidity_usgs.csv",   "turbidity"),
    ("cha",  "area8_chlorophyll_usgs.csv",  "chlorophyll"),
]:
    df = pd.read_csv(csv, parse_dates=["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["month"] = df["date"].dt.month
    df = df[(df["value"] > 0) & (df["value"] < (1000 if label=="turb" else 200))]
    
    print(f"\n{label.upper()} — mediana por mês:")
    por_mes = df.groupby("month")["value"].median()
    for m, v in por_mes.items():
        print(f"    {m}: {v:.2f}")
    print(f"  global: {df['value'].median():.2f}")