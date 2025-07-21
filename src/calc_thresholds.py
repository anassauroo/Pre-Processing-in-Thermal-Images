#!/usr/bin/env python
"""Gera thresholds.json a partir de um conjunto de imagens de referencia."""

import argparse, json, pathlib, pandas as pd, tqdm
from imgutils import load_gray, entropy, lap_var, EXTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="diretorio com imagens representativas")
    ap.add_argument("--out_json", default="thresholds.json")
    ap.add_argument("--csv",      default="metricas_dataset.csv")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.dataset_dir)
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in EXTS]

    records = []
    for p in tqdm.tqdm(files, desc="calculando metricass"):
        g = load_gray(p)
        records.append(dict(
            file=str(p.relative_to(data_dir)),
            entropy=entropy(g),
            lap_var=lap_var(g),
        ))

    df = pd.DataFrame(records).to_csv(args.csv, index=False)

    thr = {
        "ENT_P10": float(df['entropy'].quantile(.10)),
        "ENT_P25": float(df['entropy'].quantile(.25)),
        "LAP_P10": float(df['lap_var'].quantile(.10)),
        "LAP_P25": float(df['lap_var'].quantile(.25)),
    }
    pathlib.Path(args.out_json).write_text(json.dumps(thr, indent=2))
    print("thresholds salvos em", args.out_json)

if __name__ == "__main__":
    main()
