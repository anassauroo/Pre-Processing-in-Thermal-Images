#!/usr/bin/env python
"""classifica qualidade + aplica filtros adaptativos."""

import argparse, json, pathlib, cv2, numpy as np, pandas as pd, tqdm
from imgutils import (load_gray, entropy, lap_var,
                      clahe_guided, bilateral, EXTS)

def classify_quality(H: float, L: float, thr: dict):
    if H < thr["ENT_P10"] and L < thr["LAP_P10"]:
        return "blurred", "clahe_guided"
    elif H < thr["ENT_P25"] or  L < thr["LAP_P25"]:
        return "medium",  "bilateral"
    return "good", "none"

def process_dir(input_dir, output_dir, thr, samples=10, use_guided=True):
    input_dir, output_dir = map(pathlib.Path, (input_dir, output_dir))
    output_dir.mkdir(exist_ok=True)

    records, thumbs = [], {"blurred":[], "medium":[], "good":[]}
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in EXTS]

    for p in tqdm.tqdm(files, desc="classificando"):
        g  = load_gray(p)
        H, L = entropy(g), lap_var(g)
        label, action = classify_quality(H, L, thr)

        img_u8 = (g*255).astype(np.uint8)
        if label == "blurred":
            out = clahe_guided(img_u8) if use_guided else img_u8
        elif label == "medium":
            out = bilateral(img_u8)
        else:
            out = img_u8

        (output_dir/label).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir/label/p.name), out)

        records.append(dict(file=p.name, entropy=H,
                            lap_var=L, quality=label, filter=action))
        if len(thumbs[label]) < samples:
            thumbs[label].append((img_u8, out))

    return pd.DataFrame(records), thumbs

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--outdir",      default="processed")
    ap.add_argument("--thr_json",    default="thresholds.json")
    ap.add_argument("--csv_name",    default="classificacao.csv")
    args = ap.parse_args()

    thr = json.loads(pathlib.Path(args.thr_json).read_text())
    df, _ = process_dir(args.input_dir, args.outdir, thr)
    df.to_csv(pathlib.Path(args.outdir)/args.csv_name, index=False)
    print("salvo em", pathlib.Path(args.outdir)/args.csv_name)

if __name__ == "__main__":
    cli()
