#!/usr/bin/env python
"""Pipeline completo: classifica imagens, gera histograma e relatorio HTML."""

import argparse, json, pathlib, base64, cv2, matplotlib.pyplot as plt
from jinja2 import Template
from classify_images import process_dir
import pandas as pd

def img_b64(img_u8):
    _, buf = cv2.imencode(".jpg", img_u8)
    return base64.b64encode(buf).decode("utf-8")

HTML_TEMPLATE = Template("""
<h2>Relatório – Mini-pipeline v0.1</h2>
<p><b>Total de imagens:</b> {{total}}</p>
<table border="1" cellpadding="4">
  <tr><th>Classe</th><th>Qtd</th></tr>
  {% for c,n in counts.items() %}
    <tr><td>{{c}}</td><td>{{n}}</td></tr>
  {% endfor %}
</table>

<img src="hist.png" width="500"><br>

{% for c in ['blurred','medium','good'] %}
<h3>{{c}}</h3>
<table border="1"><tr>
{% for before,after in thumbs[c] %}
  <td>
    <img src="data:image/jpeg;base64,{{before}}" width="120"><br>
    <img src="data:image/jpeg;base64,{{after}}"  width="120">
  </td>
{% endfor %}
</tr></table>
{% endfor %}
""")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--thr_json", default="thresholds.json")
    ap.add_argument("--outdir",   default="output")
    ap.add_argument("--samples",  type=int, default=10)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(exist_ok=True)
    thr    = json.loads(pathlib.Path(args.thr_json).read_text())

    # ---- 1 classificacao --------------------------------------
    df, thumbs = process_dir(args.input_dir, outdir, thr,
                             samples=args.samples)

    csv_path = outdir/'classificacao.csv'
    df.to_csv(csv_path, index=False)

    # ---- 2 histograma -----------------------------------------
    plt.figure(figsize=(6,3))
    plt.hist(df['entropy'],  bins=30, alpha=.6, label='Entropy')
    plt.xlabel('Entropy'); plt.ylabel('freq')
    plt.twinx()
    plt.hist(df['lap_var'], bins=30, alpha=.3, color='orange',
             label='Laplacian Var'); plt.ylabel('freq')
    plt.title('Distribuição das métricas'); plt.tight_layout()
    plt.savefig(outdir/'hist.png'); plt.close()

    # ---- 3 HTML -----------------------------------------------
    html_str = HTML_TEMPLATE.render(
        total=len(df),
        counts=df['quality'].value_counts().to_dict(),
        thumbs={k:[(img_b64(b), img_b64(a)) for b,a in v]
                for k,v in thumbs.items()}
    )
    (outdir/'report.html').write_text(html_str, encoding='utf-8')
    print("relatorio gerado em", outdir/'report.html')

if __name__ == "__main__":
    main()
