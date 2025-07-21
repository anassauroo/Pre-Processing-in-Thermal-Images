# Pré-processamento adaptativo para imagens térmicas capturadas por drones  
> Define automaticamente se cada frame precisa de CLAHE + Guided, Bilateral leve ou nenhum filtro antes de entrar no YOLO.


## 1. Visão geral

Este repositório está organizado em **quatro** blocos de código:

| Script / Módulo      | Papel no fluxo                                                                                                                                  | Entradas                             | Principais saídas                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| `imgutils.py`        | Funções utilitárias                                                                                                  | –                                    | –                                                                |
| `calc_thresholds.py` | Varrimento de um conjunto **padrão‑ouro** de imagens boas; calcula métricas de contraste e nitidez e grava limiares estatísticos (percentis). | pasta de referência                  | `metricas_dataset.csv`, `thresholds.json`                        |
| `classify_images.py` | Classifica o diretório de imagens como `good / medium / blurred` aplicando filtros adaptativos conforme o rótulo.                    | pasta de entrada + `thresholds.json` | pasta `processed/`, `classificacao.csv`                          |
| `build_report.py`    | Orquestra a execução do classificador, gera histograma e monta um relatório HTML com miniaturas *before/after*.                                 | pasta de entrada + `thresholds.json` | pasta `output/` |



## 2. Instalação rápida

```bash
python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
```

Dependências principais: **OpenCV‑contrib**, **Pandas**, **Matplotlib**, **Jinja2** & **tqdm**.



## 3. Como usar

```bash
# 1) Calibrar limiares (execute 1× com um dataset representativo)
python calc_thresholds.py examples/ref_good/  --out_json thresholds.json

# 2) Classificar e gerar relatório completo
python build_report.py  examples/sample_batch/  --thr_json thresholds.json  --outdir output/

# 3) Abrir o resultado
xdg-open output/report.html   # ou simplesmente abrir no navegador

```

## 4. Métricas adotadas

| Métrica                     | O que mede                                 | Implementação            |
| --------------------------- | ------------------------------------------ | ------------------------ |
| **Entropia de Shannon**     | Dispersão do histograma ⇒ contraste global | `cv2.calcHist` + `numpy` |
| **Variância do Laplaciano** | Força das bordas ⇒ nitidez                 | `cv2.Laplacian`          |

A classe (`blurred`, `medium`, `good`) é decidida comparando entropia **H** e variação **L** aos percentis **p10** e **p25** calculados no passo de calibração:

```python
if H < ENT_P10 and L < LAP_P10:
    # contraste & nitidez muito baixos (≤ 10 %)
    label, action = 'blurred', 'clahe_guided'
elif H < ENT_P25 or L < LAP_P25:
    # pelo menos uma métrica entre 10 % e 25 %
    label, action = 'medium',  'bilateral'
else:
    label, action = 'good',    'none'
```

* Filtro `clahe_guided` = CLAHE seguido de Guided Filter → realça contraste em imagens muito planas.
* Filtro `bilateral`   = suavização que preserva arestas → remove ruído moderado.