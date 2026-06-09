# PLAN-Etapa3 — Pipeline de Treino Multi-Seed

Plano de execução da **Etapa 3** do projeto "Análise Arquitetural do Congelamento de Camadas em Transformers Multilíngues". Dá continuidade ao `PLAN-Etapa2.md` (Etapa 2 concluída e verificada no Colab) e detalha a Etapa 3 em fases, com roadmap da Etapa 4.

## Overview
A Etapa 3 treina os **12 modelos** do estudo — 4 configurações de congelamento (C1–C4) × 3 seeds `{42, 123, 2024}` — em `S1_train` (EN/Eletrônicos), com early stopping em validação, e registra **curvas de loss + métricas**. O foco é um pipeline **reprodutível, resumível e barato em disco**, que sobreviva a desconexões do Colab. Sem análise estatística aqui — isso é Etapa 4.

## Project Type
**BACKEND / DATA SCIENCE** (Treino de Transformer / fine-tuning supervisionado)

## Seeds padronizadas
`{42, 123, 2024}` — uma execução independente por seed, em cada config.

## Success Criteria
* Função `treinar_run(config, seed)` que: fixa seed → `carregar_modelo` → `freeze_layers` → treina com HF `Trainer` (fp16, early stopping) → salva métricas/curvas no Drive.
* Loop dos **12 treinos** **idempotente**: relê o que já terminou e **pula**, retomando após qualquer desconexão.
* `results.csv` com **48 linhas** (12 runs × 4 células T1–T4) de F1-macro — gerado por **avaliação inline** logo após cada treino (ver D4).
* Curvas de loss (treino vs. validação) por run salvas em JSON.
* Entregável de código: `src/train.py` testado, importável pela análise da Etapa 4.

## Tech Stack
* **DL/NLP:** PyTorch + Hugging Face `transformers` (`Trainer`, `TrainingArguments`, `EarlyStoppingCallback`), `datasets`, `evaluate`.
* **Métricas:** F1-macro (primária), accuracy, F1 por classe — via `sklearn`/`evaluate`.
* **Ambiente:** Google Colab (GPU T4) + Google Drive (persistência de métricas/curvas).

## Pré-requisitos (entra da Etapa 2)
* `src/model.py` no Drive: `carregar_modelo(seed)`, `freeze_layers(model, config)`, `CONFIGS`, `contar_parametros`.
* `tests/test_model.py` (verde).
* Os 5 CSVs em `MyDrive/TrabalhoRNP/data_processed/` (colunas `id, idioma, dominio, label, texto`).
* GPU confirmada (Etapa 2 rodou em T4 15,6 GB).

## File Structure (acréscimos desta etapa)
```
/TrabalhoRNP/  (Drive)
├── src/
│   ├── data_pipeline.py    # (Etapa 1)
│   ├── model.py            # (Etapa 2)
│   └── train.py            # (Etapa 3) Trainer + treinar_run + loop  ← NOVO
├── resultados/             # (Etapa 3) saídas dos treinos             ← NOVO
│   ├── runs/  run_{config}_{seed}.json   # curvas + best eval por run
│   └── results.csv                        # 48 linhas (avaliação inline)
└── notebooks/
    └── ..._Etapa3.ipynb    # (Etapa 3) treino multi-seed (lê do Drive) ← NOVO
```

## ⚙️ Hiperparâmetros (Seção 7 da Metodologia — fixos em todas as configs)
| Parâmetro | Valor |
|-----------|-------|
| Otimizador | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 128 |
| Weight decay | 0.01 |
| Warmup ratio | 0.10 |
| Épocas máximas | 3 |
| Early stopping | paciência 1, monitora `eval_loss` |
| Precisão | fp16 |
| Função de custo | Cross-Entropy (do `*ForSequenceClassification`) |

---

## Decisões a travar antes de codar

| # | Decisão | Recomendação |
|---|---------|--------------|
| **D3** | Laço de treino: `Trainer` vs. loop manual | **HF `Trainer`** ✅ *(travado)* — fp16, warmup, early stopping (callback), logging de curvas (`log_history`) e save/resume com pouco código e baixa superfície de bug nos 12 runs. `freeze_layers` é aplicado **antes** de instanciar o `Trainer`; o otimizador só pega quem tem `requires_grad=True`. |
| **D4** | Onde avaliar nas 4 células e o que guardar em disco | **Avaliação inline + descartar checkpoint** ✅ *(travado)*. 12 checkpoints × ~1,1 GB ≈ **13 GB** — estoura a cota de 15 GB do Drive grátis. Em vez disso: treina → avalia em T1–T4 → grava 4 linhas no `results.csv` → **apaga o checkpoint**. A Etapa 4 vira só análise/gráficos a partir do `results.csv`. |
| **D5** | Validação para early stopping | **Separar um `val'` de dentro do `S1_train` (~10%)** ✅ *(travado)*, deixando `S1_val` **intocado** como T1. Evita o vazamento de usar a própria célula de teste (T1) para seleção de modelo (sem isso, T1 fica otimista e infla os Δ-shift). |
| **D6** | Notebook | **Novo `..._Etapa3.ipynb`** ✅ *(travado)* — importa `src/`, lê do Drive; mantém o runtime enxuto. |

> ⚠️ **Compute/sessão:** 12 treinos em T4 ≈ **1,5–2,5 h**. O Colab grátis desconecta — por isso o loop é **resumível** (pula runs já no `results.csv`). Dá pra rodar em 2–3 sessões: é só re-executar o loop, que ele continua de onde parou.

---

## Task Breakdown

### Fase 3.0 — Setup, dados e split de validação
*   **Priority:** P0 · **Dependencies:** Etapa 2
*   **INPUT:** `src/` + 5 CSVs no Drive.
*   **AÇÕES:** montar Drive; importar `src.model`; carregar `S1_train` e as 4 células de teste; **split D5** (`S1_train` → `train'` 90% / `val'` 10%, estratificado, `seed` fixo); definir `tokenizar(df)`.
*   **OUTPUT:** `Dataset`/tensores tokenizados de `train'` e `val'`; dict das 4 células de teste.
*   **VERIFY:** tamanhos batem (`train'` ≈ 9,6k; `val'` ≈ 1,1k; cada célula ≈ 2,68k); rótulos balanceados.

### Fase 3.1 — Componentes do Trainer *(src/train.py)*
*   **Priority:** P0 · **Dependencies:** 3.0
*   **AÇÕES:**
    *   `compute_metrics(eval_pred)` → F1-macro, accuracy, F1 por classe (sklearn/evaluate).
    *   `criar_training_args(out_dir, seed)` → `TrainingArguments` com os hiperparâmetros (fp16, lr 2e-5, warmup 0.1, wd 0.01, batch 16, `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `greater_is_better=False`, `seed=seed`).
    *   `EarlyStoppingCallback(early_stopping_patience=1)`.
*   **OUTPUT:** funções em `src/train.py`.
*   **VERIFY:** `TrainingArguments` instancia sem erro; `compute_metrics` retorna as chaves esperadas num batch dummy.

### Fase 3.2 — Treino de 1 run *(treinar_run)*
*   **Priority:** P0 · **Dependencies:** 3.1
*   **AÇÕES:** `treinar_run(config, seed)`:
    1.  `fixar_seed(seed)` → `model, tok = carregar_modelo(seed)` → `freeze_layers(model, config)`.
    2.  `Trainer(model, args, train=train', eval=val', compute_metrics, callbacks=[EarlyStopping])` → `train()`.
    3.  Salvar `runs/run_{config}_{seed}.json` com `log_history` (curvas de train/val loss + métricas por época) e o melhor `eval_loss`.
    4.  **(D4)** avaliar o melhor modelo em T1–T4 → 4 linhas no `results.csv`.
    5.  **(D4)** apagar o checkpoint local.
*   **OUTPUT:** 1 JSON de curvas + 4 linhas no `results.csv` por run.
*   **VERIFY:** JSON criado; loss de treino cai; 4 linhas novas com F1-macro ∈ [0,1].

### Fase 3.3 — Loop dos 12 treinos (resumível)
*   **Priority:** P0 · **Dependencies:** 3.2
*   **AÇÕES:** `for config in [C1..C4]: for seed in {42,123,2024}:` — se `(config, seed)` já está no `results.csv`, **pula**; senão `treinar_run`. Tudo gravado incrementalmente no Drive.
*   **OUTPUT:** `results.csv` com **48 linhas**; 12 JSONs de curvas.
*   **VERIFY:** 48 linhas (12 pares únicos × 4 células); nenhum par faltando.

### Fase 3.4 — Sanity das curvas
*   **Priority:** P1 · **Dependencies:** 3.3
*   **AÇÕES:** plotar curvas train/val loss por config (1 seed ou média) a partir dos JSONs.
*   **VERIFY:** val loss desce e estabiliza/early-stop; sem divergência grosseira; C4 (só head) deve ter loss de treino mais alta que C1 — sanity esperado.

---

## Roadmap (Etapa 4 — Avaliação e Gráficos)
A partir do `results.csv` (48 linhas) já produzido aqui:
* Média ± desvio de F1-macro por config × cenário (sobre as 3 seeds).
* **Δ-shift:** `F1(T1) − F1(Tx)` por config; heatmap config × cenário; barplot de Δ-shift com error bars; curvas de loss.
* Estatística: Mann-Whitney U / t pareado + Cohen's d.
* Veredito **H1** (`F1(C2,T3) − F1(C1,T3) ≥ 3`, p<0.1) e **H2** (`F1(C3,T2) − F1(C1,T2) ≥ 3`, p<0.1).

---

## Verification Checklist (antes de fechar a Etapa 3)
- [ ] `src/train.py` versionado e importável; `compute_metrics` e `TrainingArguments` OK.
- [ ] `treinar_run` salva curvas (JSON) + avalia inline em T1–T4.
- [ ] Loop resumível: re-rodar pula runs já feitos.
- [ ] `results.csv` com **48 linhas**, F1-macro válido, 12 pares (config, seed) completos.
- [ ] Curvas de loss plausíveis (sanity 3.4).
- [ ] Split D5 aplicado (T1/`S1_val` não usado para early stopping).
- [ ] Sem estouro de cota do Drive (checkpoints descartados — D4).
