# PLAN-Etapa2 — Implementação do Modelo e Algoritmo de Congelamento

Plano de execução da **Etapa 2** do projeto acadêmico de Redes Neurais Profundas: **Análise Arquitetural do Congelamento de Camadas em Transformers Multilíngues**. Dá continuidade ao `PLAN-congelamento-transformers.md` (Etapa 1 concluída) e detalha a Etapa 2 em fases, com roadmap das Etapas 3–4.

## Overview
A Etapa 2 implementa e valida (a) a **arquitetura do classificador** sobre o XLM-RoBERTa-base e (b) a **lógica de congelamento seletivo de camadas** (configurações C1–C4), deixando o modelo pronto para o treino multi-seed da Etapa 3. Nada de treino completo aqui — o foco é ter o modelo e o `freeze_layers` corretos e **verificados por testes**.

## Project Type
**BACKEND / DATA SCIENCE** (Processamento de Linguagem Natural / Aprendizado de Máquina)

## Seeds padronizadas
`{42, 123, 2024}` — usar em todos os experimentos.

## Success Criteria
* Modelo `xlm-roberta-base` carregado com head de classificação binária (saída `[batch, 2]`).
* Função `freeze_layers(model, config)` cobrindo C1–C4, com a *classification head* sempre treinável.
* Testes unitários confirmando `requires_grad` correto por configuração e a **contagem real de parâmetros treináveis** (fecha o `[CONFIRMAR]` da metodologia §6).
* Smoke test com dados reais provando que o gradiente flui **apenas** pelos parâmetros não congelados.
* Entregável de código: `src/model.py` testado, pronto para a Etapa 3.

## Tech Stack
* **Linguagem:** Python 3.10+
* **Deep Learning:** PyTorch
* **NLP:** Hugging Face `transformers` (`XLMRobertaForSequenceClassification`, `XLMRobertaTokenizerFast`)
* **Métricas/análise:** Scikit-Learn, Pandas, Matplotlib, Seaborn
* **Ambiente:** Google Colab (GPU T4/A100) + Google Drive (persistência)
* **IDE:** VS Code / Cursor (prototipação dos módulos `src/`)

## Pré-requisitos (entra da Etapa 1)
* 5 CSVs: `S1_train`, `S1_val`, `S2`, `S3`, `S4` (treino ≈10,7k; cada célula de teste ≈2,7k).
* Módulo `src/data_pipeline.py`.
* ⚠️ **Persistência:** o `/content` do Colab é efêmero. Salvar os 5 CSVs no **Google Drive** e ler de lá na Etapa 2.

## File Structure (acréscimos desta etapa)
```
/Trabalho de redes neurais profundas/
├── src/
│   ├── data_pipeline.py        # (Etapa 1) preparação de dados
│   └── model.py                # (Etapa 2) carregamento do modelo + freeze_layers   ← NOVO
├── tests/
│   └── test_model.py           # (Etapa 2) verificação do congelamento               ← NOVO
└── notebooks/
    └── ..._Etapa2.ipynb        # (Etapa 2) modelagem + congelamento (lê CSVs do Drive) ← NOVO
```

---

## Decisões a travar antes de codar

| # | Decisão | Recomendação |
|---|---------|--------------|
| **D1** | Notebook novo `..._Etapa2.ipynb` vs estender o notebook da Etapa 1 | **Novo notebook** — separa preparação de dados de modelagem, runtime do Colab mais leve, entregáveis isolados. Importa `src/` e lê os CSVs do Drive. |
| **D2** | *Classification head*: padrão do HF vs mínima (só Dropout+Linear) | **Padrão do HF** (`XLMRobertaForSequenceClassification`: dense→tanh→dropout→linear sobre `<s>`). Mais robusto e canônico. Ajustar 1 frase da metodologia §3.2. |
| **D3** | (olhando p/ Etapa 3) HF `Trainer` vs loop manual | **Decidir na Etapa 3.** Não bloqueia a Etapa 2. |

---

## Task Breakdown

### Fase 2.0: Setup e carga dos dados
*   **Priority:** P0 · **Dependencies:** Etapa 1 concluída
*   **INPUT:** 5 CSVs da Etapa 1 (no Google Drive).
*   **OUTPUT:** ambiente do Colab pronto (libs, GPU, seeds), Drive montado e 5 DataFrames carregados.
*   **VERIFY:** os tamanhos batem (treino ≈10,7k; células ≈2,7k).

### Fase 2.1: Arquitetura do Classificador *(Task 2.1)*
*   **Priority:** P0 · **Dependencies:** 2.0
*   **INPUT:** `xlm-roberta-base` (HuggingFace).
*   **AÇÕES:**
    *   `set_seed(seed)` **antes** de instanciar (a head nasce aleatória — seed importa, sobretudo no C4).
    *   `XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)` + `XLMRobertaTokenizerFast`.
    *   Encapsular em `src/model.py`: `carregar_modelo(seed) -> (model, tokenizer)`.
*   **OUTPUT:** `src/model.py` (carregamento) + célula de sanity.
*   **VERIFY:** forward com tensor dummy `[B, 128]` → logits `[B, 2]`; contagem total ≈ **278M** parâmetros.

### Fase 2.2: Lógica de Congelamento C1–C4 *(Task 2.2)*
*   **Priority:** P0 · **Dependencies:** 2.1
*   **INPUT:** `model` + config ∈ {C1, C2, C3, C4}.
*   **AÇÕES:** `freeze_layers(model, config)` setando `requires_grad=False` por prefixo em `named_parameters`. A `classifier.*` **sempre** treinável.

| Config | Nome | Congela | Treinável |
|--------|------|---------|-----------|
| **C1** | Full FT (teto) | — | tudo |
| **C2** | Freeze Lower (testa H1) | `roberta.embeddings.*` + `encoder.layer.0–5` | camadas 6–11 + head |
| **C3** | Freeze Upper (testa H2) | `encoder.layer.6–11` | embeddings + camadas 0–5 + head |
| **C4** | Frozen Encoder (piso) | `roberta.embeddings.*` + `encoder.layer.0–11` | só a head |

*   **OUTPUT:** `freeze_layers` em `src/model.py`, retornando a contagem de treináveis.
*   **VERIFY:** imprime nº de parâmetros treináveis por config.
*   **NOTA:** a matriz de embeddings do XLM-R é gigante (vocab 250k × 768 ≈ **192M**, ~70% do modelo) → congelar embeddings (C2, C4) congela a maioria dos pesos. Os números `[CONFIRMAR]` da metodologia (C2 ~85M, C4 ~1.5k) estão imprecisos e serão corrigidos na Fase 2.3. (Também é material de discussão pro paper.)

### Fase 2.3: Testes de Verificação *(VERIFY da Task 2.2)*
*   **Priority:** P0 · **Dependencies:** 2.2
*   **INPUT:** `freeze_layers` + `model`.
*   **AÇÕES:** `tests/test_model.py` (ou célula de asserts):
    *   Para cada config, varrer `named_parameters` e assertar `requires_grad` correto (congelado vs treinável) por prefixo.
    *   Assertar que `classifier` está sempre treinável.
    *   Registrar a **tabela real de parâmetros treináveis por config** (fecha o `[CONFIRMAR]` da metodologia §6).
*   **OUTPUT:** testes passando + tabela confirmada.
*   **VERIFY:** todos os asserts passam; a contagem bate com a definição de cada config.

### Fase 2.4: Smoke Test com Dados Reais (forward + backward)
*   **Priority:** P1 · **Dependencies:** 2.3
*   **INPUT:** 1 batch real de `S1_train`, tokenizer, model com cada freezing.
*   **AÇÕES:** tokenizar 1 batch (max_len=128) → forward → loss (Cross-Entropy) → backward, para **cada** config.
*   **OUTPUT:** célula de smoke test.
*   **VERIFY:** `.grad` **nulo nos parâmetros congelados** e **não-nulo nos treináveis** (prova que o congelamento funciona no fluxo real); loss finita.

---

## Roadmap (próximos passos — Etapas 3 e 4)

### Etapa 3 — Pipeline de Treino Multi-Seed
* Loop com AdamW (lr 2e-5), warmup 10% dos passos, `fp16`, early stopping (paciência 1 em `eval_loss`), 3 épocas.
* **12 treinos** = 4 configs × 3 seeds `{42,123,2024}`. Salvar curvas de loss + checkpoints no Drive.
* Saída: `src/train.py` + métricas por run em JSON.

### Etapa 4 — Avaliação Zero-Shot e Gráficos
* Cada modelo avaliado nas 4 células (T1=S1-val, T2=S2, T3=S3, T4=S4) → `results.csv` com **48 linhas** (12 treinos × 4 cenários).
* Visualizações: heatmap config × cenário, barplot de Δ-shift com error bars, curvas de loss.
* Estatística: Mann-Whitney U / t pareado + Cohen's d. Veredito de **H1** (`F1(C2,T3) − F1(C1,T3) ≥ 3`) e **H2** (`F1(C3,T2) − F1(C1,T2) ≥ 3`), p < 0.1.

---

## Verification Checklist (antes de fechar a Etapa 2)
- [ ] Modelo carrega e produz logits `[batch, 2]`; ~278M parâmetros.
- [ ] `freeze_layers` cobre C1–C4; `classifier` sempre treinável.
- [ ] `tests/test_model.py` passando para as 4 configs.
- [ ] Tabela de parâmetros treináveis por config **confirmada** (atualizar metodologia §6).
- [ ] Smoke test: gradiente flui só pelos parâmetros não congelados.
- [ ] `src/model.py` versionado e importável pela Etapa 3.
