# PLAN-Etapa4 — Análise Estatística e Visualização de Dados

Plano de execução da **Etapa 4** do projeto "Análise Arquitetural do Congelamento de Camadas em Transformers Multilíngues". Dá continuidade ao `PLAN-Etapa3.md` (Etapa 3 concluída) e detalha como extrair, processar e plotar as métricas para a defesa acadêmica.

## Overview
A Etapa 4 consome os artefatos gerados na Etapa 3 (`results.csv` e arquivos JSON na pasta `runs/`) para comprovar ou refutar as Hipóteses 1 e 2 do trabalho. O script fará a sumarização das médias de F1-macro através das três *seeds*, aplicará o teste não-paramétrico de Mann-Whitney U (ou t-pareado) para checagem de p-valor, e desenhará os gráficos exigidos na Metodologia.

## Project Type
**DATA SCIENCE / VISUALIZATION** (Análise exploratória, Estatística Inferencial e Plotagem)

## Success Criteria
* Tabela consolidada com a Média e o Desvio-Padrão do F1-macro para as 16 combinações de Configurações (C1–C4) × Cenários de Teste (T1–T4).
* Cálculo matemático do $\Delta$-shift isolando Domain Shift (T1 - T2) e Language Shift (T1 - T3) para cada arquitetura de congelamento.
* Obtenção do p-valor comparando C2 e C3 contra a baseline (C1) para atestar a significância estatística das melhoras observadas no arquivo de resultados.
* Plotagem de 3 tipos de gráficos essenciais para o relatório:
  1. Matriz de Calor (Heatmap) cruzando as 4 Configurações com os 4 Testes.
  2. Gráfico de Barras (Barplot) ilustrando as quedas (Deltas) de F1-macro.
  3. Curvas de Loss (Treino e Validação) com *overlay* para identificar comportamentos de *overfitting* ou convergência prematura.

## Tech Stack
* **Linguagem:** Python
* **Bibliotecas:** `pandas`, `numpy`, `matplotlib`, `seaborn` (para visualização esteticamente agradável), `scipy` (para `mannwhitneyu` ou `ttest_ind`).
* **Ambiente:** Jupyter Notebook (local ou Colab). Não há mais a necessidade de aceleração em GPU.

## File Structure (acréscimos desta etapa)
```
/TrabalhoRNP/
├── docs/
│   └── PLAN-Etapa4.md      # Este plano
├── notebooks/
│   └── Trabalho_RNP_Colab_Etapa4.ipynb   # Notebook de Análise e Gráficos
├── resultados/             # Criado na Etapa 3 (Input da Etapa 4)
│   ├── results.csv
│   └── runs/
│       └── run_C*.json
```

---

## Passo a Passo (Roadmap)

### Fase 4.0: Setup e Ingestão de Dados
- **Ações:** Importar as bibliotecas `pandas` e `seaborn`. Configurar um tema *dark* ou *whitegrid* (moderno e limpo) e carregar o arquivo raiz `results.csv`.
- **Verify:** O `DataFrame` deve conter 48 linhas e 7 colunas (incluindo `f1_macro` e `accuracy`).

### Fase 4.1: Tabela de Agregação de Médias e Desvio Padrão
- **Ações:** Fazer um `groupby(["config", "teste"])` calculando `.agg(["mean", "std"])` na coluna `f1_macro`.
- **Output:** Tabela (pivot table) com 4 linhas (C1-C4) e 4 colunas (T1-T4) mostrando a média cruzando as 3 seeds. Este formato é o que vai direto para o paper acadêmico.

### Fase 4.2: Testes Estatísticos (p-valor)
- **Ações:** Isolar os arrays das 3 seeds do C1 (Baseline) e compará-los através do `scipy.stats.ttest_ind`:
  - Para o *Language Shift* (T3): Comparar `C2[f1_macro]` vs `C1[f1_macro]`.
  - Para o *Domain Shift* (T2): Comparar `C3[f1_macro]` vs `C1[f1_macro]` e `C2[f1_macro]` vs `C1[f1_macro]`.
- **Output:** Relatório impresso de significância indicando se o p-valor < 0.1, de acordo com as regras estabelecidas na `Metodologia_Rascunho.md`.

### Fase 4.3: Visualizações Estáticas (Gráficos)
- **Heatmap Geral:** Um mapa de calor de fácil leitura cruzando C1-C4 no eixo Y e T1-T4 no eixo X.
- **Gráfico de Deltas:** Um Barplot lado a lado. Eixo X são as configurações. Barras duplas: "Queda de Língua (T1-T3)" e "Queda de Domínio (T1-T2)". Evidencia quem segurou mais a perda de performance.
- **Gráfico de Aprendizado (Opcional/Debug):** Extrair do `runs_info.json` a `log_history` e desenhar as linhas de Train Loss e Eval Loss ao longo das épocas de treinamento. Isso mostrará por que o Early Stopping cortou a execução de certos testes na época 2.
