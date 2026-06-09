# PLAN-congelamento-transformers

Plano de execução detalhado do projeto acadêmico de Redes Neurais Profundas: **Análise Arquitetural do Congelamento de Camadas em Transformers Multilíngues**.

## Overview
O objetivo deste projeto é avaliar a eficácia do congelamento seletivo de pesos (*Layer Freezing*) em arquiteturas Transformer (XLM-RoBERTa-Base) na mitigação de *Domain Shift* e *Language Shift* em tarefas de classificação de sentimentos *Zero-Shot Cross-Lingual*. O modelo será treinado em avaliações de produtos em inglês e testado em cenários de e-commerce e logística em português brasileiro.

## Project Type
**BACKEND / DATA SCIENCE** (Processamento de Linguagem Natural / Aprendizado de Máquina)

## Seeds padronizadas
`{42, 123, 2024}` — usar em todos os experimentos.

## Success Criteria
*   Pipeline de processamento e filtragem de datasets implementado com validação manual de 100 amostras (precisão >= 80%).
*   Pipeline de modelagem capaz de carregar o XLM-RoBERTa-Base, congelar camadas dinamicamente e executar o treinamento.
*   Matriz de experimentos executada para as 4 configurações (C1, C2, C3, C4) utilizando 3 seeds diferentes cada (totalizando 12 treinos).
*   Geração de tabelas de comparação de F1-Macro e gráficos das curvas de perda (*Loss*) de validação e treino.
*   Seções de Introdução, Metodologia e Discussão do artigo acadêmico redigidas.

## Tech Stack
*   **Linguagem de Programação:** Python 3.10+
*   **Framework de Deep Learning:** PyTorch
*   **Biblioteca de NLP:** Hugging Face `transformers` & `datasets`
*   **Métricas e Análises:** Scikit-Learn, Pandas, Matplotlib, Seaborn
*   **Ambiente de Execução:** Google Colab (com aceleração de hardware por GPU T4/A100)
*   **IDE de Desenvolvimento:** VS Code / Cursor (para prototipagem dos scripts)

## File Structure
```
/Trabalho de redes neurais profundas/
├── docs/
│   └── PLAN-congelamento-transformers.md (Este arquivo)
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py    # Download, filtragem e validação dos dados
│   ├── model.py            # Definição do classificador e lógica de congelamento
│   ├── train.py            # Loop de treinamento principal e controle de seeds
│   └── utils.py            # Funções utilitárias e plotagem de gráficos
├── notebooks/
│   └── Experimentos_Colab.ipynb  # Notebook para execução na GPU do Colab
├── tests/
│   ├── test_data.py        # Testes unitários para o pipeline de dados
│   └── test_model.py       # Testes unitários para verificação do congelamento
├── requirements.txt        # Dependências do Python
└── README.md               # Guia de reprodução rápida do repositório
```

---

## Task Breakdown

### Etapa 1: Preparação de Dados e Auditoria Semântica
*   **Task 1.1: Download e Exploração Inicial**
    *   **Priority:** P0
    *   **Dependencies:** Nenhuma
    *   **INPUT:** Acesso aos datasets Amazon Customer Reviews (Inglês) e Olist E-commerce / B2W (Português).
    *   **OUTPUT:** Script `src/data_pipeline.py` estruturado com o download automático das bases de dados.
    *   **VERIFY:** Executar a função de carregamento básico e verificar se o número total de linhas importadas confere com as originais das APIs.
*   **Task 1.2: Filtragem por Palavras-Chave Globais**
    *   **Priority:** P0
    *   **Dependencies:** Task 1.1
    *   **INPUT:** Listas de palavras-chave para o domínio "Produto" e "Logística/Entrega" em inglês e português (ver `Metodologia_Rascunho.md` seção 5.2).
    *   **OUTPUT:** Datasets particionados em: Produto/EN (S1), Logística/EN (S2), Produto/PT (S3), Logística/PT (S4).
    *   **VERIFY:** Conferir se as partições geradas não possuem interseção direta de IDs de avaliações.
*   **Task 1.3: Auditoria Manual de Qualidade (100 Amostras)**
    *   **Priority:** P1
    *   **Dependencies:** Task 1.2
    *   **INPUT:** 100 amostras aleatórias de cada um dos quatro subconjuntos gerados.
    *   **OUTPUT:** Tabela com taxa de precisão de filtragem por subconjunto.
    *   **VERIFY:** Verificar se o critério de aceitação de pelo menos 80% de precisão manual foi atendido. Caso contrário, refinar os termos e reexecutar.

### Etapa 2: Implementação do Modelo e Algoritmo de Congelamento
*   **Task 2.1: Definição da Arquitetura do Classificador**
    *   **Priority:** P0
    *   **Dependencies:** Nenhuma
    *   **INPUT:** Importação do `xlm-roberta-base` usando a biblioteca Hugging Face.
    *   **OUTPUT:** Script `src/model.py` com o modelo customizado usando `XLMRobertaForSequenceClassification` ou estendendo `nn.Module`.
    *   **VERIFY:** Passar um tensor dummy pelo modelo e confirmar que a saída tem dimensão `[batch_size, 2]`.
*   **Task 2.2: Lógica de Congelamento de Camadas (C1 a C4)**
    *   **Priority:** P0
    *   **Dependencies:** Task 2.1
    *   **INPUT:** Parâmetro de configuração do congelamento (`C1`, `C2`, `C3` ou `C4`).
    *   **OUTPUT:** Função `freeze_layers(model, config)` que desativa `requires_grad` nos tensores corretos.
    *   **VERIFY:** Teste unitário em `tests/test_model.py` que varre os parâmetros e valida se `requires_grad` está correto para cada caso. Confirmar contagem de parâmetros treináveis por config.

### Etapa 3: Pipeline de Treinamento Multi-Seed
*   **Task 3.1: Loop de Treinamento com AdamW e Cross-Entropy**
    *   **Priority:** P0
    *   **Dependencies:** Task 2.2
    *   **INPUT:** Dataloaders de treino (S1-train) e validação (S1-val).
    *   **OUTPUT:** Script `src/train.py` com o loop de épocas, cálculo de loss, passo do otimizador, early stopping e log de métricas.
    *   **VERIFY:** Rodar um mini-treinamento com 2 batches para garantir que o gradiente flui e a loss diminui.
*   **Task 3.2: Controle de Seeds e Salvamento de Resultados**
    *   **Priority:** P1
    *   **Dependencies:** Task 3.1
    *   **INPUT:** Seeds `{42, 123, 2024}`.
    *   **OUTPUT:** Lógica de fixação de seeds globais (`torch.manual_seed`, `np.random.seed`, `transformers.set_seed`) e salvamento de métricas por run em JSON.
    *   **VERIFY:** Confirmar que duas execuções com a mesma seed produzem loss e F1 matematicamente idênticos no primeiro batch.

### Etapa 4: Avaliação Zero-Shot e Plotagem dos Gráficos
*   **Task 4.1: Avaliação nas 4 Células**
    *   **Priority:** P0
    *   **Dependencies:** Task 3.2
    *   **INPUT:** Modelos treinados e dataloaders S1-val, S2, S3, S4.
    *   **OUTPUT:** F1-Macro para cada combinação configuração × seed × cenário. Arquivo `results.csv` consolidado.
    *   **VERIFY:** Confirmar que todos os 12 experimentos (4 configs × 3 seeds) têm métricas salvas. Verificar que F1 em S1-val > F1 em S2/S3/S4 (sanity check).
*   **Task 4.2: Plotagem e Visualização**
    *   **Priority:** P1
    *   **Dependencies:** Task 4.1
    *   **INPUT:** `results.csv` com métricas dos 12 treinos.
    *   **OUTPUT:** Script `src/utils.py` com funções para: curvas de loss (treino vs val), heatmap configuração × cenário, barplot de Δ-shift com error bars.
    *   **VERIFY:** Confirmar geração correta das figuras com eixos nomeados e legíveis.

---

## Verification Checklist (antes da entrega final)

### Controles Acadêmicos
- [ ] Paper com título focado na contribuição central (Regra 1 de Mensh & Kording).
- [ ] Linguagem evita acrônimos sem explicação (Regra 2).
- [ ] Introdução e metodologia seguem o esquema Contexto-Conteúdo-Conclusão (Regra 3).
- [ ] Hipóteses H1 e H2 foram ou não suportadas — conclusão clara.

### Controles de Código
- [ ] Testes unitários em `tests/` passando (dados e modelagem).
- [ ] `results.csv` com 48 linhas (4 configs × 3 seeds × 4 cenários).
- [ ] Nenhuma variável de credencial exposta no código.
- [ ] `requirements.txt` com versões pinadas.
