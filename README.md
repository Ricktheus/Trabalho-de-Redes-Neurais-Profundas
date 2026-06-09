# Resumo do Projeto

Este documento detalha o embasamento teórico, as escolhas de dados e os resultados das Etapas 1 e 2.

## 1. Objetivo e Teoria

A pesquisa investiga o *Zero-Shot Cross-Lingual*: a perda de performance quando modelos operam em idiomas não vistos no *fine-tuning*. O estudo isola dois fatores de degradação:
- **Language Shift:** Inglês para Português.
- **Domain Shift:** "Eletrônicos" para "Beleza".

A literatura (*BERTology*) indica que as camadas inferiores (0-5) dos Transformers processam sintaxe, e as camadas superiores (6-11) processam semântica da tarefa. As hipóteses são:
* **H1:** Congelar as camadas inferiores (*Freeze Lower*) preserva o alinhamento de idiomas e mitiga o *Language Shift*.
* **H2:** Congelar as camadas superiores (*Freeze Upper*) evita a superespecialização e mitiga o *Domain Shift*.

## 2. Engenharia de Dados (Etapa 1 Executada)

A escolha de categorias evitou domínios com sobreposição de vocabulário. O projeto usa **Eletrônicos × Beleza** com regras assimétricas:
* **Inglês (Amazon):** Filtro lexical por palavras-chave (`battery`, `usb`, `skin`, `perfume`).
* **Português (B2W):** Metadado oficial da categoria do produto.

### Auditoria e Rótulos
* O script converteu avaliações de 1 a 5 estrelas em binárias (1-2 = Negativo, 4-5 = Positivo).
* O modelo `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` auditou o filtro lexical em inglês, atingindo precisão superior a 90%.

### Balanceamento Desacoplado
O balanceamento convencional limitaria o volume do treino. O script desacoplou as proporções:
* O conjunto de treino (S1 - EN/Eletrônicos) retém o volume máximo: **~10.700 exemplos**.
* As quatro células de teste (T1 a T4) operam com tamanho fixo: **~2.680 exemplos** cada.
O código garante ausência de interseção de IDs entre as partições.

## 3. Arquitetura do Modelo (Etapa 2 Executada)

A implementação no Colab usa o **`XLM-RoBERTa-base`** (278 milhões de parâmetros) com uma cabeça de classificação (`RobertaClassificationHead`).

A função de *Layer Freezing* controla o número de parâmetros em treinamento:
1. **C1 (Baseline):** Treinamento completo (**278.045.186** parâmetros).
2. **C2 (Freeze Lower / H1):** Congela embeddings e camadas 0 a 5 (**43.119.362** parâmetros em treino).
3. **C3 (Freeze Upper / H2):** Congela camadas 6 a 11 (**235.517.954** parâmetros em treino).
4. **C4 (Piso / Frozen Encoder):** Congela toda a base (**592.130** parâmetros em treino).

O *Smoke Test* confirmou o fluxo de gradientes restrito aos parâmetros selecionados.

## 4. Estrutura de Treino (Etapa 3 Planejada)

A Etapa 3 utiliza o `Trainer` da Hugging Face. Parâmetros definidos:
* **Otimizador:** AdamW (LR: 2e-5, Weight Decay: 0.01)
* **Warmup:** 10% dos passos, precisão mista (`fp16`).
* **Early Stopping:** Paciência de 1 época sobre a `eval_loss` em um split de 10% do treino original.

### Grade de Avaliação
O script gera 12 modelos (4 configurações × 3 *seeds*). A avaliação ocorre em 4 cenários:
* **T1 (Controle):** Inglês / Eletrônicos
* **T2 (Domain Shift):** Inglês / Beleza
* **T3 (Language Shift):** Português / Eletrônicos
* **T4 (Combinado):** Português / Beleza

Os resultados alimentam a planilha `results.csv`.

## 5. Status

1. Metodologia documentada.
2. Dados filtrados, particionados e armazenados (Etapa 1).
3. Mecanismo de congelamento implementado e testado (Etapa 2).
4. Pipeline de treinamento projetado (Plano Etapa 3).

**Ação Pendente:** Executar o script da Etapa 3 no Colab para treinar os modelos, exportar `results.csv` e gerar os gráficos da Etapa 4.

## 6. Estrutura do Repositório

O repositório organiza o código e a documentação nos seguintes diretórios:

* **`docs/`**: Documentação principal do projeto. Contém a metodologia acadêmica, o guia do projeto, o registro de decisões/dificuldades e os planos técnicos das etapas.
* **`notebooks/`**: Arquivos executáveis do Jupyter/Colab (`.ipynb`). Inclui as execuções reais da Etapa 1 (processamento de dados) e Etapa 2 (arquitetura do modelo).
* **`src/`**: Códigos-fonte em Python (`.py`), englobando o pipeline de dados e a classe do modelo.
* **`guias/`**: Materiais de apoio para a equipe, incluindo guias de estudo para a defesa e anotações específicas sobre o congelamento de Transformers.
* **`tests/`**: Scripts de teste unitário para validar o funcionamento do modelo e das funções auxiliares.
