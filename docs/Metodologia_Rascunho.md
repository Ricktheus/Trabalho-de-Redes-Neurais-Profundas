# Metodologia

**Projeto:** Análise Arquitetural do Congelamento de Camadas na Mitigação de *Domain Shift* e *Language Shift* em Transformers Multilíngues

> **Como usar este arquivo:** este é um rascunho enxuto, em estilo de paper, das seções que vocês precisam entregar. Ajustar o texto onde estiver `[CONFIRMAR]` e onde tiverem decisões diferentes. Cole no Word/Docs para formatar.
>
> **Atualização (pós-Fase 2):** os domínios passaram de *Produto × Logística* para **Eletrônicos × Beleza**; a filtragem PT passou de palavra-chave para **categoria do produto** (`site_category_lv1`); e o balanceamento passou a ser **desacoplado** (treino independente das células de teste). As seções 1, 2, 5 e 8 já refletem essas decisões.

> **Estado atual da entrega (pós-Etapa 4 — análise concluída).** O que já está **definido e executado**: (i) pergunta de pesquisa, hipóteses e critérios de suporte (§1–§2); (ii) arquitetura, head, esquema de rótulos e as 4 configurações de congelamento C1–C4 (§3, §4, §6); (iii) pipeline de dados (filtro híbrido EN-keyword / PT-categoria, balanceamento desacoplado), com filtro EN auditado em 90–94% (§5); (iv) protocolo de treino dos 12 modelos e a matriz de 48 avaliações (§7–§8); (v) análise estatística completa — agregação, Δ-shift, testes (Welch *t*, Mann-Whitney U, Cohen's *d*) e veredito das hipóteses (§9, ver resultados em [`RESULTADOS-Etapa4.md`](RESULTADOS-Etapa4.md)).
>
> **A limitação que mais impacta a leitura dos resultados nesta entrega** é a **assimetria de qualidade entre os conjuntos de teste** (EN por palavra-chave, ruidoso; PT por categoria, *ground-truth*) somada ao fato de o *Language Shift* ter sido medido em **um único par de línguas próximas** (EN↔PT). Como detalhado em §11, essa assimetria **confunde** parte do "ganho zero-shot" observado em português e **limita o escopo** da conclusão sobre *Language Shift*. As condições que refutariam nossas conclusões estão explicitadas em §2.1 (a principal: **testar em outra língua, tipologicamente distante**).

---

## 1. Pergunta de Pesquisa

Em tarefas de classificação binária de sentimento em configuração *zero-shot cross-lingual*, **qual estratégia de congelamento seletivo de pesos** em um Transformer multilíngue (XLM-RoBERTa) mitiga **mais eficazmente** a degradação de F1-macro causada por:

- **Language Shift** (treino em inglês, teste em português), versus
- **Domain Shift** (treino em avaliações de **eletrônicos**, teste em avaliações de **beleza/perfumaria**)?

---

## 2. Hipóteses

Apoiados pela literatura de interpretabilidade de Transformers (BERTology), partimos das seguintes hipóteses falsificáveis:

- **H1 (Language Shift):** Congelar as camadas iniciais do encoder (embeddings + camadas 0 a 5), que codificam representações léxico-sintáticas multilíngues, reduz em pelo menos **3 pontos de F1-macro** a queda observada entre os subconjuntos Eletrônicos/EN (treino-validação) e Eletrônicos/PT (teste), comparado ao fine-tuning completo.

- **H2 (Domain Shift):** Congelar as camadas finais do encoder (camadas 6 a 11), que codificam abstrações semânticas alinhadas à tarefa, reduz em pelo menos **3 pontos de F1-macro** a queda observada entre Eletrônicos/EN e Beleza/EN, comparado ao fine-tuning completo.

- **H0 (Hipótese Nula):** As estratégias de congelamento têm efeito estatisticamente equivalente nas duas degradações.

### 2.1 Condições de Refutação (Falseabilidade)

Uma hipótese só é científica se for possível dizer **o que a derrubaria**. Declaramos antecipadamente os observáveis que refutam cada afirmação do trabalho:

| Afirmação | O que a **refutaria** |
|-----------|----------------------|
| **H1** — congelar a base mitiga *Language Shift* | Δ `F1(C2,T3) − F1(C1,T3) < 3 pts` ou p ≥ 0.1 (refutada). *Pré-requisito:* só faz sentido testar se existir queda de língua na baseline (`F1(T1) > F1(T3)`). |
| **H2** — congelar o topo mitiga *Domain Shift* | Δ `F1(C3,T2) − F1(C1,T2) < 3 pts` ou p ≥ 0.1 (refutada). |
| **Tese emergente** — "o XLM-R é robusto a *Language Shift* / congelar a base regulariza contra *Domain Shift*" | **O teste decisivo é replicar em outra língua, tipologicamente distante e/ou de outro sistema de escrita** (ex.: árabe, mandarim, hindi, suaíli). Se a "ausência de *Language Shift*" observada em EN→PT **não se repetir** numa língua distante, a conclusão se revela um artefato da proximidade EN/PT (ambas indo-europeias, escrita latina, alta cobertura no pré-treino), e não uma propriedade geral do modelo. |

> **Implicação metodológica.** Como a generalização da tese sobre *Language Shift* exige um segundo par de línguas, **nesta entrega ela é tratada como conclusão de escopo restrito** (válida para EN↔PT), não como lei geral. Adicionar uma língua distante é o próximo passo experimental prioritário (§11, §12).

---

## 3. Arquitetura

### 3.1 Modelo base

Adotamos o **XLM-RoBERTa-base** (Conneau et al., 2020), modelo Transformer pré-treinado em CommonCrawl multilíngue (~2,5 TB, 100 idiomas) via Masked Language Modeling. Composição:

- 12 camadas Transformer encoder
- 768 dimensões de hidden state
- 12 cabeças de atenção por camada
- ~278M de parâmetros totais

**Justificativa:** o modelo base é o padrão da literatura para estudos de transferência cross-lingual, balanceia capacidade representacional e custo computacional, e tem profundidade de 12 camadas — facilita a partição em "camadas iniciais" (0–5) e "camadas finais" (6–11) usadas nos experimentos.

### 3.2 Classification Head

Adotamos a *classification head* **padrão do Hugging Face** para `XLMRobertaForSequenceClassification` (`RobertaClassificationHead`), aplicada sobre o vetor do token `<s>` (equivalente ao `[CLS]` no BERT) da última camada:

- Dropout (p = 0.1)
- Camada densa Linear 768 → 768, seguida de ativação `tanh`
- Dropout (p = 0.1)
- Projeção de saída Linear 768 → 2 (logits para as classes Positivo/Negativo)

Optamos pela head canônica do HF — em vez de uma head mínima (apenas Dropout + Linear) — por ser a configuração de referência da literatura e mais robusta. A *classification head* é sempre treinável (nunca congelada), pois é inicializada do zero.

---

## 4. Tarefa e Esquema de Rótulos

A tarefa é **classificação binária de sentimento** (Positivo vs Negativo).

Os datasets adotados (Amazon Reviews em inglês e B2W Reviews em português) utilizam escala de 1 a 5 estrelas. Aplicamos o seguinte mapeamento:

- **Negativo:** notas 1 e 2
- **Positivo:** notas 4 e 5
- **Descartado:** nota 3 (sentimento ambíguo)

A escolha pelo esquema binário (em vez de ternário) reduz ambiguidade nas classes e está alinhada com a maioria dos benchmarks de transferência cross-lingual.

---

## 5. Datasets e Pipeline de Filtragem

### 5.1 Fontes

| Origem | Idioma | Atribuição de domínio | Tamanho bruto aproximado |
|--------|--------|-----------------------|--------------------------|
| Amazon (`amazon_polarity`, HuggingFace) | EN | palavra-chave (texto) | ~3,6M (lido via streaming) |
| B2W Reviews (americanas-tech) | PT | categoria (`site_category_lv1`) | ~132k |

> O **Olist** foi descartado: seu dataset de avaliações não traz a categoria do produto no nível da avaliação (exigiria *join* multi-tabela), e o B2W sozinho, filtrado por categoria, já excede as metas de tamanho.

### 5.2 Filtragem por domínio (abordagem híbrida)

A atribuição de domínio usa **dois métodos**, escolhidos pela disponibilidade de metadado em cada base.

**Inglês (Amazon) — filtro lexical.** A base não traz a categoria do produto, então aplicamos um filtro por palavras-chave sobre o texto. A avaliação pertence a um domínio se contém **pelo menos uma palavra-chave dele E nenhuma do outro** (descarta ambíguos/neutros):

- **Eletrônicos (EN):** `battery`, `usb`, `charger`, `charging`, `wifi`, `bluetooth`, `smartphone`, `laptop`, `tablet`, `touchscreen`, `headphone(s)`, `earbuds`, `smartwatch`, `phone(s)`, `router`, `hdmi`
- **Beleza (EN):** `skin`, `scent`, `fragrance`, `perfume`, `cream`, `lotion`, `shampoo`, `conditioner`, `moisturizer`, `makeup`, `cosmetic`, `serum`, `sunscreen`, `soap`, `wrinkle`

As listas são deliberadamente **enxutas e inequívocas**: termos ambíguos (`screen`, `camera`, `display`, `speaker`, `keyboard`, `processor`) foram removidos por vazarem para avaliações de filme/música/cozinha, derrubando a precisão do filtro.

**Português (B2W) — categoria do produto.** A base traz o metadado `site_category_lv1`, então usamos **a categoria diretamente** (ground-truth, precisão ~100%), em vez de palavras-chave:

- **Eletrônicos (PT):** `Celulares e Smartphones` + `Informática e Acessórios` + `TV e Home Theater` (cluster de eletrônica de consumo, pareado com a definição ampla do EN)
- **Beleza (PT):** `Beleza e Perfumaria`

Essa assimetria é proposital: aproveita o metadado onde ele existe (PT) e mantém o método lexical onde não existe (EN). Definir domínio por **categoria de produto** é a operacionalização canônica de *domain shift* em análise de sentimento (Blitzer et al., 2007), o que reforça a fundamentação.

### 5.3 Validação do filtro

A validação concentra-se no **lado EN**, único que usa filtro sujeito a erro. Amostramos 100 avaliações de cada subconjunto EN (S1, S2) e as classificamos com um **anotador zero-shot multilíngue** independente (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), comparando com o domínio do filtro. Usamos rótulos curtos e simétricos (`"an electronics product"` × `"a beauty product"`) para evitar viés de comprimento. Critério de aceitação: precisão ≥ 80% por subconjunto.

- **Resultado (Fase 2):** S1 (EN/Eletrônicos) = **90%**, S2 (EN/Beleza) = **94%** — ambos aprovados.

O **lado PT (S3, S4) deriva da categoria do produto** e é, portanto, ground-truth por construção (precisão 100%); não é auditado por LLM, pois validar um metadado confiável com um modelo mais fraco apenas injetaria ruído. O uso de LLM como anotador segue prática consolidada (Zhu et al., 2023).

### 5.4 Subconjuntos finais e balanceamento

| ID | Origem | Idioma | Domínio | Função |
|----|--------|--------|---------|--------|
| S1 | Amazon (keyword) | EN | Eletrônicos | Treino (80%) + Validação (20%) |
| S2 | Amazon (keyword) | EN | Beleza | Teste — isola Domain Shift |
| S3 | B2W (categoria) | PT | Eletrônicos | Teste — isola Language Shift |
| S4 | B2W (categoria) | PT | Beleza | Teste — impacto combinado |

**Balanceamento desacoplado.** O tamanho do treino é separado do tamanho das células de teste, para que um compartimento de teste escasso não rebaixe o treino:

1. **Treino/validação:** o S1 é balanceado no seu próprio máximo (`N_s1 = min(neg, pos)` de S1) e dividido 80/20 estratificado. O treino usa todo o pool disponível.
2. **Matriz de teste:** as 4 células (S1-val, S2, S3, S4) são balanceadas a um N comum (`N_test = min(neg, pos)` entre as 4 células de teste), garantindo comparação justa.

Isso corrige a regra inicial (`N = min entre os 4 subconjuntos` *antes* do split), que acoplava o tamanho do treino ao menor compartimento de teste.

**Tamanhos realizados (Fase 2):** treino ≈ **10.700** exemplos; cada célula de teste ≈ **2.680** (1.340 por classe), todas balanceadas 50/50.

---

## 6. Configurações Experimentais de Congelamento

Avaliaremos **4 configurações** de congelamento durante o fine-tuning. Em todas, a classification head permanece treinável.

| ID | Nome | Módulos congelados | Parâmetros treináveis |
|----|------|---------------------|------------------------|
| **C1** | Full Fine-Tuning (baseline alto) | — | 278.045.186 (100%) |
| **C2** | Freeze Lower (testa H1) | embeddings + encoder.layer.0 a 5 | 43.119.362 (15,5%) |
| **C3** | Freeze Upper (testa H2) | encoder.layer.6 a 11 | 235.517.954 (84,7%) |
| **C4** | Frozen Encoder (baseline baixo) | embeddings + encoder.layer.0 a 11 | 592.130 (0,21%) — apenas a head |

A implementação é via `requires_grad = False` aplicado por *named_parameters*, com verificação por contagem de parâmetros treináveis antes do treino (`src/model.py::freeze_layers`, coberta por `tests/test_model.py`).

> **Observação (confirmada na implementação — Etapa 2).** A matriz de *word-embeddings* do XLM-R é enorme — 250.002 × 768 ≈ 192M parâmetros, ~69% do modelo. Logo, congelar os embeddings (C2 e C4) já imobiliza a maioria dos pesos: C2 treina ~15,5% do modelo (camadas 6–11 + head) e C4 apenas 0,21% (somente a *classification head*). Os 592k da head — e não ~1,5k — refletem a head **padrão do HF**, que inclui uma camada densa 768→768 (`dense`) antes da projeção 768→2 (`out_proj`). As estimativas iniciais (C2 ~85M, C3 ~193M, C4 ~1,5k) foram **substituídas** pelos valores reais acima.

---

## 7. Protocolo de Treino

Hiperparâmetros fixos em todas as configurações (para garantir comparação justa):

| Parâmetro | Valor |
|-----------|-------|
| Otimizador | AdamW (Loshchilov & Hutter, 2019) |
| Learning rate | 2e-5 |
| Batch size | 16 (ajuste fino conforme memória GPU) |
| Max sequence length | 128 tokens |
| Weight decay | 0.01 |
| Warmup ratio | 10% dos passos totais |
| Épocas máximas | 3 |
| Early stopping | paciência 1, monitora *eval_loss* |
| Função de custo | Cross-Entropy |
| Precisão | mixed precision (fp16) para reduzir uso de memória |

**Seeds:** cada configuração é treinada com 3 seeds independentes: `{42, 123, 2024}`. Total: 4 configs × 3 seeds = **12 fine-tunings**.

---

## 8. Matriz de Avaliação

Cada modelo treinado é avaliado nas 4 células:

| Cenário de Teste | Idioma | Domínio | Tipo de Shift |
|-------------------|--------|---------|---------------|
| T1 (S1-val) | EN | Eletrônicos | Nenhum (controle interno) |
| T2 (S2) | EN | Beleza | Apenas Domain Shift |
| T3 (S3) | PT | Eletrônicos | Apenas Language Shift |
| T4 (S4) | PT | Beleza | Domain + Language combinados |

Total de avaliações: 4 configs × 3 seeds × 4 cenários = **48 medições de F1-macro**.

---

## 9. Métricas e Análise Estatística

### 9.1 Métricas

- **Primária:** F1-macro
- **Complementares:** F1 por classe, accuracy, confusion matrix, loss final (treino e validação)

### 9.2 Análise

- Média e desvio-padrão de F1-macro sobre as 3 seeds, por configuração × cenário.
- **Delta de Shift:** $\Delta(\text{shift}) = F1(T_1) - F1(T_x)$, isolando o impacto de cada tipo de shift por configuração.
- **Teste de significância:** teste t pareado (ou Mann-Whitney U dado n=3) entre pares de configurações no mesmo cenário. Reportar p-valor e Cohen's d para tamanho de efeito.
- **Análise gráfica:**
  - Curvas de loss (treino vs validação) por configuração — identificar overfitting e divergência.
  - Heatmap configuração × cenário, com cor proporcional a F1-macro.
  - Barplot agrupado de Δ-shift por configuração.

### 9.3 Critério de suporte às hipóteses

- **H1 suportada** se: `F1(C2, T3) − F1(C1, T3) ≥ 3 pontos`, com p < 0.1.
- **H2 suportada** se: `F1(C3, T2) − F1(C1, T2) ≥ 3 pontos`, com p < 0.1.

Threshold de significância relaxado (0.1 em vez de 0.05) considerando o n=3.

---

## 10. Ambiente Computacional

- **IDE:** VS Code (prototipação local) + Google Colab (treinos com GPU).
- **Aceleração:** NVIDIA T4 (Colab gratuito) ou A100 (Colab Pro, se disponível).
- **Frameworks:** PyTorch, HuggingFace Transformers, Datasets, Evaluate, Accelerate.
- **Reprodutibilidade:** seeds fixadas em `torch`, `numpy`, `random`, `transformers.set_seed`. Versão dos pacotes registrada em `requirements.txt`.
- **Versionamento:** repositório Git (privado) com notebooks numerados por fase.

---

## 11. Limitações Conhecidas

Reconhecemos as seguintes limitações do desenho experimental. As três primeiras são as que **mais impactam a interpretação dos resultados** desta entrega e estão amarradas, em §11.bis, às conclusões que elas restringem.

1. **Filtro lexical EN e o problema das palavras ambíguas (a "trava" principal).** A base Amazon (`amazon_polarity`) **não traz a categoria do produto**, então o domínio no lado inglês é atribuído por palavras-chave sobre o texto — uma *proxy* imperfeita. O obstáculo de fundo é a **polissemia/homonímia**: a mesma palavra de superfície assume sentidos de domínios diferentes (ex.: *tablet* = dispositivo eletrônico **ou** comprimido/suplemento; *foundation* = base de maquiagem **ou** fundação; *charge* = carregar bateria **ou** cobrança), e o filtro não distingue o sentido — só a forma. Já mitigamos isso removendo termos comprovadamente ambíguos (`screen`, `camera`, `display`, `speaker`, `keyboard`, `processor`) e exigindo "tem keyword do domínio E nenhuma do outro", mas resta ruído residual: a auditoria mediu **90–94% de precisão**, ou seja, **6–10% das avaliações EN podem estar no domínio errado**. *Impacto direto nos resultados:* como **T1 (S1-val/EN) e T2 (S2/EN) saem do mesmo filtro ruidoso**, parte das avaliações de eletrônicos e de beleza se confundem, tornando os dois conjuntos **mais parecidos do que os domínios reais**. Isso **subestima** a magnitude verdadeira do *Domain Shift*: o Δ medido (+3,3 pts na baseline) é um **piso**, não o valor real. Aprofundar a desambiguação (ex.: classificador de domínio em vez de keyword, ou um dataset EN com categoria) fica como trabalho da próxima semana.

2. **Confound entre qualidade do conjunto e efeito de língua.** Decorre direto de (1): o lado **PT é *ground-truth*** (categoria oficial do B2W, ~100% de precisão) enquanto o lado **EN é ruidoso** (keyword, 90–94%). Logo, a baseline T1 (EN) carrega ruído de rótulo de domínio que T3 (PT) **não** tem. O "ganho zero-shot" que observamos (`F1(T3) > F1(T1)`, isto é, o modelo parece **melhor** em português do que em inglês) **não pode ser atribuído puramente à robustez cross-lingual** — ele está **confundido** com o fato de o conjunto português ser simplesmente mais limpo de rotular. É plausível que parte (ou todo) do ganho seja artefato de qualidade de dados, não transferência de língua. Essa é a limitação mais séria para a leitura de H1.

3. **Escopo de um único par de línguas (próximas).** O *Language Shift* foi medido **só** em EN→PT — duas línguas indo-europeias, de escrita latina, com muitos cognatos e **ambas de alta cobertura** no pré-treino do XLM-R. A conclusão "não há *Language Shift*" é, portanto, **de escopo restrito a esse par fácil**. Línguas tipologicamente distantes e/ou de outro sistema de escrita (árabe, mandarim, hindi, suaíli) provavelmente exibiriam degradação real — esse é exatamente o teste de refutação previsto em §2.1.

4. **Número de seeds limitado (n=3)** por restrição computacional. Reduz a robustez estatística (o Mann-Whitney U satura em p=0.10 com n=3 vs 3); reportamos Cohen's *d* (efeitos grandes) para compensar, mas os intervalos de confiança são largos.

5. **XLM-R-base, não XLM-R-large.** A escolha é por custo. O resultado pode não generalizar para modelos maiores (a capacidade extra do *large* poderia alterar o equilíbrio entre as camadas congeladas e treináveis).

6. **Tamanho das células de teste.** As 4 células são balanceadas ao menor compartimento — a validação interna S1-val (~1.340 por classe, ≈ 2.680 por célula). Suficiente para F1-macro sobre 3 seeds, mas implica intervalos de confiança um pouco mais largos.

7. **Mapeamento estrelas → sentimento binário** é uma simplificação que descarta neutralidade (nota 3).

8. **Granularidade do freezing.** Testamos uma partição binária (0–5 vs 6–11). Não exploramos partições mais finas (ex.: congelar apenas embeddings, ou apenas 0–2), que poderiam localizar melhor onde mora o efeito regularizador observado em C2.

9. **Hiperparâmetros não ajustados por configuração.** Mantemos LR fixo em 2e-5 em todas as configs para isolar a variável "freezing", mas a literatura sugere que freezing parcial pode beneficiar-se de LR mais alto — C4 (frozen encoder), em especial, pode estar subtreinado.

### 11.bis Conclusões Parciais (conscientes das limitações)

O que **podemos** afirmar com os dados atuais, **cada conclusão amarrada ao seu escopo e à limitação que a restringe**:

- **H1 refutada — mas por escopo, não por prova geral.** *Para o par EN→PT*, não há *Language Shift* a mitigar (a baseline já não cai em PT) e C2 não muda isso (Δ ≈ +0,03 pt, p ≈ 0,93). **Porém**, essa leitura está contaminada pelo confound da limitação 2 (PT mais limpo que EN) e restrita ao par fácil da limitação 3. **Conclusão honesta:** "no par EN↔PT, e dada a assimetria de qualidade dos conjuntos, não detectamos *Language Shift*" — **não** "o XLM-R é imune a *Language Shift*".
- **H2 refutada e invertida — robusta dentro do escopo de domínio testado.** Congelar o topo (C3) **piorou** o *Domain Shift* (Δ ≈ −2,97 pts, p ≈ 0,074). Esta conclusão depende menos do confound de língua (T1 e T2 são ambos EN), mas seu **escopo é o par de domínios Eletrônicos→Beleza**; outros pares de domínio podem se comportar diferente.
- **Achado emergente (C2 regulariza contra *Domain Shift*) — promissor, com magnitude subestimada.** C2 vs C1 em T2: Δ ≈ +1,19 pt, p ≈ 0,086, efeito grande. Como o *Domain Shift* real é maior que o medido (limitação 1), o efeito protetor de C2 também pode estar subestimado. Conclusão de escopo restrito a Eletrônicos→Beleza, EN.
- **C4 (frozen encoder) é piso inviável** — mas ver limitação 9 (pode estar subtreinado pelo LR fixo).

---

## 12. Cronograma de Execução

| Semana | Entregável |
|--------|------------|
| 2 (atual) | **Metodologia (este documento)** |
| 3 | Datasets adquiridos, EDA completa, filtros validados |
| 4 | Pipeline de treino implementado e baseline (C1) rodando |
| 5 | Baseline avaliado nas 4 células; sanity check |
| 6 | Configurações C2, C3, C4 treinadas (3 seeds cada) |
| 7 | Buffer / re-runs / debug |
| 8 | Análise de resultados, gráficos, testes estatísticos |
| 9 | Draft do paper — outline + primeiro rascunho |
| 10 | Revisão do paper |
| 11 | Slides de apresentação + ensaio |
| 12 | Apresentação final |

**Próximos passos prioritários (decorrentes das limitações §11), em ordem:**

1. **Testar em uma língua tipologicamente distante** (ex.: árabe, mandarim ou hindi) — é o teste de refutação de §2.1 e o que separa "robustez real a *Language Shift*" de "artefato do par fácil EN↔PT". **Maior prioridade científica.**
2. **Desambiguar o domínio no lado EN** — trocar o filtro por palavra-chave por um classificador de domínio (ou adotar um dataset EN com categoria), atacando a polissemia da limitação 1 e removendo o confound da limitação 2. *(pode ficar para a próxima semana)*
3. **Partições de freezing mais finas** (só embeddings, só 0–2, etc.) para localizar onde mora o efeito regularizador de C2.

---

## Referências preliminares

- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. ACL.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Rogers, A., Kovaleva, O., Rumshisky, A. (2020). *A Primer in BERTology: What We Know About How BERT Works*. TACL.
- Peters, M. E., Ruder, S., Smith, N. A. (2019). *To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks*. RepL4NLP.
- Blitzer, J., Dredze, M., Pereira, F. (2007). *Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification*. ACL.
- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
- Zhu, Y. et al. (2023). *Is GPT-4 a Good Data Annotator?* (anotação automática via LLM).
- Mensh, B., Kording, K. (2017). *Ten Simple Rules for Structuring Papers*. PLOS Computational Biology.
