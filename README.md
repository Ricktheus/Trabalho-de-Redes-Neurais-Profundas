# 🚀 O Que Fizemos Até Agora (Resumo Completo da Trajetória)

> **Para que serve este documento:** Um compêndio completo e estruturado para que qualquer membro da equipe ou avaliador entenda a fundo a trajetória do projeto. Ele consolida o **embasamento teórico**, as **decisões de engenharia de dados**, e os **resultados reais já executados** nas Etapas 1 e 2.

---

## 1. O Nosso Objetivo e Embasamento Teórico

Estamos investigando o fenômeno do *Zero-Shot Cross-Lingual*, onde modelos de linguagem pré-treinados perdem performance quando aplicados em um idioma diferente daquele usado no *fine-tuning*. 
Queremos descobrir quem é o maior vilão da degradação de performance (avaliada via *F1-macro*):
- **Language Shift:** Mudar o idioma de Inglês para Português.
- **Domain Shift:** Mudar o contexto do assunto de "Eletrônicos" para "Beleza".

**A nossa Hipótese (baseada em *BERTology*):**
A literatura de *BERTology* sugere que em modelos Transformer, as camadas iniciais (0-5) codificam informações léxico-sintáticas (idioma), enquanto as camadas finais (6-11) codificam informações semânticas de alto nível (a tarefa/domínio). Portanto:
* **H1:** Congelar as camadas iniciais (*Freeze Lower*) ajuda a preservar o alinhamento de idiomas, mitigando o *Language Shift*.
* **H2:** Congelar as camadas finais (*Freeze Upper*) evita que o modelo sofra *over-especialização* no domínio de treino, mitigando o *Domain Shift*.

---

## 2. A Trajetória de Dados (Etapa 1 Executada e Validada ✅)

O projeto sofreu iterações profundas na escolha de dados para garantir rigor científico. Os testes iniciais com os domínios "Produto vs Logística" ou "Eletrônicos vs Livros" falharam devido à alta sobreposição de vocabulário e escassez de avaliações negativas. 

### A Estratégia Híbrida de Domínios
Decidimos utilizar os domínios **Eletrônicos × Beleza**, operando de forma assimétrica para garantir a máxima qualidade possível:
* **Inglês (Dataset Amazon):** Como a base não tem categoria de produto, usamos um **filtro lexical por palavras-chave** inequívocas (ex: `battery`, `usb` vs `skin`, `perfume`).
* **Português (Dataset B2W):** Como a base possui o metadado `site_category_lv1`, usamos a **categoria real do produto** (ground-truth, ~100% de precisão).

### Auditoria Zero-Shot e Mapeamento
* **Mapeamento de Rótulos:** Convertemos a escala de 1 a 5 estrelas em binária: 1-2 = Negativo, 4-5 = Positivo, descartando o 3 (neutro).
* **Auditoria LLM:** Para provar que o filtro em inglês era robusto, passamos amostras pelo classificador `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (Zero-Shot). O resultado foi excelente: batemos mais de **90% de precisão**, superando nosso limiar de 80%.

### O Balanceamento Desacoplado (A Grande Sacada)
Se balanceássemos todos os conjuntos pelo menor deles, o nosso treino seria minúsculo. Em vez disso, nós *desacoplamos*:
* O **Treino (S1 - EN/Eletrônicos)** foi balanceado no seu próprio máximo, resultando em incríveis **~10.700 exemplos**.
* As **Células de Teste (T1 a T4)** foram niveladas exatamente pelo mesmo tamanho, resultando em **~2.680 exemplos** para cada.
O código garantiu que **não há interseção** (nenhum ID repetido) entre as partições.

---

## 3. A Cirurgia Matemática no Modelo (Etapa 2 Executada ✅)

Nesta etapa, instanciamos a nossa arquitetura no Colab: o **`XLM-RoBERTa-base`** (pré-treinado em 100 idiomas com ~2.5TB de dados), totalizando cerca de 278 milhões de parâmetros. Adicionamos uma nova cabeça de classificação (`RobertaClassificationHead`).

Nós criamos, testamos via código, e comprovamos o funcionamento da nossa função de *Layer Freezing*. Abaixo, o número **real** de parâmetros que serão treinados em cada cenário:

1. **C1 (Baseline / Full FT):** Treina o modelo inteiro (**278.045.186** parâmetros).
2. **C2 (Freeze Lower / H1):** Congela os embeddings e as camadas 0 a 5. Treina apenas as camadas superiores e a cabeça (**43.119.362** parâmetros).
3. **C3 (Freeze Upper / H2):** Congela as camadas 6 a 11. Treina as camadas inferiores, embeddings e a cabeça (**235.517.954** parâmetros).
4. **C4 (Piso / Frozen Encoder):** Congela toda a base do XLM-R. Treina somente a nova cabeça de classificação (**apenas 592.130** parâmetros).

Rodamos um *Smoke Test* (forward e backward pass em um lote de dados reais) e o PyTorch comprovou matematicamente que o gradiente fluiu apenas pelos parâmetros não congelados.

---

## 4. A Matriz de Avaliação e o Motor de Treino (Etapa 3 Planejada 🔜)

Com tudo testado, a Etapa 3 será a "fábrica" de treinamento. O plano já estruturado no documento `PLAN-Etapa3.md` prevê o uso do `Trainer` da Hugging Face com os seguintes hiperparâmetros fixos:
* **Otimizador:** AdamW (Learning Rate: 2e-5, Weight Decay: 0.01)
* **Warmup:** 10% dos passos, em precisão mista (`fp16`) para economizar memória GPU.
* **Early Stopping:** Paciência de 1 época monitorando a `eval_loss` em um split de 10% do treino original, evitando *overfitting*.

### A Matriz de 48 Testes
Nós rodaremos cada uma das 4 configurações (C1-C4) utilizando **3 seeds diferentes (42, 123, 2024)** para garantir relevância estatística, totalizando **12 treinamentos**.
Cada um desses 12 modelos será avaliado em **4 cenários (Células de Teste)**:
* **T1 (Controle):** Inglês / Eletrônicos (S1-val)
* **T2 (Domain Shift):** Inglês / Beleza (S2)
* **T3 (Language Shift):** Português / Eletrônicos (S3)
* **T4 (Combinado):** Português / Beleza (S4)

O script vai gravar em tempo real os resultados em uma planilha final (`results.csv`).

---

## 5. Onde Estamos Exatamente Agora?

1. A teoria está documentada com rigor na metodologia.
2. Os dados de teste e treino estão limpos, balanceados e salvos no Drive (Etapa 1 rodada).
3. O código que congela as camadas está matematicamente provado no PyTorch (Etapa 2 rodada).
4. O plano de treino e os hiperparâmetros estão definidos e prontos para virar código.

**Próximo Passo:** Pegar as diretrizes do `PLAN-Etapa3.md`, transformá-las no script do Colab da Etapa 3 e dar o "Play". O sistema rodará os 12 experimentos sozinho. A partir dos resultados, montaremos os gráficos (Etapa 4) e descobriremos se as nossas hipóteses se sustentam!
