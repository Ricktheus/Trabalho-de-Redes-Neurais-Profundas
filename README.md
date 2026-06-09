# Análise Arquitetural do Congelamento de Camadas na Mitigação de *Domain Shift* e *Language Shift* em Transformers Multilíngues

Este repositório documenta um projeto de pesquisa focado em testar a robustez de modelos baseados em Transformers (XLM-RoBERTa) em cenários de classificação *zero-shot cross-lingual*. 

Ao longo do desenvolvimento, este projeto passou por diversas pivotagens e tomadas de decisão baseadas em dados, resultando em um framework de testes sólido e revelando descobertas inesperadas sobre o comportamento interno das camadas de atenção.

---

## 1. A Jornada do Projeto: Decisões e Obstáculos Iniciais

A construção dos dados (Etapa 1) provou ser o maior desafio metodológico do trabalho. A meta inicial era treinar um modelo em Inglês e testá-lo em Português (Language Shift), treinando em um domínio "A" e testando em um domínio "B" (Domain Shift).

### A Busca pelos Domínios Perfeitos
1. **Tentativa 1 (Logística):** Originalmente, tentamos usar "Avaliações de Produtos" vs "Avaliações de Logística/Entrega". No entanto, descobrimos que o vocabulário se sobrepunha excessivamente (palavras como *box*, *arrived*, *package* apareciam em ambos). O filtro de domínio falhou nos testes de auditoria, alcançando apenas 52% de precisão.
2. **Tentativa 2 (Livros):** Mudamos para "Eletrônicos vs Livros". O problema? A base de dados brasileira B2W possuía raríssimas avaliações *negativas* de livros. Por conta do nosso algoritmo de balanceamento rigoroso, isso fazia o conjunto inteiro de treinamento desabar para pífias 358 amostras. O XLM-RoBERTa não conseguiria aprender com tão pouco.
3. **A Solução (Beleza):** Optamos por **Eletrônicos vs Beleza**. O vocabulário é distinto, a base possui milhares de avaliações e nos permite testar o *Domain Shift* de forma clássica.

### Superando o Gargalo do Balanceamento e da Filtragem
Para garantir a sanidade dos dados, tomamos duas grandes decisões de engenharia:
- **Filtragem Híbrida Assimétrica:** No lado Inglês (Amazon), usamos palavras-chave precisas (`battery`, `skin`, etc.) validadas por uma inteligência artificial que comprovou precisão superior a 90%. No lado Português (B2W), usamos diretamente o metadado `Categoria` oficial da loja (ground-truth com precisão próxima a 100%).
- **Balanceamento Desacoplado:** Modificamos o algoritmo de split para que o gargalo de dados de um teste não punisse o conjunto de treino. O resultado final nos garantiu **10.718 exemplos no treino** e **2.680 exemplos exatos para cada uma das 4 células de teste**.

---

## 2. A Teoria e Nossas Hipóteses

Estudos de *BERTology* sugerem que as camadas de um Transformer funcionam como uma hierarquia:
- **Camadas Inferiores (0-5):** Especializam-se em processamento léxico e sintático profundo (linguagem).
- **Camadas Superiores (6-11):** Especializam-se em abstrações semânticas alinhadas à tarefa (no nosso caso, sentimento de domínio).

A partir disso, formulamos:
- **Hipótese 1 (H1 - Language Shift):** O congelamento das camadas inferiores (`Freeze Lower` - **C2**) preservará o alinhamento idiomático do pré-treinamento, reduzindo a queda de performance quando o modelo, treinado em Inglês, precisar inferir em Português.
- **Hipótese 2 (H2 - Domain Shift):** O congelamento das camadas superiores (`Freeze Upper` - **C3**) impedirá que o modelo fique viciado ("overfitting") nos jargões de Eletrônicos, reduzindo a queda de performance ao testar em Beleza.

Definimos 4 configurações de arquitetura (C1 completo, C2 Lower Freeze, C3 Upper Freeze, C4 Frozen Encoder) rodando em 3 *seeds* independentes para garantir consistência.

---

## 3. O Experimento e os Resultados Surpreendentes

Após treinar os 12 modelos (Etapa 3) com *early stopping* sobre a validação, nós realizamos avaliações massivas gerando 48 medições de F1-macro e Acurácia. A extração dos resultados nos deixou boquiabertos. Ambas as nossas hipóteses foram esmagadas pela realidade empírica do XLM-RoBERTa.

### O *Language Shift* Simplesmente não Existiu
O modelo completo (C1) treinado em Eletrônicos/Inglês alcançou um F1-macro de **0.947**. Ao colocá-lo para prever avaliações de Eletrônicos em Português (onde esperávamos uma queda bruta), o modelo alcançou **0.955**! O XLM-R foi **melhor** no idioma em que não treinou (zero-shot). 
Como a degradação de idioma foi nula, a nossa **H1 não pôde ser comprovada** (não se pode mitigar uma queda que não existe).

### O *Domain Shift* Foi Real, mas H2 foi Invertida
Quando testamos os modelos no domínio de Beleza (Inglês), a queda foi sentida. O modelo completo (C1) caiu para **0.914** de F1-macro.
- A nossa aposta para salvar o modelo no domínio cruzado era a configuração **C3** (congelar o topo). Ironicamente, o C3 despencou ainda mais, marcando **0.884** (foi a pior das estratégias ativas).
- A verdadeira heroína foi a configuração **C2** (congelar a base sintática). O C2 não só resistiu à queda como pontuou **0.926** em Beleza, superando o próprio modelo de *fine-tuning* completo.

Esse resultado é ouro puro: **prova empírica de que as representações multilíngues iniciais do XLM-RoBERTa atuam como poderosos regularizadores contra o enviesamento de domínio!**

---

## 4. Status e Próximos Passos

1. Engenharia de Dados (Filtros, Auditoria, Splits) — **Concluído (Etapa 1)**
2. Implementação da Arquitetura C1-C4 — **Concluído (Etapa 2)**
3. Pipeline de Treinamento e Extração de Métricas — **Concluído (Etapa 3)**
4. **[AÇÃO PENDENTE] Análise Estatística e Visualização — (Etapa 4):**
   Vamos calcular os p-valores (`mannwhitneyu`) e gerar matrizes de calor (Heatmaps) e gráficos de barras com `seaborn` para ilustrar de forma irrefutável os deltas das hipóteses derrubadas. O guia de execução desta etapa já está traçado no arquivo `docs/PLAN-Etapa4.md`.

---

## 5. Estrutura do Repositório

Nossa documentação e código estão enxutos e categorizados:

* **`docs/`**: O coração literário e investigativo do projeto.
  * `Guia_Estudo_e_Defesa.md`: A bíblia do projeto (estude conceitos complexos e veja 27 perguntas preparatórias de defesa).
  * `DECISOES_E_DIFICULDADES.md`: O relatório técnico completo sobre as pivotagens de domínio citadas acima.
  * `Metodologia_Rascunho.md`: Estrutura base do *paper* científico.
  * `PLAN-Etapa*.md`: Documentos técnicos de planejamento de execução de código.
* **`src/`**: Códigos-fonte (`.py`).
* **`notebooks/`**: Onde a mágica acontece. Ambientes interativos (`.ipynb`) do Google Colab.
* **`tests/`**: Testes unitários para proteção da lógica.
* **`resultados/`**: (Opcional localmente) Armazena arquivos brutos de teste estatístico como o `results.csv`.
