# Análise Arquitetural do Congelamento de Camadas na Mitigação de *Domain Shift* e *Language Shift* em Transformers Multilíngues

Estudo empírico sobre a robustez do XLM-RoBERTa em classificação binária de sentimento *zero-shot cross-lingual*. O trabalho avalia em que medida o congelamento seletivo de camadas durante o *fine-tuning* mitiga a degradação de desempenho sob dois tipos de deslocamento de distribuição: mudança de idioma (*Language Shift*) e mudança de domínio (*Domain Shift*).

O ciclo experimental está completo: preparação de dados, implementação da arquitetura, treino dos modelos e análise estatística. As decisões de projeto foram, em vários pontos, revisadas com base em medições — o histórico dessas revisões está documentado em `docs/DECISOES_E_DIFICULDADES.md`.

---

## 1. Dados e Decisões de Engenharia (Etapa 1)

A preparação de dados foi a etapa de maior complexidade metodológica. O objetivo é treinar em inglês/Domínio A e avaliar em quatro combinações de idioma × domínio para isolar cada tipo de deslocamento.

### 1.1 Seleção dos domínios

A escolha do par de domínios passou por três iterações, cada uma motivada por uma medição:

1. **Produto × Logística (descartado).** Vocabulário excessivamente sobreposto (termos como *box*, *arrived*, *package* ocorrem em ambos). A auditoria do filtro atingiu apenas 52% de precisão, abaixo do critério de 80%.
2. **Eletrônicos × Livros (descartado).** A base em português (B2W) contém pouquíssimas avaliações negativas de livros; sob a regra de balanceamento, o conjunto de treino caía para 358 amostras, insuficiente para o *fine-tuning*.
3. **Eletrônicos × Beleza (adotado).** Vocabulário distinto, classe negativa robusta em ambos os domínios e volume adequado. A operacionalização de domínio por categoria de produto segue Blitzer et al. (2007).

### 1.2 Filtragem híbrida e balanceamento

- **Filtragem assimétrica por base.** No inglês (Amazon, `amazon_polarity`), que não dispõe de metadado de categoria, o domínio é atribuído por palavras-chave sobre o texto; a precisão do filtro foi auditada por um classificador zero-shot multilíngue (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`), resultando em 90% (Eletrônicos) e 94% (Beleza). No português (B2W), o domínio vem da categoria oficial `site_category_lv1` (*ground-truth*, precisão ≈ 100%).
- **Balanceamento desacoplado.** O dimensionamento do treino foi separado do das células de teste, evitando que o compartimento de teste mais escasso reduzisse o conjunto de treino. Resultado: 10.718 exemplos de treino e 2.680 exemplos por célula de teste (1.340 por classe).

---

## 2. Hipóteses

A literatura de interpretabilidade de Transformers (*BERTology*) sugere uma hierarquia funcional entre as camadas:

- **Camadas inferiores (0–5):** processamento léxico e sintático (representações associadas ao idioma).
- **Camadas superiores (6–11):** abstrações semânticas alinhadas à tarefa.

A partir dessa hierarquia, foram formuladas duas hipóteses:

- **H1 (*Language Shift*):** congelar as camadas inferiores (`Freeze Lower`, **C2**) preserva o alinhamento multilíngue do pré-treino e reduz a queda de desempenho ao inferir em português um modelo treinado em inglês.
- **H2 (*Domain Shift*):** congelar as camadas superiores (`Freeze Upper`, **C3**) reduz a especialização excessiva no domínio de Eletrônicos e a queda de desempenho ao avaliar em Beleza.

O desenho compara quatro configurações de congelamento (C1 *Full Fine-Tuning*, C2 *Freeze Lower*, C3 *Freeze Upper*, C4 *Frozen Encoder*) sob três *seeds* independentes.

---

## 3. Resultados Experimentais

Após o treino dos 12 modelos (Etapa 3) com *early stopping* sobre a validação, cada modelo foi avaliado nas quatro células de teste, produzindo 48 medições de F1-macro e acurácia. Ambas as hipóteses iniciais foram refutadas pelos dados.

### 3.1 Language Shift: ausência de degradação mensurável

O modelo completo (C1), treinado em Eletrônicos/Inglês, obteve F1-macro de 0.947 na própria condição (T1, EN/Eletrônicos) e 0.956 em português (T3, PT/Eletrônicos) — desempenho *zero-shot* ligeiramente superior ao da condição de treino. Não havendo degradação de idioma a mitigar, **H1 não pôde ser comprovada**.

### 3.2 Domain Shift: real, mas com efeito de H2 invertido

Na transferência para o domínio de Beleza (T2, EN/Beleza), o C1 caiu para 0.914. A configuração prevista para mitigar essa queda (C3, congelamento do topo) obteve o pior resultado entre as estratégias ativas (0.885). A menor degradação foi observada em C2 (congelamento da base), que atingiu 0.926 — acima do *fine-tuning* completo.

O resultado indica que as representações multilíngues das camadas iniciais, quando preservadas, atuam como regularizador contra o viés de domínio.

---

## 4. Análise Estatística (Etapa 4)

As 48 medições (`results.csv`) foram submetidas a testes formais para decidir o destino das hipóteses. O teste primário é o **t de Welch** (variâncias desiguais); o **Mann-Whitney U** serve como confirmação não-paramétrica e o **Cohen's *d*** como tamanho de efeito. Cada estratégia foi comparada à baseline C1, com limiar de significância p < 0.10 (adequado a n=3).

- **H1 (*Language Shift*) — refutada por inexistência do fenômeno.** A baseline C1 não se degrada ao migrar para o português (Δ T1−T3 = −0.83 pp). C2 vs C1 em PT: Δ = +0.03 pp (p = 0.93), efeito desprezível.
- **H2 (*Domain Shift*) — refutada e invertida.** Congelar o topo (C3) piorou o desempenho em Beleza de forma significativa: Δ = −2.97 pp vs C1 (p = 0.074).
- **Achado emergente — confirmado.** A mitigação do *Domain Shift* vem de **C2 (*Freeze Lower*)**: Δ = +1.19 pp sobre a baseline (p = 0.086; Cohen's *d* = +2.45, efeito grande).

Tabelas, gráficos (heatmap e barplot de deltas) e o relatório completo estão em **[`docs/RESULTADOS-Etapa4.md`](docs/RESULTADOS-Etapa4.md)**. Reprodução: `python src/analise_etapa4.py` ou o notebook `notebooks/Trabalho_RNP_Colab_Etapa4.ipynb` (não requer GPU).

> **Escopo das conclusões.** O resultado sobre *Language Shift* é restrito ao par EN↔PT — duas línguas próximas e de alta cobertura no XLM-R — e está confundido com a assimetria de qualidade dos conjuntos: o lado português é *ground-truth* (categoria), enquanto a baseline inglesa é ruidosa (filtro por palavra-chave, 90–94% de precisão). Não se afirma, portanto, que o XLM-R seja imune a *Language Shift*. A condição que refutaria essa tese é a avaliação em uma língua tipologicamente distante. As limitações completas e as condições de refutação estão em [`docs/Metodologia_Rascunho.md`](docs/Metodologia_Rascunho.md) (§11 e §2.1).

---

## 5. Status do Projeto

| Etapa | Descrição | Status |
|-------|-----------|--------|
| 1 | Engenharia de dados (filtros, auditoria, *splits*) | Concluída |
| 2 | Implementação da arquitetura e do congelamento C1–C4 | Concluída |
| 3 | Pipeline de treino e extração de métricas (12 modelos) | Concluída |
| 4 | Análise estatística e visualização | Concluída |
| 5 | Replicação (6–8 seeds, 3 integrantes) e teste de língua distante | Concluída (Passo A em aberto — ver abaixo) |

O ciclo experimental está fechado (dados → arquitetura → treino → análise). Os entregáveis de análise (tabelas, gráficos e veredito das hipóteses) estão disponíveis para a defesa.

### 4.1 Etapa 5 — Replicação independente e língua distante

Três integrantes re-executaram as configurações C1–C4 com **seeds independentes** (`results_etapa5.csv`, 182 medições), elevando o N de 3 para **6–8 seeds** por configuração e adicionando uma tentativa de teste em **língua distante** (Japonês/Mandarim, dataset MARC).

- **Replicação confirma a Etapa 4.** As configs viáveis (C1/C2/C3) reproduzem as médias da Etapa 4 dentro de **< 1 pp**, com seeds totalmente diferentes. H1 e H2 seguem **refutadas** — H2 agora com evidência forte (C3 piora o domínio: **Δ = −3.10 pp, p < 0.001**, antes apenas p = 0.074).
- **Correção honesta.** O "achado emergente" da Etapa 4 (C2 mitiga *Domain Shift*) **não replica**: com mais seeds, C2 vs C1 em Beleza dá Δ = +0.21 pp (p = 0.42). C2 é **estatisticamente equivalente** à C1 — seu valor é a **eficiência** (treina ~50% menos parâmetros sem perda), não a regularização.
- **C4 (Frozen Encoder) é instável**, não apenas fraco: o F1 varia de 0.36 a 0.81 entre seeds (desvio ≈ 16× o das demais). *Probing* linear do XLM-R nesta tarefa é uma loteria de seed.
- **Passo A em aberto.** As células de língua distante (T5=JA, T6=ZH, T7=EN-âncora) saíram **byte-a-byte idênticas** em todas as 26 execuções — o carregador do espelho MARC ignorou o filtro de língua e avaliou o mesmo conjunto três vezes. A decomposição EN→JA vs EN→ZH não está disponível; a correção (pequena) e o caminho de re-execução barata estão documentados.

Relatório completo: **[`docs/RESULTADOS-Etapa5.md`](docs/RESULTADOS-Etapa5.md)**. Conteúdo para slides (Gamma): **[`docs/APRESENTACAO-Etapa5.md`](docs/APRESENTACAO-Etapa5.md)**. Reprodução: `python src/analise_etapa5.py` (não requer GPU).

---

## 6. Estrutura do Repositório

* **`docs/`** — documentação do projeto.
  * `Guia_Estudo_e_Defesa.md`: material de estudo dos conceitos e 13 perguntas preparatórias de defesa.
  * `DECISOES_E_DIFICULDADES.md`: registro técnico das revisões de domínio e filtragem (Etapa 1).
  * `Metodologia_Rascunho.md`: estrutura base do paper, incluindo limitações (§11) e falseabilidade (§2.1).
  * `PLAN-Etapa*.md`: documentos de planejamento de execução por etapa.
  * `RESULTADOS-Etapa4.md`: relatório da análise estatística (tabelas, p-valores, deltas e veredito das hipóteses).
  * `RESULTADOS-Etapa5.md`: replicação com 6–8 seeds, teste de língua distante e alerta de integridade do Passo A.
  * `APRESENTACAO-Etapa5.md`: conteúdo pronto para gerar os slides (Gamma).
* **`src/`** — código-fonte.
  * `model.py`: arquitetura e algoritmo de congelamento (C1–C4).
  * `analise_etapa4.py`: pipeline reprodutível da análise estatística e dos gráficos (Etapa 4).
  * `analise_etapa5.py`: combinação dos CSVs dos três integrantes, testes estatísticos e gráficos (Etapa 5).
* **`notebooks/`** — notebooks executáveis (Google Colab / Kaggle). Destaque para `Trabalho_RNP_Colab_Completo.ipynb`, que reúne **todo o percurso ponta a ponta** (pergunta de pesquisa, fundamentação, hipóteses, engenharia de dados, arquitetura, treino e análise estatística consolidada multi-seed); inclui também `..._Etapa4.ipynb` (análise) e `Trabalho_RNP_Kaggle_Etapa5.ipynb` (experimento de língua distante).
* **`tests/`** — testes unitários da lógica de congelamento.
* **`resultados/`** — saídas das análises: tabelas (`.csv`) e gráficos. Etapa 4: `heatmap_f1_macro.png`, `barplot_deltas.png`. Etapa 5: `*_etapa5.png/.csv` e os CSVs brutos por integrante em `etapa5_raw/`.
* **`results.csv`** — as 48 medições da Etapa 3 (entrada da Etapa 4).
* **`results_etapa5.csv`** — as 182 medições combinadas dos três integrantes (entrada da Etapa 5).
