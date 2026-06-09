# 📝 Relatório de Decisões e Dificuldades — Etapa 1

> **O que é este documento:** o registro honesto da trajetória da preparação de dados (Etapa 1) — cada dificuldade encontrada, o diagnóstico e a correção. Serve para (a) alinhar o grupo, (b) responder perguntas do professor na defesa, e (c) alimentar a seção de **Discussão/Limitações** do paper, onde uma iteração orientada por dados conta como rigor.
>
> **Resumo em uma frase:** o desenho experimental (2×2 de freezing) nunca mudou; o que evoluiu foi *como os dados são selecionados e validados* — em três correções encadeadas, cada uma motivada por um resultado empírico.

---

## 0. O que NÃO mudou (a espinha dorsal)

- **Pergunta de pesquisa:** qual estratégia de *layer freezing* no XLM-RoBERTa mitiga melhor Language Shift vs Domain Shift, em sentimento binário zero-shot cross-lingual.
- **Desenho 2×2:** treina em EN/Domínio-A; testa em EN/A (controle), EN/B (Domain Shift), PT/A (Language Shift), PT/B (combinado).
- **4 configs de freezing** (C1–C4) × **3 seeds** {42,123,2024} = 12 treinos × 4 células = 48 medições de F1-macro.
- **XLM-RoBERTa-base**, rótulos binários (1–2 neg, 4–5 pos, 3 descartado).

---

## 1. Dificuldade nº 1 — Escolha dos domínios

### Tentativa A: Produto × Logística
- **Problema:** vocabulário se sobrepõe muito ("arrived", "package", "box" aparecem em avaliações de produto e de entrega).
- **Evidência:** auditoria do filtro de logística deu **52% de precisão** — abaixo do critério de 80%.
- **Decisão:** descartado.

### Tentativa B: Eletrônicos × Livros
- **Problema:** o B2W (americanas) tem **pouquíssimas avaliações negativas de livros**.
- **Evidência:** o conjunto de treino caía para **~358 amostras** — inviável para o XLM-R.
- **Decisão:** descartado.

### Solução: Eletrônicos × Beleza
- Beleza é categoria grande no B2W, com classe negativa robusta, e vocabulário bem distinto de eletrônicos.
- Bônus metodológico: usar **categorias de produto** como domínios é a operacionalização canônica de *domain shift* na literatura (**Blitzer et al., 2007**), o que fortalece a fundamentação.

---

## 2. Dificuldade nº 2 — Treino com apenas 358 amostras (balanceamento acoplado)

- **Diagnóstico (causa-raiz):** a regra original `N = min(neg, pos) entre os 4 subconjuntos` era aplicada **antes** de separar treino e teste. Assim, o compartimento mais escasso (a classe negativa de PT) **rebaixava o treino inteiro**, mesmo o treino sendo em inglês (onde há dado de sobra).
- **Correção — balanceamento desacoplado:**
  1. O **S1 (treino)** é balanceado no seu **próprio máximo** e dividido 80/20.
  2. Só as **4 células de teste** vão a um N comum (`N_test`).
- **Resultado:** o treino deixou de depender do gargalo de teste. (Mesmo com Livros, o treino voltaria a ser grande; com Beleza, ficou ~10.7k.)

---

## 3. Dificuldade nº 3 — Beleza parecia pequena (filtro errado, não domínio errado)

- **Observação:** mesmo com Beleza, o filtro por **palavra-chave** em PT capturava só **~300** avaliações negativas — quase tão pouco quanto livros.
- **Diagnóstico:** o gargalo era o **método de filtragem**, não o domínio. Rodamos uma **Etapa 0** (notebook `00_Exploracao_Dominios.ipynb`) agrupando o B2W pela coluna de **categoria do produto** (`site_category_lv1`): a categoria `Beleza e Perfumaria` tem **2.372 negativos / 5.515 positivos** — ~8× mais que o filtro por keyword capturava.
- **Correção — filtragem PT por categoria:** PT passou a usar o metadado de categoria (ground-truth, precisão ~100%); EN continua por keyword (o `amazon_polarity` não tem categoria).
  - PT Eletrônicos = `Celulares e Smartphones` + `Informática e Acessórios` + `TV e Home Theater`.
  - PT Beleza = `Beleza e Perfumaria`.
- **Resultado:** células de teste subiram de ~600 → ~2.680.

---

## 4. Dificuldade nº 4 — Auditoria com precisão "ruim" (era o auditor, não o filtro)

- **Observação:** a auditoria zero-shot deu S1=42% e S3=14% (eletrônicos), mas S2=98% e S4=98% (beleza).
- **Diagnóstico:** o **auditor (LLM zero-shot)** estava enviesado para "beleza", não o filtro. Provas:
  1. S3 (PT/Eletrônicos) vem da **categoria** (ground-truth) — são reviews de celular/notebook; 14% é impossível ser erro de filtro. É o LLM falhando (rótulos em inglês sobre texto PT + viés).
  2. O LLM chamava quase tudo de "beleza" (por isso beleza dava 98% e eletrônicos despencava) — viés de comprimento dos rótulos longos.
- **Correções:**
  1. **Auditar só o EN** (S1, S2) — o único lado com filtro sujeito a erro. PT é ground-truth, não auditado.
  2. **Rótulos curtos e simétricos** no zero-shot (`"an electronics product"` × `"a beauty product"`).
  3. **Remover keywords EN ambíguas** (`screen`, `camera`, `display`, `speaker`, `keyboard`, `processor`) que vazavam para reviews de filme/música/cozinha.
- **Resultado:** S1 saltou de **42% → 90%**; S2 = **94%**. Ambos acima do critério de 80%.

---

## 5. Estado final da Etapa 1 (números realizados)

| Item | Valor | Status |
|---|---|---|
| Treino (S1) | **10.718** | ✅ |
| Células de teste (T1–T4) | **2.680 cada** (1.340/classe, balanceadas) | ✅ |
| Precisão filtro EN (auditoria) | **90% / 94%** | ✅ ≥ 80% |
| Precisão PT (categoria) | **100%** (ground-truth) | ✅ |
| Interseção de IDs entre partições | nenhuma | ✅ |

**Subconjuntos:** S1 EN/Eletrônicos (treino+val) · S2 EN/Beleza (Domain Shift) · S3 PT/Eletrônicos (Language Shift) · S4 PT/Beleza (combinado).

---

## 6. Como usar isto no paper / na defesa

- **Discussão:** o trabalho fez **seleção de domínio e de método de filtragem orientada por dados** (Etapa 0), em vez de escolher arbitrariamente. Isso é um diferencial de rigor.
- **Limitações honestas:** (a) a atribuição de domínio é **assimétrica** (EN por keyword ~90–94% de precisão; PT por categoria ~100%); (b) as células de teste (~2.680) ficam abaixo do teto ideal de 5.000.
- **Perguntas prováveis da banca e respostas curtas:**
  - *"Por que filtrar diferente em cada idioma?"* → Porque o B2W tem categoria de produto (metadado confiável) e o `amazon_polarity` não; usamos o melhor método disponível em cada base.
  - *"Por que não auditaram o PT?"* → Porque o domínio em PT vem da categoria do produto, que é ground-truth; auditar isso com um LLM mais fraco só injetaria ruído.
  - *"Por que trocaram de domínio duas vezes?"* → Cada troca foi motivada por um problema medido (precisão de filtro 52%; treino de 358 amostras), não por gosto.
