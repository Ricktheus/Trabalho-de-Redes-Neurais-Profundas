# Resultados — Etapa 5: Replicação com Mais Seeds e Teste de Língua Distante

Relatório da **Etapa 5** do projeto *"Análise Arquitetural do Congelamento de Camadas na Mitigação de Domain Shift e Language Shift em Transformers Multilíngues"*. A etapa tinha três objetivos:

1. **Consolidar** os resultados gerados independentemente por três integrantes (Sebastião, Pedro e Geovanna), cada um com suas próprias *seeds*.
2. **Estressar estatisticamente** as conclusões da Etapa 4, que se apoiavam em apenas 3 *seeds*, agora com 6–8 *seeds* independentes (replicação).
3. **Passo A** — testar se a ausência de *Language Shift* observada em EN→PT (línguas próximas) também vale para uma língua **tipologicamente distante** (Japonês / Mandarim), via o dataset MARC.

> **Reprodução:** `python src/analise_etapa5.py` (combina os CSVs, roda os testes e gera tabelas `.csv` e gráficos `.png` em `resultados/`). Não requer GPU.

---

## 1. Material de entrada

Três CSVs independentes, em `resultados/etapa5_raw/`, todos gerados pelo notebook [`notebooks/Trabalho_RNP_Kaggle_Etapa5.ipynb`](../notebooks/Trabalho_RNP_Kaggle_Etapa5.ipynb):

| Contribuidor | Configs | Seeds | Linhas |
|---|---|---|---|
| Sebastião | C1–C4 | `2718, 4242, 9001` | 84 |
| Pedro | C1–C4 | `1234, 5678, 91011` | 84 |
| Geovanna | **só C1** | `13, 888` | 14 |

Combinados em **`results_etapa5.csv`** (182 linhas; coluna `contribuidor` adicionada para proveniência). Schema: `config, seed, teste, f1_macro, accuracy, f1_negativo, f1_positivo, contribuidor`.

**Seeds efetivas por configuração** (N dos testes estatísticos):

| Config | N (seeds) | Observação |
|---|:---:|---|
| **C1** | **8** | 6 de Sebastião+Pedro + 2 de Geovanna |
| **C2** | 6 | Sebastião + Pedro |
| **C3** | 6 | Sebastião + Pedro |
| **C4** | 6 | Sebastião + Pedro |

> ⚠️ **Run parcial.** O notebook da Geovanna completou apenas a C1 (provável esgotamento de cota de GPU). O N fica desigual (C1=8, demais=6); o **t de Welch** e o **Mann-Whitney U** toleram amostras de tamanhos diferentes, e o desbalanceamento favorece a precisão da *baseline* (mais seeds em C1). Reportado com transparência.

**Células de teste** (treino sempre em S1 = EN/Eletrônicos):

| Célula | Conjunto | Idioma / Domínio | Tipo de shift |
|---|---|---|---|
| **T1** | S1_val | EN / Eletrônicos | nenhum (*baseline*) |
| **T2** | S2 | EN / Beleza | **Domain Shift** |
| **T3** | S3 | PT / Eletrônicos | **Language Shift próximo** (B2W, *ground-truth*) |
| **T4** | S4 | PT / Beleza | Domínio + Língua |
| **T5/T6/T7** → **TM** | MARC | JA / ZH / EN-âncora | **Language Shift distante** — ⚠️ ver §2 |

---

## 2. ⚠️ Alerta de integridade dos dados — o Passo A não mediu o que pretendia

**As células T5 (JA), T6 (ZH) e T7 (EN-âncora) têm F1-macro byte-a-byte idêntico em TODAS as 26 execuções** (config × seed). Exemplo (C1, seed 2718): `T5 = T6 = T7 = 0.8675234985312674`.

Três conjuntos de texto distintos — japonês, mandarim e inglês — **não podem** produzir o mesmo F1 com 16 casas decimais. A única explicação é que o carregador do espelho `mteb/amazon_reviews_multi` **ignorou o argumento de língua** (`name=lang`): o trecho `for kwargs in ({"name": lang}, {})` falhou na primeira tentativa e caiu no *fallback* `{}`, que devolve **o mesmo split padrão** para `ja`, `zh` e `en`. Como o balanceamento usa `random_state` fixo, as três células viraram **o mesmo conjunto amostrado**.

**Consequências:**

- A decomposição **EN→JA vs EN→ZH** — o coração do Passo A — **não está disponível** nestes dados.
- Tratamos T5=T6=T7 como uma **única célula consolidada `TM`** (MARC, **domínio misto**, **língua indeterminada**).
- O degrau de F1 em `TM` (§3) **não pode ser lido como *Language Shift***: ele confunde língua, domínio (misto, não Eletrônicos) e fonte de rótulo. O veredito do Passo A fica **em aberto**, pendente de re-execução. A correção exata está em [§7](#7-passo-a-em-aberto--correção-para-re-executar).

> Esta seção é o achado metodológico mais importante da Etapa 5: o experimento foi reproduzido por três pessoas e **o mesmo bug apareceu nas três** — evidência de que está na lógica do notebook, não no ambiente de cada um.

---

## 3. F1-macro consolidado (média ± desvio das seeds)

| Config | T1 · EN/Elec | T2 · EN/Beleza (Domínio) | T3 · PT/Elec (Língua próx.) | T4 · PT/Beleza | TM · MARC* |
|---|:---:|:---:|:---:|:---:|:---:|
| **C1** · Full | 0.9412 ± 0.0041 | 0.9174 ± 0.0063 | **0.9552 ± 0.0026** | **0.9500 ± 0.0048** | 0.8794 ± 0.0194 |
| **C2** · Freeze Lower | 0.9373 ± 0.0059 | **0.9195 ± 0.0029** | 0.9536 ± 0.0030 | 0.9488 ± 0.0021 | **0.8886 ± 0.0077** |
| **C3** · Freeze Upper | 0.9255 ± 0.0037 | 0.8864 ± 0.0083 | 0.9365 ± 0.0124 | 0.9335 ± 0.0159 | 0.8052 ± 0.0506 |
| **C4** · Frozen Encoder | 0.6474 ± 0.0776 | 0.4584 ± 0.0779 | 0.6416 ± 0.1052 | 0.5593 ± 0.1324 | 0.6042 ± 0.0672 |

> `*` TM = T5/T6/T7 colapsadas (ver §2). Negrito = melhor da coluna.

![Heatmap F1-macro Etapa 5](../resultados/heatmap_f1_macro_etapa5.png)

Leituras imediatas:
- **C1 e C2 são praticamente empatadas** em T1–T4 — diferenças ≤ 0.4 pp.
- **C3 (Freeze Upper) é consistentemente pior** que C1/C2 em todas as células.
- **C4 (Frozen Encoder) colapsa** (0.46–0.65) **e com desvio enorme** (até ±0.13) — ver §6.

---

## 4. Δ-shift por configuração (queda de F1-macro vs. baseline T1, em pp)

Convenção: `Δ = F1(T1) − F1(Tx)`. **> 0 = perda** sob shift; **< 0 = ganho** zero-shot.

| Config | Δ Domínio (T1−T2) | Δ Língua próx. (T1−T3) | Δ Ambos (T1−T4) | Δ MARC* (T1−TM) |
|---|:---:|:---:|:---:|:---:|
| **C1** | +2.38 | **−1.40** | −0.88 | +6.17 |
| **C2** | +1.78 | −1.63 | −1.15 | +4.88 |
| **C3** | +3.90 | −1.11 | −0.80 | +12.03 |
| **C4** | +18.90 | +0.58 | +8.82 | +4.32 |

![Barplot Δ-shift Etapa 5](../resultados/barplot_deltas_etapa5.png)

- **Language Shift próximo (EN→PT) continua ausente** (barras azuis negativas): todas as configs *ativas* (C1–C3) vão **igual ou melhor** em PT do que em EN — ganho *zero-shot cross-lingual*, reconfirmado com 6–8 seeds.
- **Domain Shift (EN/Elec→EN/Beleza) é real** (barras vermelhas positivas) e cresce de C2 (+1.78) para C3 (+3.90) e dispara em C4 (+18.90).
- **MARC (TM)** mostra um degrau grande (C1 = +6.17 pp), mas — repita-se — **confundido** (§2), não interpretável como língua.

---

## 5. Testes estatísticos vs. baseline C1 (Welch t + Mann-Whitney U + Cohen's *d*)

Teste primário **Welch *t*** (variâncias desiguais); **Mann-Whitney U** como confirmação não-paramétrica; **Cohen's *d*** como tamanho de efeito. Limiar **p < 0.10** (Metodologia). N = 6–8 seeds por grupo.

| Cenário | Comparação | Δ (pp) | p (Welch) | p (MWU) | Cohen's *d* | Magnitude | Sig. p<0.10 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| T2 | C2 vs C1 | +0.21 | 0.417 | 0.662 | +0.41 | pequeno | ❌ não |
| T2 | C3 vs C1 | **−3.10** | **0.0000** | 0.001 | −4.32 | grande | ✅ SIM (pior) |
| T2 | C4 vs C1 | **−45.90** | **0.0000** | 0.001 | −9.08 | grande | ✅ SIM (pior) |
| T3 | C2 vs C1 | −0.16 | 0.317 | 0.228 | −0.58 | médio | ❌ não |
| T3 | C3 vs C1 | **−1.87** | **0.013** | 0.003 | −2.27 | grande | ✅ SIM (pior) |
| T3 | C4 vs C1 | **−31.36** | **0.001** | 0.001 | −4.62 | grande | ✅ SIM (pior) |
| TM | C2 vs C1 | +0.91 | 0.257 | 0.573 | +0.58 | médio | ❌ não |
| TM | C3 vs C1 | **−7.43** | **0.014** | 0.008 | −2.07 | grande | ✅ SIM (pior) |
| TM | C4 vs C1 | **−27.53** | **0.0001** | 0.001 | −6.01 | grande | ✅ SIM (pior) |

Com mais seeds, o **MWU agora atinge p ≈ 0.001** (não satura mais em 0.10 como ocorria com n=3 na Etapa 4): a evidência ficou muito mais forte. Em **nenhuma** célula uma config congelada supera a C1 de forma significativa; C3 e C4 são significativamente **piores** em todas.

---

## 6. Veredito das hipóteses — replicação independente da Etapa 4

A Etapa 5 usa **seeds totalmente diferentes** das da Etapa 4 (`42, 123, 2024`). Logo, é uma **replicação independente**, não apenas "mais do mesmo".

### ❌ H1 — *Freeze Lower (C2) mitiga o Language Shift* → **REFUTADA** (reconfirmada)
Não há shift a mitigar: baseline C1 tem `T1 − T3 = −1.40 pp` (ganha em PT). C2 vs C1 em T3: **Δ = −0.16 pp, p = 0.32** (não significativo). Idêntico à Etapa 4.

### ❌ H2 — *Freeze Upper (C3) mitiga o Domain Shift* → **REFUTADA e invertida** (agora com evidência forte)
C3 vs C1 em T2: **Δ = −3.10 pp, p < 0.001** (Cohen's *d* = −4.32). Congelar o topo **piora** o domínio. Na Etapa 4 esse efeito era marginal (p = 0.074); com 6–8 seeds tornou-se **inequívoco**.

### ⚠️ Achado emergente da Etapa 4 (*C2 protege contra Domain Shift*) → **NÃO REPLICA**
Este é o ponto onde a Etapa 5 **corrige** a Etapa 4. Lá, C2 vs C1 em T2 dava Δ = +1.19 pp (p = 0.086, marginalmente significativo). Aqui, com mais seeds independentes:

> **C2 vs C1 em T2: Δ = +0.21 pp, p = 0.42 — não significativo.**

A vantagem de C2 sobre C1 no domínio **se dissolve** sob replicação. Conclusão revisada e mais honesta: **C2 (Freeze Lower) é estatisticamente equivalente à C1** em todas as células — nunca significativamente melhor, nunca pior. O valor real da C2 **não é "proteger contra Domain Shift"**, e sim **treinar ~50% menos parâmetros sem custo de desempenho** (eficiência, não regularização).

### Tabela-resumo da replicação (Etapa 4 → Etapa 5)

| Conclusão | Etapa 4 (3 seeds) | Etapa 5 (6–8 seeds) | Status |
|---|---|---|---|
| H1 (C2 mitiga língua) | Refutada (sem shift) | Refutada (sem shift) | ✅ Mantida |
| H2 (C3 mitiga domínio) | Refutada, p=0.074 | Refutada, **p<0.001** | ✅ Mantida e reforçada |
| C2 protege domínio | "Confirmado", +1.19 pp p=0.086 | +0.21 pp, p=0.42 | ❌ **Revertida** (não replica) |
| C4 é piso inviável | F1≈0.82 (estável) | F1≈0.65 (instável) | ✅ Mantida (e pior — §6.1) |

> **Configs viáveis (C1/C2/C3) replicam dentro de ≤ 0.75 pp** entre as duas etapas (ver `resultados/comparacao_etapa4_vs_etapa5.csv`). O núcleo das conclusões da Etapa 4 é robusto a seeds.

### 6.1 C4 (Frozen Encoder) não é um piso — é uma loteria de seed
Entre as seeds, o F1 de C4 varia de **0.36 a 0.81**; o desvio-padrão entre seeds é **≈ 0.098**, cerca de **16× maior** que o das configs viáveis (≈ 0.006). Com só a *head* linear treinável, o resultado depende fortemente da inicialização. A Etapa 4 (3 seeds afortunadas) viu ~0.82; com seeds independentes a média cai para ~0.65. **Lição:** *probing* linear do XLM-R nesta tarefa é **não-reprodutível**, e não apenas fraco. (Possível mitigação: LR maior ou mais épocas — fora do escopo, que fixa hiperparâmetros para isolar o congelamento.)

---

## 7. Passo A em aberto — correção para re-executar

O Passo A continua **sem resposta**: não sabemos se uma língua distante (JA/ZH) provoca *Language Shift* real. O bug (§2) está isolado e a correção é pequena. Substituir o carregador `carregar_marc` por uma versão que **valida a língua carregada** (e usa um repositório com config de língua confiável):

```python
# Correção do Passo A: garante que cada célula traz a língua pedida.
def carregar_marc(lang):
    """Carrega o split de teste do MARC e CONFIRMA a língua (evita o fallback mudo)."""
    erros = []
    for repo in ["mteb/amazon_reviews_multi", "amazon_reviews_multi"]:
        try:
            ds = load_dataset(repo, lang, split="test")   # config POSICIONAL de língua
        except Exception as e:
            erros.append(f"{repo} name={lang}: {str(e)[:90]}")
            continue
        cols = ds.column_names
        # valida: se houver coluna 'language'/'lang', tem de bater com `lang`
        lc = next((c for c in cols if c.lower() in ("language", "lang")), None)
        if lc is not None:
            langs = set(map(str, set(ds[lc])))
            assert langs == {lang}, f"{repo} devolveu línguas {langs}, esperava {{{lang}}}"
        return ds, repo, {"name": lang}
    raise RuntimeError("Falha ao carregar MARC por língua:\n" + "\n".join(erros))
```

Sanidade adicional, antes de treinar: `assert not df_ja_raw.equals(df_zh_raw)` (os DataFrames brutos de JA e ZH **não** podem ser iguais). Re-rodar só o loop de avaliação (os modelos C1–C4 já treinados podem ser reaproveitados) resolve o Passo A sem novo treino caro.

---

## 8. Conclusões da Etapa 5 (cientes das limitações)

1. **As conclusões da Etapa 4 são robustas a seeds.** Replicadas por 3 pessoas com 6–8 seeds independentes; C1/C2/C3 batem dentro de < 1 pp.
2. **H1 e H2 seguem refutadas** — H2 agora com evidência forte (p < 0.001), não marginal.
3. **Correção honesta:** o "achado emergente" da Etapa 4 (C2 protege contra Domain Shift) **não replica**; C2 ≡ C1 em desempenho. C2 vale pela **eficiência** (metade dos parâmetros treináveis), não por regularização.
4. **C4 é inviável e instável** — *probing* linear é loteria de seed (std ≈ 16× as demais).
5. **Passo A (língua distante) permanece em aberto** por um bug de carregamento reproduzido pelos três integrantes; a correção e o caminho de re-execução barata estão em §7.

> **Escopo.** O veredito de "sem *Language Shift*" continua restrito a **EN↔PT** (línguas próximas) e confundido com a assimetria de qualidade dos conjuntos (PT é *ground-truth*; EN é filtro ruidoso 90–94%). A pergunta que **refutaria** essa tese — uma língua tipologicamente distante — **ainda não foi respondida** porque o teste colapsou. Limitações completas em [`Metodologia_Rascunho.md` §11 e §2.1](Metodologia_Rascunho.md).

---

## 9. Artefatos gerados

| Arquivo | Conteúdo |
|---|---|
| `results_etapa5.csv` | Base combinada (182 linhas) — os 3 contribuidores |
| `resultados/etapa5_raw/*.csv` | CSVs brutos originais (proveniência) |
| `resultados/tabela_f1_media_desvio_etapa5.csv` | F1-macro média ± desvio (§3) |
| `resultados/tabela_deltas_etapa5.csv` | Δ-shifts por config (§4) |
| `resultados/tabela_testes_etapa5.csv` | Welch t, MWU, Cohen's *d* vs C1 (§5) |
| `resultados/comparacao_etapa4_vs_etapa5.csv` | Replicação independente (§6) |
| `resultados/heatmap_f1_macro_etapa5.png` | Heatmap config × célula (§3) |
| `resultados/barplot_deltas_etapa5.png` | Barplot dos Δ-shift (§4) |
| `src/analise_etapa5.py` | Script reprodutível de toda a análise |

> O conteúdo pronto para virar slides (Gamma) está em [`APRESENTACAO-Etapa5.md`](APRESENTACAO-Etapa5.md).
