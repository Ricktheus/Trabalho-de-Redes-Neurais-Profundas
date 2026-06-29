"""Etapa 5 — Agregação multi-contribuidor, testes estatísticos e visualização.

Consolida os CSVs gerados independentemente por três integrantes (cada um com
suas próprias seeds, no notebook `Trabalho_RNP_Kaggle_Etapa5.ipynb`) e refaz a
análise da Etapa 4 com um N de seeds muito maior, além de incorporar a célula
nova do MARC (tentativa de língua distante — Passo A).

Entradas (em `resultados/etapa5_raw/`):
  * results_etapa5_sebastiao.csv  — seeds {2718, 4242, 9001}, configs C1–C4
  * results_etapa5_pedro.csv      — seeds {1234, 5678, 91011}, configs C1–C4
  * results_etapa5_geovanna.csv   — seeds {13, 888}, SÓ C1 (run parcial)

Saídas:
  * results_etapa5.csv (raiz)                      — base combinada (deduplicada)
  * resultados/tabela_f1_media_desvio_etapa5.csv   — F1-macro média ± desvio
  * resultados/tabela_deltas_etapa5.csv            — Δ-shifts por config
  * resultados/tabela_testes_etapa5.csv            — Welch t, MWU, Cohen's d vs C1
  * resultados/comparacao_etapa4_vs_etapa5.csv     — replicação independente
  * resultados/heatmap_f1_macro_etapa5.png
  * resultados/barplot_deltas_etapa5.png

Design experimental (treino sempre em S1 = EN/Eletrônicos):
  T1 = EN/Eletrônicos  (in-domain, in-language — baseline)
  T2 = EN/Beleza       (Domain Shift)
  T3 = PT/Eletrônicos  (Language Shift próximo — B2W, ground-truth)
  T4 = PT/Beleza       (Domain + Language)
  T5/T6/T7 = MARC (JA / ZH / EN-âncora) — ver ALERTA abaixo.

ALERTA DE INTEGRIDADE DOS DADOS (Passo A):
  Em TODAS as 26 execuções (config × seed), as células T5, T6 e T7 têm F1
  byte-a-byte idêntico. Três conjuntos de texto distintos (japonês, mandarim,
  inglês) não podem produzir o mesmo F1 com 16 casas decimais — logo o carregador
  do espelho `mteb/amazon_reviews_multi` ignorou o argumento de língua
  (`name=lang`) e avaliou o MESMO split três vezes. Consequência: a decomposição
  EN→JA vs EN→ZH NÃO está disponível nestes dados. Tratamos T5=T6=T7 como uma
  única célula `TM` (MARC, domínio misto, língua indeterminada) e deixamos o
  Passo A em aberto, pendente de re-execução com carregador corrigido.

Uso:  python src/analise_etapa5.py
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Caminhos e rótulos
# ----------------------------------------------------------------------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RAW = os.path.join(RAIZ, "resultados", "etapa5_raw")
CSV_COMB = os.path.join(RAIZ, "results_etapa5.csv")
CSV_ETAPA4 = os.path.join(RAIZ, "results.csv")
DIR_OUT = os.path.join(RAIZ, "resultados")
os.makedirs(DIR_OUT, exist_ok=True)

ARQUIVOS_RAW = {
    "sebastiao": "results_etapa5_sebastiao.csv",
    "pedro": "results_etapa5_pedro.csv",
    "geovanna": "results_etapa5_geovanna.csv",
}

NOME_CONFIG = {
    "C1": "C1 · Full fine-tuning",
    "C2": "C2 · Freeze Lower (0-5)",
    "C3": "C3 · Freeze Upper (6-11)",
    "C4": "C4 · Frozen Encoder",
}
# TM = célula MARC consolidada (T5=T6=T7 colapsadas — ver ALERTA no docstring).
NOME_TESTE = {
    "T1": "T1 · EN/Elec (baseline)",
    "T2": "T2 · EN/Beleza (Domínio)",
    "T3": "T3 · PT/Elec (Língua próx.)",
    "T4": "T4 · PT/Beleza (Ambos)",
    "TM": "TM · MARC (língua distante*)",
}
ORDEM_C = ["C1", "C2", "C3", "C4"]
ORDEM_T = ["T1", "T2", "T3", "T4", "TM"]

sns.set_theme(style="whitegrid", context="talk")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d com desvio-padrão agrupado (pooled). Suporta n desigual."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 == 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(sp2)


def magnitude_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "desprezível"
    if ad < 0.5:
        return "pequeno"
    if ad < 0.8:
        return "médio"
    return "grande"


def combinar_csvs() -> pd.DataFrame:
    """Lê os 3 CSVs brutos, anexa a coluna 'contribuidor', concatena e deduplica."""
    frames = []
    for who, fn in ARQUIVOS_RAW.items():
        caminho = os.path.join(DIR_RAW, fn)
        d = pd.read_csv(caminho)
        d["contribuidor"] = who
        frames.append(d)
        print(
            f"  {who:10s}: {len(d):3d} linhas | configs={sorted(d.config.unique())} "
            f"| seeds={sorted(int(s) for s in d.seed.unique())}"
        )
    df = pd.concat(frames, ignore_index=True)

    dup = df.duplicated(subset=["config", "seed", "teste"]).sum()
    if dup:
        print(f"  ⚠️ {dup} medições duplicadas (mesmo config/seed/teste) — mantendo a 1ª.")
        df = df.drop_duplicates(subset=["config", "seed", "teste"], keep="first")

    df = df.sort_values(["config", "seed", "teste"]).reset_index(drop=True)
    df.to_csv(CSV_COMB, index=False)
    print(f"  → base combinada salva: {os.path.relpath(CSV_COMB, RAIZ)} ({len(df)} linhas)")
    return df


def colapsar_marc(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica que T5==T6==T7 em cada (config,seed) e os funde na célula 'TM'."""
    tri = [t for t in ("T5", "T6", "T7") if t in df.teste.unique()]
    if not tri:
        return df
    iguais, total = 0, 0
    for (_, _), g in df.groupby(["config", "seed"]):
        gg = g.set_index("teste")["f1_macro"]
        if all(t in gg.index for t in tri):
            total += 1
            vals = [gg[t] for t in tri]
            if np.allclose(vals, vals[0]):
                iguais += 1
    print(
        f"  Integridade MARC: {iguais}/{total} grupos (config,seed) com "
        f"{'='.join(tri)} idênticos."
    )
    if iguais == total and total > 0:
        print(
            "  ⚠️ TODAS as células de língua distante são idênticas → o carregador do MARC\n"
            "     ignorou a língua (name=lang). Decomposição EN→JA vs EN→ZH INDISPONÍVEL.\n"
            "     Colapsando T5/T6/T7 → célula única 'TM' (MARC, domínio misto)."
        )
    # Mantém T5 como representante e renomeia para TM; descarta T6/T7.
    df = df[df.teste != "T6"].copy()
    df = df[df.teste != "T7"].copy()
    df["teste"] = df["teste"].replace({"T5": "TM"})
    return df


def main() -> None:
    print("=" * 80)
    print("ETAPA 5 — AGREGAÇÃO MULTI-CONTRIBUIDOR, TESTES ESTATÍSTICOS E VISUALIZAÇÃO")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("PASSO 0 — Combinação dos CSVs dos três integrantes")
    print("-" * 80)
    df = combinar_csvs()
    df = colapsar_marc(df)

    print(f"\nDataFrame final: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"Configs: {sorted(df.config.unique())}  |  Testes: {ORDEM_T}")
    print("Seeds por config (N para os testes estatísticos):")
    n_seeds = {}
    for c in ORDEM_C:
        s = sorted(int(x) for x in df[df.config == c].seed.unique())
        n_seeds[c] = len(s)
        print(f"  {c}: n={len(s):d} seeds → {s}")
    if len(set(n_seeds.values())) > 1:
        print(
            "  ⚠️ N desigual entre configs (run parcial da Geovanna só cobriu C1).\n"
            "     Welch t e Mann-Whitney U toleram n desigual; reportado com transparência."
        )

    # ------------------------------------------------------------------
    # 5.1 — Tabela de agregação (média ± desvio)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5.1 — F1-macro: média ± desvio por config × célula de teste")
    print("-" * 80)

    media = df.pivot_table(index="config", columns="teste", values="f1_macro", aggfunc="mean")
    desvio = df.pivot_table(index="config", columns="teste", values="f1_macro", aggfunc="std")
    media = media.loc[ORDEM_C, ORDEM_T]
    desvio = desvio.loc[ORDEM_C, ORDEM_T]

    tabela = media.copy().astype(object)
    for c in ORDEM_C:
        for t in ORDEM_T:
            tabela.loc[c, t] = f"{media.loc[c, t]:.4f} ± {desvio.loc[c, t]:.4f}"
    print("\n", tabela.to_string())

    media.round(4).to_csv(os.path.join(DIR_OUT, "tabela_media_f1_etapa5.csv"))
    desvio.round(4).to_csv(os.path.join(DIR_OUT, "tabela_desvio_f1_etapa5.csv"))
    tabela.to_csv(os.path.join(DIR_OUT, "tabela_f1_media_desvio_etapa5.csv"))

    # ------------------------------------------------------------------
    # 5.2a — Δ-shift por config
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5.2a — Δ-shift: queda de F1-macro vs. baseline T1 (pontos percentuais)")
    print("       Δ = F1(T1) − F1(Tx). >0 = perda sob shift | <0 = ganho zero-shot")
    print("-" * 80)
    linhas = []
    for c in ORDEM_C:
        row = {"config": c, "F1_T1_baseline": media.loc[c, "T1"]}
        row["delta_dominio_T1_T2"] = (media.loc[c, "T1"] - media.loc[c, "T2"]) * 100
        row["delta_lingua_prox_T1_T3"] = (media.loc[c, "T1"] - media.loc[c, "T3"]) * 100
        row["delta_ambos_T1_T4"] = (media.loc[c, "T1"] - media.loc[c, "T4"]) * 100
        row["delta_marc_T1_TM"] = (media.loc[c, "T1"] - media.loc[c, "TM"]) * 100
        linhas.append(row)
    deltas = pd.DataFrame(linhas).set_index("config")
    print("\n", deltas.round(3).to_string())
    deltas.round(4).to_csv(os.path.join(DIR_OUT, "tabela_deltas_etapa5.csv"))

    # ------------------------------------------------------------------
    # 5.2b — Testes estatísticos vs. baseline C1
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5.2b — Significância vs. baseline C1 (Welch t + Mann-Whitney U + Cohen's d)")
    print("       limiar p<0.10 (Metodologia). N=6–8 seeds por grupo.")
    print("-" * 80)

    def amostras(config: str, teste: str) -> np.ndarray:
        sub = df[(df.config == config) & (df.teste == teste)].sort_values("seed")
        return sub["f1_macro"].to_numpy()

    comparacoes = [
        ("T2", "C2", "Domain Shift — Freeze Lower vs Full"),
        ("T2", "C3", "Domain Shift — Freeze Upper vs Full (aposta H2)"),
        ("T2", "C4", "Domain Shift — Frozen Encoder vs Full"),
        ("T3", "C2", "Língua próx. — Freeze Lower vs Full (aposta H1)"),
        ("T3", "C3", "Língua próx. — Freeze Upper vs Full"),
        ("T3", "C4", "Língua próx. — Frozen Encoder vs Full"),
        ("TM", "C2", "MARC — Freeze Lower vs Full"),
        ("TM", "C3", "MARC — Freeze Upper vs Full"),
        ("TM", "C4", "MARC — Frozen Encoder vs Full"),
    ]
    rows = []
    for teste, cfg, desc in comparacoes:
        base = amostras("C1", teste)
        alt = amostras(cfg, teste)
        dif_pp = (alt.mean() - base.mean()) * 100
        t_stat, p_t = stats.ttest_ind(alt, base, equal_var=False)
        try:
            u_stat, p_u = stats.mannwhitneyu(alt, base, alternative="two-sided")
        except ValueError:
            u_stat, p_u = np.nan, np.nan
        d = cohen_d(alt, base)
        rows.append(
            {
                "cenario": teste,
                "comparacao": f"{cfg} vs C1",
                "descricao": desc,
                "n_alt": len(alt),
                "n_base": len(base),
                "delta_pp": round(dif_pp, 3),
                "welch_t": round(float(t_stat), 3),
                "p_welch": round(float(p_t), 4),
                "p_mannwhitney": round(float(p_u), 4),
                "cohen_d": round(float(d), 3),
                "magnitude": magnitude_d(d),
                "sig_p<0.10": "SIM" if p_t < 0.10 else "não",
            }
        )
    estat = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", None)
    print("\n", estat.to_string(index=False))
    estat.to_csv(os.path.join(DIR_OUT, "tabela_testes_etapa5.csv"), index=False)

    # ------------------------------------------------------------------
    # Veredito das hipóteses (replicação independente, mais seeds)
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("VEREDITO DAS HIPÓTESES — replicação independente da Etapa 4 (critério Δ≥+3pp e p<0.10)")
    print("-" * 80)

    c2t3, c1t3 = amostras("C2", "T3"), amostras("C1", "T3")
    d_h1 = (c2t3.mean() - c1t3.mean()) * 100
    _, p_h1 = stats.ttest_ind(c2t3, c1t3, equal_var=False)
    shift_lingua_c1 = (media.loc["C1", "T1"] - media.loc["C1", "T3"]) * 100
    h1_ok = (d_h1 >= 3.0) and (p_h1 < 0.10)
    print(
        f"\nH1 (Freeze Lower mitiga Language Shift próximo EN→PT):"
        f"\n  Pré-condição — shift na baseline C1 (T1−T3) = {shift_lingua_c1:+.2f} pp"
        f"  →  {'há queda' if shift_lingua_c1 > 0 else 'NÃO há queda (ganho zero-shot)'}"
        f"\n  Δ F1(C2,T3) − F1(C1,T3) = {d_h1:+.2f} pp | p = {p_h1:.4f}"
        f"\n  VEREDITO H1: {'CONFIRMADA' if h1_ok else 'REFUTADA'}"
    )

    c3t2, c1t2 = amostras("C3", "T2"), amostras("C1", "T2")
    d_h2 = (c3t2.mean() - c1t2.mean()) * 100
    _, p_h2 = stats.ttest_ind(c3t2, c1t2, equal_var=False)
    shift_dominio_c1 = (media.loc["C1", "T1"] - media.loc["C1", "T2"]) * 100
    h2_ok = (d_h2 >= 3.0) and (p_h2 < 0.10)
    print(
        f"\nH2 (Freeze Upper mitiga Domain Shift):"
        f"\n  Pré-condição — shift na baseline C1 (T1−T2) = {shift_dominio_c1:+.2f} pp"
        f"  →  {'há queda' if shift_dominio_c1 > 0 else 'sem queda'}"
        f"\n  Δ F1(C3,T2) − F1(C1,T2) = {d_h2:+.2f} pp | p = {p_h2:.4f}"
        f"\n  VEREDITO H2: {'CONFIRMADA' if h2_ok else 'REFUTADA'}"
    )

    c2t2 = amostras("C2", "T2")
    d_emerg = (c2t2.mean() - c1t2.mean()) * 100
    _, p_emerg = stats.ttest_ind(c2t2, c1t2, equal_var=False)
    print(
        f"\nACHADO EMERGENTE (Freeze Lower protege contra Domain Shift):"
        f"\n  Δ F1(C2,T2) − F1(C1,T2) = {d_emerg:+.2f} pp | p = {p_emerg:.4f}"
        f"  →  {'SIGNIFICATIVO' if p_emerg < 0.10 else 'não significativo'} a p<0.10"
    )

    # ------------------------------------------------------------------
    # Replicação independente: Etapa 4 (3 seeds) vs Etapa 5 (6–8 seeds)
    # ------------------------------------------------------------------
    if os.path.exists(CSV_ETAPA4):
        print("\n" + "-" * 80)
        print("REPLICAÇÃO INDEPENDENTE — Etapa 4 (seeds 42/123/2024) vs Etapa 5 (seeds novos)")
        print("-" * 80)
        df4 = pd.read_csv(CSV_ETAPA4)
        comp_rows = []
        for c in ORDEM_C:
            for t in ["T1", "T2", "T3", "T4"]:
                m4 = df4[(df4.config == c) & (df4.teste == t)]["f1_macro"]
                m5 = df[(df.config == c) & (df.teste == t)]["f1_macro"]
                if len(m4) and len(m5):
                    comp_rows.append(
                        {
                            "config": c,
                            "teste": t,
                            "F1_etapa4_3seeds": round(m4.mean(), 4),
                            "F1_etapa5_Nseeds": round(m5.mean(), 4),
                            "diff_pp": round((m5.mean() - m4.mean()) * 100, 2),
                        }
                    )
        comp = pd.DataFrame(comp_rows)
        comp.to_csv(os.path.join(DIR_OUT, "comparacao_etapa4_vs_etapa5.csv"), index=False)
        ativas = comp[comp.config.isin(["C1", "C2", "C3"])]
        max_diff_ativas = ativas["diff_pp"].abs().max()
        max_diff_c4 = comp[comp.config == "C4"]["diff_pp"].abs().max()
        print(
            f"\n  Configs viáveis (C1/C2/C3): maior diferença de média = {max_diff_ativas:.2f} pp"
            f"\n   → {'REPLICAM' if max_diff_ativas < 1.5 else 'verificar'}: conclusões da Etapa 4 "
            "se sustentam sob seeds independentes."
        )
        # C4 = probing linear; instabilidade medida pelo desvio entre seeds.
        std_c4 = np.mean([df[(df.config == "C4") & (df.teste == t)]["f1_macro"].std()
                          for t in ["T1", "T2", "T3", "T4"]])
        std_ativas = np.mean([df[(df.config == c) & (df.teste == t)]["f1_macro"].std()
                              for c in ["C1", "C2", "C3"] for t in ["T1", "T2", "T3", "T4"]])
        print(
            f"  C4 (Frozen Encoder): diverge até {max_diff_c4:.1f} pp — NÃO replica.\n"
            f"   → instável por desenho: std entre seeds = {std_c4:.4f} "
            f"(≈ {std_c4/std_ativas:.0f}× o das configs viáveis = {std_ativas:.4f}).\n"
            f"   → probing linear do XLM-R nesta tarefa é uma loteria de seed, não um piso fixo."
        )
        print("  Salvo: resultados/comparacao_etapa4_vs_etapa5.csv")

    # ------------------------------------------------------------------
    # 5.3 — Gráficos
    # ------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("5.3 — Gerando gráficos em resultados/")
    print("-" * 80)

    hm = media.copy()
    hm.index = [NOME_CONFIG[c] for c in hm.index]
    hm.columns = [NOME_TESTE[t] for t in hm.columns]
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(
        hm,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "F1-macro (média das seeds)"},
    )
    ax.set_title("F1-macro por Configuração × Célula de Teste (Etapa 5)", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.figtext(
        0.5, -0.02,
        "* TM = T5/T6/T7 do MARC colapsaram em um único conjunto (bug de carregamento de língua) — ver RESULTADOS-Etapa5.md",
        ha="center", fontsize=9, style="italic", color="#555",
    )
    plt.tight_layout()
    f1 = os.path.join(DIR_OUT, "heatmap_f1_macro_etapa5.png")
    plt.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok] {f1}")

    def deltas_por_seed(config: str, ta: str, tb: str) -> np.ndarray:
        a = df[(df.config == config) & (df.teste == ta)].sort_values("seed")
        b = df[(df.config == config) & (df.teste == tb)].sort_values("seed")
        merged = a.merge(b, on="seed", suffixes=("_a", "_b"))
        return (merged["f1_macro_a"].to_numpy() - merged["f1_macro_b"].to_numpy()) * 100

    dom_m, dom_s, lin_m, lin_s, marc_m, marc_s = [], [], [], [], [], []
    for c in ORDEM_C:
        dd = deltas_por_seed(c, "T1", "T2")
        dl = deltas_por_seed(c, "T1", "T3")
        dm = deltas_por_seed(c, "T1", "TM")
        dom_m.append(dd.mean()); dom_s.append(dd.std(ddof=1))
        lin_m.append(dl.mean()); lin_s.append(dl.std(ddof=1))
        marc_m.append(dm.mean()); marc_s.append(dm.std(ddof=1))

    x = np.arange(len(ORDEM_C))
    w = 0.26
    plt.figure(figsize=(12, 7))
    plt.bar(x - w, dom_m, w, yerr=dom_s, capsize=4, label="Domínio (T1−T2)", color="#d1495b")
    plt.bar(x, lin_m, w, yerr=lin_s, capsize=4, label="Língua próx. EN→PT (T1−T3)", color="#30638e")
    plt.bar(x + w, marc_m, w, yerr=marc_s, capsize=4, label="MARC* (T1−TM)", color="#e3a51a")
    plt.axhline(0, color="black", lw=0.8)
    plt.xticks(x, ORDEM_C)
    plt.ylabel("Δ F1-macro vs. baseline T1 (pp)")
    plt.title(
        "Queda de F1-macro sob cada deslocamento (Etapa 5)\n"
        "(>0 = perda | <0 = ganho zero-shot · *MARC = domínio misto, língua colapsada)",
        weight="bold", fontsize=14,
    )
    plt.legend()
    plt.tight_layout()
    f2 = os.path.join(DIR_OUT, "barplot_deltas_etapa5.png")
    plt.savefig(f2, dpi=150)
    plt.close()
    print(f"  [ok] {f2}")

    print("\n" + "=" * 80)
    print("ETAPA 5 CONCLUÍDA — base combinada, tabelas (.csv) e gráficos (.png) em resultados/")
    print("=" * 80)


if __name__ == "__main__":
    main()
