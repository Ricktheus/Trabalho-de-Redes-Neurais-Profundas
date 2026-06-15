"""Etapa 4 — Análise estatística e visualização dos resultados (results.csv).

Consome o `results.csv` da Etapa 3 (48 linhas = 4 configs × 3 seeds × 4 testes)
e produz, sem depender de GPU:

  * Tabela pivot de F1-macro (média ± desvio) por config × cenário de teste.
  * Δ-shift isolando Domain Shift (T1−T2) e Language Shift (T1−T3) por config.
  * Testes estatísticos (Welch t e Mann-Whitney U) de C2/C3/C4 contra a
    baseline C1, com Cohen's d, nos cenários de domínio (T2) e língua (T3).
  * Veredito formal das hipóteses H1 e H2.
  * Gráficos: heatmap config×teste e barplot dos Δ-shift com barras de erro.

Mapeamento dos cenários (design 2×2, treino sempre em S1 = EN/Eletrônicos):
  T1 = S1_val  EN/Eletrônicos  (in-domain, in-language — baseline)
  T2 = S2      EN/Beleza       (Domain Shift)
  T3 = S3      PT/Eletrônicos  (Language Shift)
  T4 = S4      PT/Beleza       (Domain + Language)

Uso:  python src/analise_etapa4.py
"""

from __future__ import annotations

import os
import sys

# Console Windows (cp1252) não encoda Δ/±; força UTF-8 na saída.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")  # backend sem display, salva direto em arquivo
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Caminhos e rótulos
# ----------------------------------------------------------------------------
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "results.csv")
DIR_OUT = os.path.join(RAIZ, "resultados")
os.makedirs(DIR_OUT, exist_ok=True)

NOME_CONFIG = {
    "C1": "C1 · Full fine-tuning",
    "C2": "C2 · Freeze Lower (0-5)",
    "C3": "C3 · Freeze Upper (6-11)",
    "C4": "C4 · Frozen Encoder",
}
NOME_TESTE = {
    "T1": "T1 · EN/Elec (baseline)",
    "T2": "T2 · EN/Beleza (Domínio)",
    "T3": "T3 · PT/Elec (Língua)",
    "T4": "T4 · PT/Beleza (Ambos)",
}
ORDEM_C = ["C1", "C2", "C3", "C4"]
ORDEM_T = ["T1", "T2", "T3", "T4"]

sns.set_theme(style="whitegrid", context="talk")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d com desvio-padrão agrupado (pooled)."""
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


def main() -> None:
    df = pd.read_csv(CSV)
    print("=" * 78)
    print("ETAPA 4 — ANÁLISE ESTATÍSTICA E VISUALIZAÇÃO")
    print("=" * 78)
    print(f"\nDataFrame carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    assert df.shape[0] == 48, "Esperado 48 linhas (4 configs × 3 seeds × 4 testes)"
    print(f"Configs: {sorted(df.config.unique())}  |  Seeds: {sorted(df.seed.unique())}")
    print(f"Testes:  {sorted(df.teste.unique())}")

    # ------------------------------------------------------------------
    # Fase 4.1 — Tabela de agregação (média ± desvio sobre as 3 seeds)
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("FASE 4.1 — F1-macro: média ± desvio (3 seeds) por config × teste")
    print("-" * 78)

    media = df.pivot_table(index="config", columns="teste", values="f1_macro", aggfunc="mean")
    desvio = df.pivot_table(index="config", columns="teste", values="f1_macro", aggfunc="std")
    media = media.loc[ORDEM_C, ORDEM_T]
    desvio = desvio.loc[ORDEM_C, ORDEM_T]

    tabela = media.copy().astype(object)
    for c in ORDEM_C:
        for t in ORDEM_T:
            tabela.loc[c, t] = f"{media.loc[c, t]:.4f} ± {desvio.loc[c, t]:.4f}"
    print("\n", tabela.to_string())

    media.round(4).to_csv(os.path.join(DIR_OUT, "tabela_media_f1.csv"))
    desvio.round(4).to_csv(os.path.join(DIR_OUT, "tabela_desvio_f1.csv"))
    tabela.to_csv(os.path.join(DIR_OUT, "tabela_f1_media_desvio.csv"))

    # ------------------------------------------------------------------
    # Fase 4.2a — Δ-shift por config
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("Δ-SHIFT — queda de F1-macro vs. baseline T1 (em pontos percentuais)")
    print("-" * 78)
    linhas = []
    for c in ORDEM_C:
        t1, t2, t3, t4 = (media.loc[c, t] for t in ORDEM_T)
        linhas.append(
            {
                "config": c,
                "F1_T1_baseline": t1,
                "delta_dominio_T1_T2": (t1 - t2) * 100,
                "delta_lingua_T1_T3": (t1 - t3) * 100,
                "delta_ambos_T1_T4": (t1 - t4) * 100,
            }
        )
    deltas = pd.DataFrame(linhas).set_index("config")
    print("\n", deltas.round(3).to_string())
    deltas.round(4).to_csv(os.path.join(DIR_OUT, "tabela_deltas.csv"))

    # ------------------------------------------------------------------
    # Fase 4.2b — Testes estatísticos vs. baseline C1
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("FASE 4.2 — Significância vs. baseline C1 (Welch t + Mann-Whitney U)")
    print("            n=3 seeds por grupo | limiar p<0.10 (Metodologia)")
    print("-" * 78)

    def amostras(config: str, teste: str) -> np.ndarray:
        sub = df[(df.config == config) & (df.teste == teste)].sort_values("seed")
        return sub["f1_macro"].to_numpy()

    comparacoes = [
        ("T2", "C2", "Domain Shift — Freeze Lower vs Full"),
        ("T2", "C3", "Domain Shift — Freeze Upper vs Full (aposta H2)"),
        ("T2", "C4", "Domain Shift — Frozen Encoder vs Full"),
        ("T3", "C2", "Language Shift — Freeze Lower vs Full (aposta H1)"),
        ("T3", "C3", "Language Shift — Freeze Upper vs Full"),
        ("T3", "C4", "Language Shift — Frozen Encoder vs Full"),
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
        sig = "SIM" if p_t < 0.10 else "não"
        rows.append(
            {
                "cenario": teste,
                "comparacao": f"{cfg} vs C1",
                "descricao": desc,
                "delta_pp": round(dif_pp, 3),
                "welch_t": round(float(t_stat), 3),
                "p_welch": round(float(p_t), 4),
                "p_mannwhitney": round(float(p_u), 4),
                "cohen_d": round(float(d), 3),
                "magnitude": magnitude_d(d),
                "sig_p<0.10": sig,
            }
        )
    estat = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", None)
    print("\n", estat.to_string(index=False))
    estat.to_csv(os.path.join(DIR_OUT, "tabela_testes_estatisticos.csv"), index=False)

    # ------------------------------------------------------------------
    # Veredito das hipóteses
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("VEREDITO DAS HIPÓTESES (critério: Δ ≥ +3 pp E p<0.10)")
    print("-" * 78)

    # H1: C2 mitiga Language Shift (T3) -> F1(C2,T3) - F1(C1,T3) >= +3pp, p<0.10
    c2t3, c1t3 = amostras("C2", "T3"), amostras("C1", "T3")
    d_h1 = (c2t3.mean() - c1t3.mean()) * 100
    _, p_h1 = stats.ttest_ind(c2t3, c1t3, equal_var=False)
    # Pré-condição: existe Language Shift na baseline? (T1 - T3 em C1)
    shift_lingua_c1 = (media.loc["C1", "T1"] - media.loc["C1", "T3"]) * 100
    h1_ok = (d_h1 >= 3.0) and (p_h1 < 0.10)
    print(
        f"\nH1 (Freeze Lower mitiga Language Shift):"
        f"\n  Pré-condição — Language Shift na baseline C1 (T1−T3) = {shift_lingua_c1:+.2f} pp"
        f"  →  {'há queda' if shift_lingua_c1 > 0 else 'NÃO há queda (ganho zero-shot)'}"
        f"\n  Δ F1(C2,T3) − F1(C1,T3) = {d_h1:+.2f} pp  |  p = {p_h1:.4f}"
        f"\n  VEREDITO H1: {'CONFIRMADA' if h1_ok else 'REFUTADA'}"
    )

    # H2: C3 mitiga Domain Shift (T2)
    c3t2, c1t2 = amostras("C3", "T2"), amostras("C1", "T2")
    d_h2 = (c3t2.mean() - c1t2.mean()) * 100
    _, p_h2 = stats.ttest_ind(c3t2, c1t2, equal_var=False)
    shift_dominio_c1 = (media.loc["C1", "T1"] - media.loc["C1", "T2"]) * 100
    h2_ok = (d_h2 >= 3.0) and (p_h2 < 0.10)
    print(
        f"\nH2 (Freeze Upper mitiga Domain Shift):"
        f"\n  Pré-condição — Domain Shift na baseline C1 (T1−T2) = {shift_dominio_c1:+.2f} pp"
        f"  →  {'há queda' if shift_dominio_c1 > 0 else 'sem queda'}"
        f"\n  Δ F1(C3,T2) − F1(C1,T2) = {d_h2:+.2f} pp  |  p = {p_h2:.4f}"
        f"\n  VEREDITO H2: {'CONFIRMADA' if h2_ok else 'REFUTADA'}"
    )

    # Achado emergente: C2 mitiga Domain Shift (inverso do esperado)
    c2t2 = amostras("C2", "T2")
    d_emerg = (c2t2.mean() - c1t2.mean()) * 100
    _, p_emerg = stats.ttest_ind(c2t2, c1t2, equal_var=False)
    print(
        f"\nACHADO EMERGENTE (Freeze Lower protege contra Domain Shift):"
        f"\n  Δ F1(C2,T2) − F1(C1,T2) = {d_emerg:+.2f} pp  |  p = {p_emerg:.4f}"
        f"  →  {'SIGNIFICATIVO' if p_emerg < 0.10 else 'não significativo'} a p<0.10"
    )

    # ------------------------------------------------------------------
    # Fase 4.3 — Gráficos
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("FASE 4.3 — Gerando gráficos em resultados/")
    print("-" * 78)

    # (1) Heatmap config × teste
    hm = media.copy()
    hm.index = [NOME_CONFIG[c] for c in hm.index]
    hm.columns = [NOME_TESTE[t] for t in hm.columns]
    plt.figure(figsize=(11, 7))
    ax = sns.heatmap(
        hm,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=hm.values.min(),
        vmax=hm.values.max(),
        linewidths=0.5,
        cbar_kws={"label": "F1-macro (média de 3 seeds)"},
    )
    ax.set_title("F1-macro por Configuração × Cenário de Teste", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    f1 = os.path.join(DIR_OUT, "heatmap_f1_macro.png")
    plt.savefig(f1, dpi=150)
    plt.close()
    print(f"  [ok] {f1}")

    # (2) Barplot de Δ-shift (Domínio T1-T2 e Língua T1-T3) com barras de erro
    # erro propagado: std da diferença por seed
    def deltas_por_seed(config: str, ta: str, tb: str) -> np.ndarray:
        a = df[(df.config == config) & (df.teste == ta)].sort_values("seed")["f1_macro"].to_numpy()
        b = df[(df.config == config) & (df.teste == tb)].sort_values("seed")["f1_macro"].to_numpy()
        return (a - b) * 100

    dom_m, dom_s, lin_m, lin_s = [], [], [], []
    for c in ORDEM_C:
        dd = deltas_por_seed(c, "T1", "T2")
        dl = deltas_por_seed(c, "T1", "T3")
        dom_m.append(dd.mean())
        dom_s.append(dd.std(ddof=1))
        lin_m.append(dl.mean())
        lin_s.append(dl.std(ddof=1))

    x = np.arange(len(ORDEM_C))
    w = 0.38
    plt.figure(figsize=(11, 7))
    plt.bar(x - w / 2, dom_m, w, yerr=dom_s, capsize=5, label="Domain Shift (T1−T2)", color="#d1495b")
    plt.bar(x + w / 2, lin_m, w, yerr=lin_s, capsize=5, label="Language Shift (T1−T3)", color="#30638e")
    plt.axhline(0, color="black", lw=0.8)
    plt.xticks(x, ORDEM_C)
    plt.ylabel("Δ F1-macro vs. baseline T1 (pp)")
    plt.title("Queda de F1-macro sob Domain e Language Shift\n(valores > 0 = perda; < 0 = ganho zero-shot)", weight="bold")
    plt.legend()
    plt.tight_layout()
    f2 = os.path.join(DIR_OUT, "barplot_deltas.png")
    plt.savefig(f2, dpi=150)
    plt.close()
    print(f"  [ok] {f2}")

    # (3) Curvas de loss: dependem dos JSONs em resultados/runs/ (gerados na
    #     Etapa 3 no Drive). Geramos só se estiverem presentes localmente.
    dir_runs = os.path.join(DIR_OUT, "runs")
    if os.path.isdir(dir_runs) and any(f.endswith(".json") for f in os.listdir(dir_runs)):
        import json

        plt.figure(figsize=(11, 7))
        for fn in sorted(os.listdir(dir_runs)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(dir_runs, fn), encoding="utf-8") as fh:
                info = json.load(fh)
            hist = info.get("log_history", [])
            ep_tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "epoch" in h]
            ep_ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
            label = fn.replace("run_", "").replace(".json", "")
            if ep_tr:
                xs, ys = zip(*ep_tr)
                plt.plot(xs, ys, "--", alpha=0.6, label=f"{label} train")
            if ep_ev:
                xs, ys = zip(*ep_ev)
                plt.plot(xs, ys, "-o", alpha=0.8, label=f"{label} eval")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title("Curvas de Loss (treino vs. validação)", weight="bold")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        f3 = os.path.join(DIR_OUT, "curvas_loss.png")
        plt.savefig(f3, dpi=150)
        plt.close()
        print(f"  [ok] {f3}")
    else:
        print("  [skip] curvas_loss.png — resultados/runs/*.json não disponíveis "
              "localmente (estão no Drive da Etapa 3). Opcional/debug no plano.")

    print("\n" + "=" * 78)
    print("ETAPA 4 CONCLUÍDA — tabelas (.csv) e gráficos (.png) em resultados/")
    print("=" * 78)


if __name__ == "__main__":
    main()
