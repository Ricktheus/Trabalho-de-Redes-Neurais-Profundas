# Projeto de Redes Neurais Profundas
### "Análise Arquitetural do Congelamento de Camadas na Mitigação de *Domain Shift* e *Language Shift* em Transformers Multilíngues"

Este arquivo é o **mapa do tesouro** da documentação do projeto. Todo membro do grupo deve ler isso antes de abrir qualquer outro arquivo.

---

## 📌 Visão Geral do Projeto

Investigamos como o congelamento seletivo de camadas (*Layer Freezing*) no **XLM-RoBERTa-Base** mitiga duas fontes de degradação em classificação de sentimentos *Zero-Shot Cross-Lingual*:

1. **Language Shift:** o modelo treina em inglês e precisa inferir em português brasileiro.
2. **Domain Shift:** o modelo treina em avaliações de **produto** e precisa inferir sobre **logística e entrega**.

---

## ⚡ Qual arquivo abrir agora?

| Situação | Abra este arquivo |
|----------|-------------------|
| Preciso entregar a metodologia **amanhã** | [`Metodologia_Rascunho.md`](Metodologia_Rascunho.md) |
| Quero entender os conceitos / treinar para a defesa | [`Guia_Estudo_e_Defesa.md`](Guia_Estudo_e_Defesa.md) |
| Vou **apresentar em 10 minutos** e preciso revisar rápido | [`Guia_Metodologia_Projeto.md`](Guia_Metodologia_Projeto.md) |
| Vou **começar a programar** (semana 3 em diante) | [`PLAN-congelamento-transformers.md`](PLAN-congelamento-transformers.md) |

---

## 📂 Guia de Documentos

### 📄 Entregas e Rascunhos Acadêmicos

**[`Metodologia_Rascunho.md`](Metodologia_Rascunho.md)**
- *O que é:* O texto formal e científico da seção de metodologia (entrega Semana 2).
- *Como usar:* **Este é o arquivo para entregar ao professor.** Está no formato de paper científico, com hipóteses matemáticas ($H_1$, $H_2$, $H_0$), hiperparâmetros definidos, Δ-shift e plano estatístico (Mann-Whitney U + Cohen's d). Revise, cole no Word/LaTeX e entregue.
- *Gerado por:* Claude (Anthropic)

---

### 💡 Guias de Estudo e Preparação para Apresentações

**[`Guia_Estudo_e_Defesa.md`](Guia_Estudo_e_Defesa.md)**
- *O que é:* O guia mais completo do projeto. A enciclopédia particular do grupo.
- *Como usar:* Leia antes de qualquer apresentação ou quando travar num conceito. **Parte 1** explica 17 termos técnicos com analogias do dia a dia (Transformer, XLM-R, Layer Freezing, Domain/Language Shift, F1-macro, seeds…). **Parte 2** traz 27 perguntas que o professor pode fazer, com respostas prontas para memorizar nas mais críticas.
- *Gerado por:* Claude (Anthropic) — consolidado com o melhor das duas ferramentas

**[`Guia_Metodologia_Projeto.md`](Guia_Metodologia_Projeto.md)**
- *O que é:* Versão compacta e escaneável do guia de conceitos.
- *Como usar:* Leitura obrigatória **10 minutos antes de apresentar**. Contém as 4 analogias principais (supertradutor poliglota, fábrica de 12 andares, etc.), a tabela das 4 configurações e respostas prontas para as perguntas mais prováveis da banca.
- *Gerado por:* Google Antigravity IDE (Gemini)

---

### 🛠️ Planejamento de Código e Execução

**[`PLAN-congelamento-transformers.md`](PLAN-congelamento-transformers.md)**
- *O que é:* O plano de desenvolvimento técnico como projeto de software.
- *Como usar:* Abra quando for para o VS Code ou Colab. Define a estrutura de pastas do repositório Git, quais módulos Python criar (`data_pipeline.py`, `model.py`, `train.py`, `utils.py`), e as tasks de cada etapa com INPUT, OUTPUT e critério de VERIFY — inclusive testes unitários para confirmar que o congelamento está correto antes de rodar os 12 treinos.
- *Gerado por:* Google Antigravity IDE (Gemini) — seeds atualizadas para `{42, 123, 2024}`

---

### 📚 Referências do Professor

**`Proposta de tema.pdf`** — proposta inicial entregue na Semana 1. Ponto de partida do projeto.

**`10 passos para estruturar papers.pdf`** — artigo de Mensh & Kording sobre estruturação de papers científicos. Leitura obrigatória antes de redigir o paper final.

---

## ⚙️ Decisões Padronizadas (não alterar sem consenso do grupo)

| Decisão | Valor |
|---------|-------|
| Modelo | `xlm-roberta-base` — leve, roda no Colab T4 gratuito |
| Seeds | `{42, 123, 2024}` — 3 por configuração |
| C1 — Full Fine-Tuning | Nada congelado (teto de referência) |
| C2 — Freeze Lower | Embeddings + camadas 0–5 (testa $H_1$ — Language Shift) |
| C3 — Freeze Upper | Camadas 6–11 (testa $H_2$ — Domain Shift) |
| C4 — Frozen Encoder | Todo o encoder congelado (piso de referência) |
| Esquema de rótulos | Binário: 1–2 = negativo, 4–5 = positivo, 3 = descartado |
| Métrica primária | F1-macro + Δ-shift |
| Critério do filtro | ≥ 80% de precisão (validação manual de 100 amostras) |
| Total de treinos | **12** (4 configs × 3 seeds) |
| Total de avaliações | **48** (12 treinos × 4 cenários de teste) |
