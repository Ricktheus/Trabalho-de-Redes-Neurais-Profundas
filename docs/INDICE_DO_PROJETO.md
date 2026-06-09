# 🗂️ Índice do Projeto — O que é cada arquivo e o que está atual

> **Projeto:** Análise Arquitetural do Congelamento de Camadas na Mitigação de *Domain Shift* e *Language Shift* em Transformers Multilíngues (XLM-RoBERTa).
>
> **Para que serve este arquivo:** mapa atualizado de **todos os arquivos do projeto**, dizendo o que é cada um e **se está atual ou não**. Leia isto antes de abrir qualquer outra coisa — várias documentações antigas ainda descrevem o desenho original (que mudou).
>
> **Última atualização:** 2026-06-08 (após conclusão da Etapa 1).

---

## ⚠️ Leia primeiro: o que MUDOU em relação ao plano original

O projeto **mudou de desenho** durante a Fase 2 (preparação de dados). Vários documentos antigos ainda descrevem a versão original. As decisões **atuais** são:

| Item | Versão ORIGINAL (em docs antigos) | Versão ATUAL (correta) |
|---|---|---|
| Domínios | Produto × **Logística** | Eletrônicos × **Beleza** |
| Filtro de domínio | palavra-chave nos 2 idiomas | EN por **keyword**, PT por **categoria** (`site_category_lv1`) |
| Balanceamento | `N = min` dos 4 (acoplado) | **desacoplado** (treino independente dos testes) |
| Auditoria do filtro | 100 amostras nos 4 subconjuntos | só **EN** (PT é ground-truth por categoria) |

> Se um documento fala de "logística", "produto × logística" ou "filtro por palavra-chave em português", ele está **desatualizado naquele ponto**.

---

## ✅ Arquivos ATUAIS (use estes)

| Arquivo | O que é | Status |
|---|---|---|
| **`Trabalho_RNP_Colab_Etapa1.ipynb`** | Notebook principal — Etapa 1 (preparação de dados) completa e executável no Colab. Gera os 5 CSVs (S1_train, S1_val, S2, S3, S4) e o módulo `src/data_pipeline.py`. | ✅ **Atual** — entregável principal |
| **`Metodologia_Rascunho.md`** (raiz) | Texto formal da metodologia (entregável). Reescrito p/ refletir Eletrônicos×Beleza, filtro EN-keyword/PT-categoria, balanceamento desacoplado, auditoria EN-only. | ✅ **Atual** — entregar este |
| **`DECISOES_E_DIFICULDADES.md`** | Relatório da trajetória do projeto: cada problema encontrado, o diagnóstico e a correção. Base para a Discussão do paper. | ✅ **Atual** |
| **`INDICE_DO_PROJETO.md`** | Este arquivo. | ✅ **Atual** |

---

## 🟡 Arquivos com conteúdo VÁLIDO mas EXEMPLOS DESATUALIZADOS

> Continuam úteis (conceitos, estrutura, perguntas de defesa), mas **os exemplos de domínio citam "Produto × Logística"**. Use o conteúdo conceitual, ignore/substitua os exemplos de domínio.

| Arquivo | O que é | O que está desatualizado |
|---|---|---|
| `Guia_Projeto.md` (raiz) | Guia educacional completo (do zero, com analogias). | Exemplos de domínio (Produto/Logística). Conceitos OK. |
| `guias/Guia_Estudo_e_Defesa.md` | Enciclopédia do grupo: 17 conceitos + 27 perguntas de defesa. | 8 menções a logística; ajustar exemplos p/ Eletrônicos/Beleza. |
| `guias/Guia_Metodologia_Projeto.md` | Versão compacta p/ revisar antes de apresentar. | Exemplos de domínio. |
| `guias/PLAN-congelamento-transformers.md` | Plano técnico (tasks, estrutura de pastas, critérios). Estrutura ainda válida. | Overview cita produto/logística; auditoria descrita como "manual" (virou zero-shot EN). |
| `guias/README.md` | Mapa antigo da documentação. | "Visão Geral" e tabela de decisões citam logística / validação manual. **Substituído por este índice.** |

---

## ❌ Arquivos OBSOLETOS (podem ser removidos)

| Arquivo | Por que está obsoleto | Ação sugerida |
|---|---|---|
| `auditoria_S1.csv` | Sobra da 1ª tentativa (domínio "produto", coluna `correto` vazia). Não é usado por nada. | **Deletar** |
| `guias/Metodologia_Rascunho.md` | **Duplicata antiga** da metodologia (versão Produto×Logística). Confunde com a da raiz. | **Deletar** (ou substituir pela da raiz) |

---

## 📚 Referências (permanentes, não mexer)

| Arquivo | O que é |
|---|---|
| `Proposta de tema.pdf` | Proposta inicial (Semana 1). Registro histórico do tema original. |
| `10 passos para estruturar papers.pdf` | Mensh & Kording — guia de estruturação de papers. Ler antes de redigir. |

---

## 🔜 Pendências / o que ainda falta no repositório

- [ ] **Salvar `00_Exploracao_Dominios.ipynb`** na pasta do projeto. Hoje ele só existe no Colab. É o notebook da Etapa 0 que justifica a escolha de domínio (achado: Beleza tem 2.372 negativos por categoria vs ~300 por keyword). **Recomendado entregar como notebook suplementar.**
- [ ] **Deletar** `auditoria_S1.csv` e `guias/Metodologia_Rascunho.md` (obsoletos).
- [ ] (Opcional) Atualizar os exemplos de domínio nos guias de estudo (Produto/Logística → Eletrônicos/Beleza) antes da defesa.
- [ ] **Etapa 2** — arquitetura XLM-RoBERTa + lógica de congelamento C1–C4 (ainda não iniciada).

---

## 🧭 "Qual arquivo abrir agora?"

| Situação | Abra |
|---|---|
| Entender o estado atual do projeto | **Este índice** + `DECISOES_E_DIFICULDADES.md` |
| Entregar a metodologia | `Metodologia_Rascunho.md` (raiz) |
| Rodar a preparação de dados | `Trabalho_RNP_Colab_Etapa1.ipynb` (Colab) |
| Estudar conceitos / treinar defesa | `guias/Guia_Estudo_e_Defesa.md` (lembrar: domínios = Eletrônicos×Beleza) |
| Começar a programar a Etapa 2 | `guias/PLAN-congelamento-transformers.md` |
