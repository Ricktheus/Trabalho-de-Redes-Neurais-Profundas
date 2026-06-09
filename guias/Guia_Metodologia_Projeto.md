# Guia de Metodologia e Estruturação do Projeto
### "Do Zero ao Sucesso na Apresentação: Análise Arquitetural do Congelamento de Camadas em Transformers Multilíngues"

> **Documento para revisão rápida antes da apresentação.** Para estudo aprofundado, use `Guia_Estudo_e_Defesa.md`.

---

## 1. As Analogias: Entendendo os Conceitos do Zero

### A) O que é o XLM-RoBERTa?
*   **A Analogia:** O XLM-R é um **supertradutor poliglota** que trabalha em uma central internacional. Em vez de traduzir palavras literalmente, ele entende a *essência* de uma frase e a representa num "idioma mental universal" (o espaço vetorial compartilhado). "Bom" em português e "good" em inglês ficam no mesmo ponto desse mapa interno.
*   **Técnico:** Transformer pré-treinado em 100 idiomas. Versão base: 12 camadas, ~278M parâmetros.

### B) O que são as 12 Camadas do Encoder?
*   **A Analogia:** Uma **fábrica de processamento de texto com 12 andares**.
    *   **Andares 0–5 (baixos):** analisam gramática, ortografia e **em que idioma o texto está**. Não se importam com o sentido profundo.
    *   **Andares 6–11 (altos):** esquecem o idioma original e focam no **sentimento profundo, ironia, intenção do cliente**.
*   **Técnico (BERTology):** camadas rasas → sintaxe/léxico/idioma; camadas profundas → semântica/tarefa.

### C) O que é Language Shift vs Domain Shift?
*   **Language Shift:** mesma receita de bolo, mas o livro de receitas mudou de inglês para português.
*   **Domain Shift:** continua em português, mas saiu da cozinha (avaliando bolos) e foi para a mecânica (avaliando consertos). Vocabulário, intenção e critérios de avaliação mudaram completamente.

### D) O que é Layer Freezing?
*   **A Analogia:** Mandar os funcionários dos andares congelados **cruzarem os braços e usarem tampões de ouvido** durante o treinamento. Eles processam o texto normalmente, mas **não aprendem nada novo** — seus pesos ficam fixos do pré-treino.
*   **Técnico:** `requires_grad = False` nos parâmetros das camadas selecionadas. Forward pass normal; backward não atualiza esses pesos.

---

## 2. A Pergunta de Pesquisa em Linguagem Simples

> **"Se treinarmos o modelo em avaliações de PRODUTO em INGLÊS e testarmos em PORTUGUÊS em contextos de LOGÍSTICA, qual parte da rede sofre mais — e como o congelamento seletivo ajuda a evitar que o modelo se perca?"**

---

## 3. As 4 Configurações Experimentais

| Config | Nome | O que está congelado | O que testa |
|--------|------|----------------------|-------------|
| **C1** | Full Fine-Tuning | Nada | Linha de base (teto) |
| **C2** | Freeze Lower | Embeddings + camadas 0–5 | H1: mitiga Language Shift preservando alinhamento multilíngue |
| **C3** | Freeze Upper | Camadas 6–11 | H2: mitiga Domain Shift preservando semântica da tarefa |
| **C4** | Frozen Encoder | Todo o encoder | Linha de base (piso) — só a head treina |

**Seeds padronizadas:** `{42, 123, 2024}` — 3 por configuração = **12 treinos totais**.

---

## 4. Pipeline de Dados e Validação

1. **Filtragem por palavras-chave** (ver lista completa em `Metodologia_Rascunho.md` seção 5.2)
2. **Validação manual de 100 amostras** por subconjunto — critério de aceitação: ≥ 80% de precisão
3. **Declarar o resultado** abertamente na seção de Limitações do paper

---

## 5. Respostas Prontas para a Apresentação

**"Por que congelar os embeddings com as camadas iniciais em C2?"**
> *"Os embeddings guardam a representação léxica básica das palavras — é onde mora a informação de qual idioma o token pertence. Congelar embeddings + camadas 0–5 preserva o espaço vetorial compartilhado que o XLM-R construiu no pré-treino para 100 idiomas, evitando que o fine-tuning em inglês distorça esse alinhamento."*

**"Por que usar F1-Score Macro em vez de Accuracy?"**
> *"Datasets de e-commerce têm desbalanceamento natural: muito mais notas extremas do que intermediárias. Um classificador que chuta sempre 'positivo' teria accuracy alta. O F1-macro calcula separado para cada classe e faz a média, penalizando o modelo se ele ignorar a classe minoritária."*

**"Qual é o gap acadêmico do projeto?"**
> *"A literatura confirma que Domain Shift e Language Shift degradam performance, mas poucos estudos isolam os dois simultaneamente em Transformers multilíngues. Nossa lacuna é investigar como intervenções arquiteturais específicas (freezing de camadas distintas) afetam cada tipo de degradação de forma independente."*

**"O que vocês esperam como resultado?"**
> *"Nossa hipótese é que C2 (freeze lower) terá melhor desempenho mitigando Language Shift, pois preserva o alinhamento multilíngue das camadas baixas. C3 (freeze upper) deve mitigar melhor Domain Shift, pois preserva a semântica aprendida de sentimento em geral. C4 deve ser o pior em tudo."*
