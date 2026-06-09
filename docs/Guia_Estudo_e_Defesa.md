# Guia de Estudo e Defesa
### Análise Arquitetural do Congelamento de Camadas em Transformers Multilíngues

> **Para quem é este documento:** para quem precisa entender os conceitos do projeto a fundo — não decorar — e para quem vai estar na frente do professor respondendo perguntas. Comece pela Parte 1 se você ainda tem dúvidas de vocabulário. Vá direto para a Parte 2 se está estudando para a apresentação.
>
> **Regra de ouro:** se você leu e não consegue explicar em voz alta sem olhar o guia, você ainda não entendeu. Releia, depois tente falar para um amigo.

---

## Como usar este guia

| Situação | O que fazer |
|----------|-------------|
| Tenho 1h+ para estudar | Leia a Parte 1 inteira, depois a Parte 2 |
| Tenho 30 min | Leia o Bloco 3 da Parte 1 e as perguntas com resposta escrita da Parte 2 |
| Tenho 10 min antes da apresentação | Leia só as respostas prontas (marcadas com 💬) da Parte 2 |
| Travei em alguma pergunta do professor | Ache a pergunta na Parte 2 e releia a analogia correspondente na Parte 1 |

---

# PARTE 1 — Vocabulário com Analogias

Os conceitos estão agrupados por tema, na ordem em que aparecem no projeto.

---

## Bloco A — O Modelo

### 1. Rede Neural

🍳 **Analogia:** uma linha de produção de fábrica. Entra matéria-prima (uma frase de texto) em um lado, sai produto final (uma classificação: positivo ou negativo) do outro. Entre os dois, várias estações fazem transformações sequenciais.

🧠 **Definição:** uma função matemática composta por camadas de operações com parâmetros (pesos) que são ajustados automaticamente para minimizar o erro nas previsões.

**No projeto:** todo o experimento é sobre manipular essa fábrica — especificamente, travar algumas estações durante o treinamento para ver o que acontece.

---

### 2. Transformer

🍳 **Analogia:** imagine uma fábrica onde cada trabalhador, antes de fazer o seu trabalho, **olha para o que todos os outros trabalhadores do mesmo andar estão produzindo**. Não é uma esteira simples; em cada andar, todo mundo se comunica com todo mundo antes de tomar uma decisão. Isso é o **self-attention** — cada palavra "presta atenção" em todas as outras palavras da frase ao mesmo tempo.

🧠 **Definição:** arquitetura de rede neural baseada em mecanismos de self-attention multi-head. Substituiu as redes recorrentes (LSTM/GRU) em praticamente tudo de NLP a partir de 2017.

**No projeto:** o XLM-RoBERTa que vocês vão usar é um Transformer com 12 andares (camadas).

---

### 3. Token e Embedding

🍳 **Analogia:** quando você digita "entrega atrasou", o modelo não enxerga letras. Primeiro ele quebra o texto em **pedaços** (tokens — mais ou menos sílabas ou palavras). Depois converte cada pedaço em uma **lista de números** (embedding). Pense assim: cada token é um ponto num mapa com 768 dimensões. Palavras parecidas ficam próximas nesse mapa.

🧠 **Definição:** tokenização é segmentar o texto em unidades discretas (no XLM-R: subpalavras via SentencePiece). Embedding é a representação vetorial densa dessas unidades.

**No projeto:** o tokenizer do XLM-R foi treinado para funcionar em 100 idiomas. Ele já sabe "partir" texto em português sem configuração extra. (Usamos truncate de 128 tokens por otimização de processamento).

---

### 4. XLM-RoBERTa

🍳 **Analogia:** o XLM-R é um **supertradutor poliglota** que trabalha numa central internacional. Em vez de traduzir palavra por palavra, ele entende a *essência* de uma frase e a representa num "idioma mental universal" (o espaço vetorial compartilhado). "Bom" em português e "good" em inglês ficam **no mesmo ponto** desse mapa interno — é por isso que o modelo treinado em inglês consegue funcionar em português.

🧠 **Definição:** modelo Transformer pré-treinado pela Meta em ~2,5 TB de texto de 100 idiomas usando Masked Language Modeling. Versão **base**: 12 camadas, 768 dimensões, ~278M parâmetros.

**No projeto:** vocês usarão o **xlm-roberta-base** (12 camadas, divisão limpa de 6+6, roda no Colab T4 gratuito).

---

### 5. Camada (Layer) e a Hierarquia das 12 Camadas

🍳 **Analogia (da fábrica de 12 andares):**
- **Térreo e andares baixos (camadas 0–5):** os funcionários analisam a gramática básica, a ortografia e **em que idioma o texto está escrito**. Eles não se importam com o sentido profundo — só com a estrutura da língua.
- **Andares do meio e topo (camadas 6–11):** os funcionários já esqueceram qual era o idioma original e focam em entender o **sentimento profundo, a intenção do cliente**. É aqui que o modelo decide "isso é positivo ou negativo".

🧠 **Definição (BERTology):** linha de pesquisa que estuda o que cada camada do BERT/XLM-R representa. Resultado geral: camadas rasas → sintaxe/léxico/idioma; camadas profundas → semântica/tarefa.

**No projeto:** essa divisão é a **base teórica da hipótese**. Se as camadas baixas guardam info de idioma, congelar elas durante o treino deveria preservar o alinhamento multilíngue original (Language Shift).

---

### 6. Classification Head (Cabeça de Classificação)

🍳 **Analogia:** o XLM-R é um intérprete super preparado, mas sem função específica. A *classification head* é a "crachá de trabalho" que você coloca nele — "agora você é um classificador de sentimento, sua saída é positivo ou negativo". É uma camada simples adicionada no topo.

🧠 **Definição:** camada Linear (768 → número de classes) que lê o vetor do token especial `<s>` e produz logits por classe.

**No projeto:** a *head* é **sempre treinável** — ela é nova, nascida com pesos aleatórios. O que vocês variam é se o **corpo** (o encoder) treina junto ou não.

---

## Bloco B — O Treinamento

### 7. Pré-treino vs Fine-tuning

🍳 **Analogia:**
- **Pré-treino** = mandar alguém estudar 4 anos lendo tudo que existe na internet. A pessoa vira "culta" de forma geral.
- **Fine-tuning** = pegar essa pessoa e treiná-la por 2 semanas no seu trabalho específico (ex: "atender SAC de e-commerce"). Ela já sabe o idioma, então aprende a tarefa rapidamente.

🧠 **Definição:** pré-treino é massivo, auto-supervisionado. Fine-tuning é o ajuste supervisionado rápido em uma tarefa específica.

**No projeto:** vocês **não fazem pré-treino** (levaria meses). Pegam o XLM-R pronto e fazem o fine-tuning para sentimento — e nesse passo, congelam certas partes (C1-C4).

---

### 8. Backpropagation e Frozen Layers

🍳 **Analogia:** depois que o produto sai errado da fábrica, você manda uma reclamação **de volta** pela linha de produção. Cada estação recebe a reclamação e ajusta um pouco o que faz. Quando a estação está **congelada**, ela recebe a reclamação mas **não muda nada**.

🧠 **Definição:** backpropagation é o algoritmo que calcula o gradiente da função de custo em relação aos parâmetros.

**No projeto:** congelar = editar o grafo computacional usando `requires_grad = False`.

---

### 9. Otimizador AdamW e Early Stopping

🍳 **Analogia:** o "gerente" decide quanto a estação ajusta a cada erro (AdamW com Learning Rate de 2e-5). O **Early Stopping** é o inspetor que para o treinamento: "se na próxima época a avaliação no teste separado de 10% (val') começar a piorar, paramos o treino para não viciar a máquina".

**No projeto:** a métrica acompanhada para parar cedo é a `eval_loss` na validação de 10%.

---

## Bloco C — O Experimento

### 10. Zero-Shot Cross-Lingual

🍳 **Analogia:** treinar uma pessoa em **inglês** para identificar avaliações de clientes, e colocar ela para trabalhar **em português** sem nunca ter visto um exemplo em português na vida.

🧠 **Definição:** transferir um modelo treinado na tarefa no idioma A para o idioma B, sem rótulos de B.

---

### 11. Domain Shift vs Language Shift

🍳 **Analogia:**
- **Language Shift:** O roteiro mudou de Inglês para Português, mas o tema continua sendo avaliar fones de ouvido e teclados. A "embalagem linguística" mudou.
- **Domain Shift:** O idioma continua Inglês, mas saímos da loja de eletrônicos para a loja de maquiagem. O vocabulário mudou, o que é um "produto bom" tem outras palavras (`skin`, `perfume` em vez de `battery`).

**No projeto:** Matriz 2x2 isolada:
- Treino base: **Eletrônicos em Inglês**.
- Teste T2: Beleza / Inglês (Domain Shift).
- Teste T3: Eletrônicos / Português (Language Shift).
- Teste T4: Beleza / Português (Combinado).

---

### 12. As 4 Configurações de Layer Freezing

**No projeto:** 
- **C1 (Full Fine-Tuning):** Nada congelado (baseline clássico).
- **C2 (Freeze Lower):** Congela Embeddings + Camadas 0–5.
- **C3 (Freeze Upper):** Congela Camadas 6–11.
- **C4 (Frozen Encoder):** Tudo congelado, exceto a head.

---

### 13. Seed (Semente Aleatória)

🍳 **Analogia:** se você usa a mesma semente invisível, o embaralhador de cartas sempre gera a mesma ordem de dados. Em machine learning garante reprodutibilidade da head aleatória e dos batches.

**No projeto:** **seeds = {42, 123, 2024}**, 3 rodadas por configuração. Total: 12 treinos independentes. Usamos isso para tirar a média.

---

## Bloco D — As Métricas

### 14. F1-Score Macro vs Accuracy

🍳 **Analogia:** numa prova com 3 questões fáceis e 1 muito difícil, a média simples (accuracy) esconde o fato de que a sala toda errou a questão 4. O F1-macro **dá o mesmo peso para a classe minoritária**. 

**No projeto:** Reviews 5 estrelas são maioria absoluta. A accuracy premiaria o modelo por sempre chutar Positivo. O F1-macro pune isso pesadamente.

### 15. Δ-Shift (Delta de Degradação)

🍳 **Analogia:** o Δ mede o **quanto você piorou** (F1 do controle menos F1 do teste difícil), e não o quanto você tirou em absoluto. Reduzir a "queda" significa que você mitigou o shift.

---

# PARTE 2 — Perguntas de Defesa

> As perguntas marcadas com 💬 têm resposta pronta para memorizar. As demais têm pistas.

---

## Grupo 1: Dados e Metodologia

**1. Como vocês separaram "Eletrônicos" de "Beleza" e que viés isso introduz?** 💬

> *"Para o português (B2W), usamos os metadados oficiais de categoria. No inglês (Amazon), aplicamos um filtro lexical com palavras-chave exclusivas (ex: `battery`, `usb` vs. `skin`, `perfume`). Esse método foi rápido e efetivo, como atestado por um modelo menor auditor, mas ele pode apresentar o viés de não captar avaliações onde o cliente não cita a palavra do produto diretamente. Mapeamos isso como uma limitação natural para manter a pureza das categorias isoladas."*

---

**2. Por que igualar o tamanho de todas as 4 células de teste (T1 a T4) mas não do Treino?** 💬

> *"Se o teste de Beleza tiver 800 exemplos e o de Eletrônicos 5000, não dá pra saber se a queda de desempenho foi pela mudança de domínio ou porque o tamanho da amostra introduziu variância. Por isso as 4 células de teste cravam exatamente em 2.680 avaliações cada. Ao mesmo tempo, não podíamos capar o conjunto de treino para 2.680, porque um bom fine-tuning demanda dados. Por isso usamos o máximo disponível no treino principal (S1) com cerca de 10.700 exemplos."*

---

**3. Como vocês evitaram vazamento de dados (data leakage) durante o Early Stopping?** 💬

> *"O erro comum é usar o próprio conjunto de teste do baseline (nosso T1) para calcular o early stopping e salvar o melhor modelo. Isso infla o F1 do T1 de forma otimista. Para evitar isso, extraímos 10% internamente do nosso próprio `S1_train` criando um sub-set `val'` isolado. Esse sub-set só serve para o Early Stopping. As 4 células de teste (T1-T4) nunca foram vistas por nenhum passo do treinamento."*

---

**4. Por que vocês só salvaram os CSVs e as curvas json e jogaram fora os pesos dos 12 modelos treinados?**

Pistas: A arquitetura do XLM-R gera checkpoints de quase 1,1 GB. Salvar todos consumiria 13 GB rapidamente, estourando os limites do ambiente gratuito do Drive no Colab. Mais importante ainda: nossa entrega não é colocar o modelo no Hugging Face (em produção), e sim **fazer a análise do comportamento arquitetural**. Extraímos as métricas e as curvas diretamente (*inline*) e o CSV é suficiente para o objetivo do artigo.

---

## Grupo 2: O Modelo

**5. O que é o XLM-R?** 💬

> *"O XLM-RoBERTa é um modelo Transformer pré-treinado pela Meta em 2,5 TB de texto de 100 idiomas simultâneos. Ele aprendeu a representar palavras de idiomas diferentes num mesmo espaço vetorial compartilhado — 'bom' em português e 'good' em inglês ficam em coordenadas próximas. Usamos a versão base: 12 camadas e cerca de 278 milhões de parâmetros."*

---

**6. Por que congelar os embeddings em conjunto com as camadas iniciais em C2?** 💬

> *"Os embeddings guardam a fundação do vocabulário, onde mora a informação forte do 'idioma' do token. Se no C2 (Freeze Lower) nós deixarmos os embeddings soltos durante o fine-tuning no inglês, o modelo pode distorcer todo esse espaço arrastando-o a favor da sintaxe inglesa, quebrando o alinhamento valioso que o XLM-R já trazia de fábrica. Ao travar eles com as camadas 0-5, 'preservamos a ponte' multilíngue para testar a H1."*

---

## Grupo 3: Hipóteses e Resultados Esperados

**7. Qual a sua hipótese principal e como você vai saber se ela foi confirmada?** 💬

> *"H1 testa o Language Shift: prevemos que congelar a base (C2) segura a degradação para o Português melhor que o Full Fine-Tuning (C1). H2 testa o Domain Shift: prevemos que congelar o topo (C3) mitiga o viés super-especializado de 'Eletrônicos' do que C1 faria, melhorando na hora de testar 'Beleza'. Validaremos estatisticamente pelas métricas do F1-Macro observando se C2 > C1 em T3 e se C3 > C1 em T2."*

---

**8. Por que 3 seeds e não apenas 1 rodada?** 💬

> *"Com uma seed só, a `classification_head` inicializa sempre de forma aleatória; se o modelo performar bem, pode ter sido 'sorte' na inicialização e não mérito arquitetural da camada congelada. Repetindo 3 vezes (4 configs x 3 seeds = 12 runs totais), conseguimos observar a média e desvio padrão para garantir que o resultado reflete o comportamento do XLM-R e não ruído pontual."*

---

**9. E se as hipóteses falharem e o C1 continuar sendo o melhor em tudo?**

Pistas: Responder com tranquilidade. Se as hipóteses falharem (resultado nulo), o estudo ainda tem altíssimo valor acadêmico. Ele pode provar que a teoria tradicional de "BERTology" — que diz que as camadas iniciais cuidam de idioma e as finais cuidam de semântica — é mais interligada na prática ou no XLM-R do que a teoria previa inicialmente. Falsificar uma hipótese de arquitetura com rigor metodológico é ciência.

---

## Grupo 4: Limitações da Pesquisa

**10. Quais as principais limitações do estudo de vocês?** 💬

> *"Assumimos 4 frentes: (1) O filtro lexical tem uma margem de viés (falsos negativos descartados); (2) Para não onerar o pipeline, executamos apenas 3 seeds ao invés de 10; (3) Simplificamos estrelas 1 e 2 como 'Negativo' e 4 e 5 como 'Positivo', excluindo o rating 3 (neutro/misto) por ser excessivamente ambíguo; (4) Empregamos apenas a versão 'base' do XLM-R; modelos 'large' com bilhões de parâmetros podem ter dinâmicas internas de freezing diferentes."*

---

**11. Qual a contribuição central do paper em uma frase?** 💬

> *"Demonstramos empiricamente como o congelamento seletivo de camadas sintáticas e semânticas do encoder XLM-R reage individualmente à mitigação do Language Shift (idioma) e Domain Shift (tópico) na classificação de sentimento zero-shot cross-lingual, isolando ambos numa matriz cruzada rigorosa (Eletrônicos x Beleza, EN x PT)."*

---

> **Dica final de apresentação:** Quando os professores avaliadores apertarem numa pergunta difícil, reformule em voz alta: *"Se entendi bem, a questão aponta sobre como garantimos X, correto?"*. Isso mostra domínio e te concede 5 preciosos segundos para elaborar a resposta. E lembre-se: respostas diretas baseadas na limitação que já assumimos (falsa expectativa de perfeição) bloqueiam debates excessivos e demonstram maturidade científica.
