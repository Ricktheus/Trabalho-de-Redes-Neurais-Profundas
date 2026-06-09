# Guia do Projeto — Redes Neurais Profundas

> **Para quem é este guia:** para você, que sente que tem uma proposta legal nas mãos mas ainda não entende **direito** o que está acontecendo embaixo do capô. Esse documento começa do zero, com analogias do dia a dia, e te leva até o ponto em que você consegue **apresentar o projeto e responder perguntas com segurança**, sem decoreba.
>
> **Como usar:** leia linearmente uma vez. Depois volte aos capítulos específicos quando precisar. Cada seção termina com **"Você consegue explicar?"** — se a resposta for não, releia.
>
> **Convenções:**
> - 🧠 = conceito teórico
> - 🍳 = analogia do dia a dia
> - 🛠️ = ação concreta que você precisa fazer
> - ✅ = entregável / como saber que terminou
> - ❓ = pergunta que o professor pode te fazer
>
> ⚠️ **ATUALIZAÇÃO (pós-implementação):** os domínios mudaram de **Produto × Logística** para **Eletrônicos × Beleza**, e a filtragem em PT passou a usar **categoria do produto** (`site_category_lv1`), não palavra-chave. Onde este guia usar "produto/logística" como domínio, leia **eletrônicos/beleza**. Histórico completo em `DECISOES_E_DIFICULDADES.md`; índice dos arquivos em `INDICE_DO_PROJETO.md`.

---

## Sumário

0. [Como usar este guia](#parte-0--como-usar-este-guia)
1. [O projeto em uma página](#parte-1--o-projeto-em-uma-página)
2. [Vocabulário com analogias (começa do zero)](#parte-2--vocabulário-com-analogias-começa-do-zero)
3. [Análise crítica da sua proposta](#parte-3--análise-crítica-da-sua-proposta)
4. [O plano em fases — visão geral](#parte-4--o-plano-em-fases--visão-geral)
5. [FASE 1 — Metodologia (foco para amanhã)](#parte-5--fase-1--metodologia-foco-para-amanhã)
6. [FASE 2 — Aquisição e exploração dos dados](#parte-6--fase-2--aquisição-e-exploração-dos-dados-eda)
7. [FASE 3 — Baseline (modelo sem freezing)](#parte-7--fase-3--baseline-modelo-sem-freezing)
8. [FASE 4 — Experimentos de Layer Freezing](#parte-8--fase-4--experimentos-de-layer-freezing)
9. [FASE 5 — Análise de resultados](#parte-9--fase-5--análise-de-resultados)
10. [FASE 6 — Redação do paper](#parte-10--fase-6--redação-do-paper)
11. [FASE 7 — Apresentação final](#parte-11--fase-7--apresentação-final)
12. [Perguntas que você precisa saber responder](#parte-12--perguntas-que-você-precisa-saber-responder-defesa)
13. [Recursos para estudar](#parte-13--recursos-para-estudar)

---

## PARTE 0 — Como usar este guia

Você está na semana 2 de um projeto de Redes Neurais Profundas. A semana 1 era para definir tema (feito). A semana 2 é **metodologia** (entrega amanhã). Depois vêm dados, treinamento, análise e escrita.

Esse documento tem duas funções:

1. **Te ensinar do zero** o que está dentro da sua proposta. Cada termo técnico vai ser explicado com uma analogia do dia a dia, depois com a definição formal, e depois com **por que ele importa especificamente no seu projeto**.
2. **Te dar um plano executável** com tarefas concretas semana a semana, com entregáveis que dão para mostrar para o professor.

**Regra de ouro:** se você não consegue explicar um conceito sem olhar o guia, você ainda não entendeu. Volte e releia. Tente explicar para um amigo que não é da área. Se ele entender, você entendeu.

---

## PARTE 1 — O projeto em uma página

### O que o seu grupo vai investigar (em linguagem humana)

Empresas brasileiras como Mercado Livre, Magalu, iFood e Shopee precisam classificar avaliações de clientes (positiva, negativa, neutra) **em português**. Mas existe muito mais texto rotulado em inglês do que em português. Então o plano comum é:

> "Treino o modelo em inglês (que tem dado de sobra) e uso ele em português direto, sem treinar de novo."

Isso é chamado de **Zero-Shot Cross-Lingual**: zero exemplos do idioma-alvo durante o treino.

**Problema:** quando você faz isso na prática, o modelo cai feio na qualidade. A pergunta é: **por que ele cai?**

Existem duas razões possíveis (e elas se misturam):

- **(A) Mudança de idioma (Language Shift)** — o modelo viu inglês no treino e está vendo português no teste.
- **(B) Mudança de contexto/assunto (Domain Shift)** — o modelo aprendeu a julgar avaliações de **eletrônicos** (bateria, tela, conexão, carregador) e precisa julgar avaliações de **beleza** (perfume, pele, textura, fragrância), onde o vocabulário e o critério do que é "bom" mudam completamente.

Seu projeto vai responder: **qual desses dois é o maior culpado? E uma técnica chamada Layer Freezing ajuda mais a combater qual deles?**

### Por que isso é interessante

Se a resposta for "a maior culpa é do idioma", então a solução é investir em pré-treino multilíngue. Se a resposta for "a maior culpa é do contexto", então a solução é coletar exemplos do domínio-alvo (ex.: beleza), em qualquer idioma. É uma resposta que **muda decisões de produto** em empresas brasileiras. Isso é o que faz a pergunta ser boa.

### Você consegue explicar?

❓ Em uma frase, qual é o problema que motiva o projeto?
❓ Em uma frase, qual é a pergunta de pesquisa?
❓ Em uma frase, qual é a importância prática para uma empresa brasileira?

Se você travar em alguma, volte para o PDF da proposta e tente reformular com suas palavras.

---

## PARTE 2 — Vocabulário com analogias (começa do zero)

Aqui está o glossário **na ordem em que esses conceitos aparecem na vida do projeto**. Cada um tem: 🍳 analogia, 🧠 definição, e **por que importa**.

### 2.1 Rede Neural

🍳 **Analogia:** uma linha de produção de fábrica. Entra matéria-prima (uma frase) em um lado, sai produto final (uma classificação tipo "positivo") do outro. Entre os dois, várias estações fazem transformações sequenciais.

🧠 **Definição:** uma função matemática composta por camadas de operações lineares e não-lineares, com parâmetros (pesos) que são ajustados via gradiente para minimizar uma função de custo.

**Por que importa:** todo o projeto é sobre **manipular essa linha de produção**. Especificamente, **travar algumas estações** para ver o que acontece.

### 2.2 Camada (Layer)

🍳 **Analogia:** cada estação da linha de produção. Cada estação recebe o resultado da anterior, faz uma transformação, e passa para a próxima.

🧠 **Definição:** uma operação parametrizada (com pesos) que transforma um tensor de entrada em um tensor de saída.

**Por que importa:** o seu experimento literalmente consiste em **escolher quais estações podem aprender** (treinar) e quais ficam **com os pesos fixos** (congeladas).

### 2.3 Transformer

🍳 **Analogia:** imagine uma fábrica onde cada estação **conversa com todas as outras estações da mesma altura** antes de decidir o que fazer. Não é uma esteira simples linear; em cada andar, os trabalhadores olham para o que os colegas do mesmo andar estão produzindo. Isso é **self-attention**: cada token (palavra) "presta atenção" em todos os outros tokens.

🧠 **Definição:** arquitetura proposta em "Attention is All You Need" (2017) baseada em camadas de self-attention multi-head + redes feed-forward + normalização. Substituiu LSTMs/RNNs em quase tudo de NLP.

**Por que importa:** o XLM-R que você vai usar é um Transformer. Toda a literatura sobre "camadas iniciais aprendem sintaxe, camadas finais aprendem semântica" foi feita em cima desse tipo de arquitetura.

### 2.4 Token e Embedding

🍳 **Analogia:** quando você digita "entrega atrasou", o modelo não enxerga letras. Ele quebra em **pedaços** ("token" = pedaço, mais ou menos uma sílaba ou palavra) e converte cada pedaço em um **vetor numérico** (uma lista de números). Isso é o embedding. Pense: cada palavra é um ponto num mapa com centenas de dimensões.

🧠 **Definição:** tokenizar é segmentar texto em unidades discretas (no XLM-R, são subpalavras via SentencePiece). Embedding é a representação vetorial densa dessas unidades, aprendida durante o pré-treino.

**Por que importa:** o tokenizer do XLM-R foi treinado para funcionar **com 100 idiomas**, então ele consegue quebrar texto em português sem você fazer nada especial.

### 2.5 Pré-treino vs Fine-tuning

🍳 **Analogia:**
- **Pré-treino** = mandar uma pessoa estudar 4 anos de faculdade lendo tudo que existe na internet. Ela vira "culta" em geral.
- **Fine-tuning** = pegar essa pessoa culta e treinar por 2 semanas no seu emprego específico (ex: "atender chamada de SAC"). Ela já sabe português, então em pouco tempo aprende a tarefa nova.

🧠 **Definição:** pré-treino é o treinamento massivo, não supervisionado (ou auto-supervisionado), em grande corpora. Fine-tuning é o ajuste supervisionado em uma tarefa específica, partindo dos pesos pré-treinados.

**Por que importa:** o seu projeto **não vai fazer pré-treino** (caro, levaria meses). Você vai pegar o XLM-R já pré-treinado e fazer **só fine-tuning** para sentimento. E vai mexer no fine-tuning — **congelando partes** da rede.

### 2.6 XLM-R (XLM-RoBERTa)

🍳 **Analogia:** é o "candidato culto multilíngue" da analogia anterior. Foi treinado pelo Facebook (Meta) em 100 idiomas (inclusive português) lendo 2,5 TB de texto da internet.

🧠 **Definição:** modelo transformer baseado em RoBERTa, pré-treinado em CommonCrawl multilíngue (~2,5 TB) usando Masked Language Modeling. A versão **base** tem 12 camadas Transformer e ~278M de parâmetros. Existe também large (24 camadas, ~550M).

**Por que importa:** ele aprendeu a representar palavras de diferentes idiomas em um **espaço vetorial compartilhado**. "Bom" em português e "good" em inglês ficam **perto** nesse espaço. É isso que torna o zero-shot cross-lingual viável.

**Recomendação para o seu projeto:** use o **xlm-roberta-base**, não o large. O large é melhor mas precisa de GPU mais cara. Para um projeto de matéria, o base é suficiente e seus experimentos vão caber no Colab gratuito (com paciência).

### 2.7 Classification Head

🍳 **Analogia:** o XLM-R é como um "intérprete super preparado" mas sem profissão específica. A *classification head* é a "ficha de trabalho" que você coloca em cima dele — "agora você é um classificador de sentimento, sua saída vai ser positivo/negativo/neutro".

🧠 **Definição:** uma camada densa (Linear) adicionada no topo do encoder, normalmente lendo o vetor do token especial `[CLS]`, que produz logits para cada classe.

**Por que importa:** essa cabeça **sempre treina** (não tem como funcionar congelada — ela é nova). O que você vai variar é se o **corpo** (o encoder) treina junto ou não.

### 2.8 Zero-Shot Cross-Lingual

🍳 **Analogia:** treinar uma pessoa em **inglês** para identificar reviews positivas, e colocar ela para trabalhar **em português** sem nunca ter visto exemplo em português. A pessoa não tem treino em português, mas como já sabe português, dá um jeito.

🧠 **Definição:** transferir um modelo treinado em uma tarefa em um idioma A para aplicar a mesma tarefa em um idioma B, **sem usar nenhum exemplo rotulado em B no treino**.

**Por que importa:** é literalmente a configuração do seu experimento. Treina em inglês, testa em português. **Sem usar rótulos em português.**

### 2.9 Domain Shift vs Language Shift

🍳 **Analogia (Language Shift):** mesma receita de bolo, mas mudou o idioma do livro de receitas. Os ingredientes e o resultado são os mesmos, só a "embalagem linguística" mudou.

🍳 **Analogia (Domain Shift):** mesmo idioma, mas saiu da cozinha (avaliando bolos) e foi para a oficina (avaliando consertos de carro). O vocabulário é diferente, o que é "bom" é diferente, as queixas são diferentes.

🧠 **Definição:**
- **Language Shift**: a distribuição muda no nível **léxico/sintático** (idioma).
- **Domain Shift**: a distribuição muda no nível **semântico/tópico** (assunto, jargão, intenção).

**Por que importa:** essa é **a separação que sua pergunta de pesquisa quer isolar**. Treinando em Eletrônicos/Inglês, ao testar em:
- Eletrônicos/Inglês → controle (nem domain nem language shift)
- Eletrônicos/Português → só language shift
- Beleza/Inglês → só domain shift
- Beleza/Português → ambos juntos

Esse desenho é **bonito**, é o que dá força ao seu projeto.

### 2.10 BERTology

🍳 **Analogia:** é a "neurociência" dos modelos Transformer. Pesquisadores foram **abrindo a cabeça** do BERT/XLM-R para entender o que cada camada faz. A conclusão geral:

- **Camadas iniciais (1–4)**: aprendem coisas léxico-sintáticas (gramática, ordem das palavras, identidade do idioma).
- **Camadas do meio (5–8)**: aprendem semântica intermediária, relações entre palavras.
- **Camadas finais (9–12)**: aprendem semântica de alto nível ligada à tarefa específica.

🧠 **Definição:** linha de pesquisa em interpretabilidade que investiga, via probing tasks, o que cada camada do BERT/XLM-R representa.

**Por que importa:** essa é a **base teórica da sua hipótese**. Se camadas iniciais representam idioma, congelar elas durante o fine-tuning deveria **preservar o alinhamento multilíngue** — bom para combater Language Shift. Se camadas finais representam tarefa, congelar elas deveria **evitar over-especialização** no domínio do treino — bom para Domain Shift.

### 2.11 Layer Freezing (Congelamento de Camadas)

🍳 **Analogia:** sua fábrica tem 12 estações. Numa rodada de treinamento você diz: "estação 1 a 6, vocês ficam paradas, não mudem o que fazem". Só as estações 7 a 12 se adaptam ao novo cliente.

🧠 **Definição:** durante o fine-tuning, marcar `requires_grad = False` em parâmetros de certas camadas, impedindo que o backpropagation atualize seus pesos. O forward pass continua acontecendo normalmente; só o backward não modifica os parâmetros congelados.

**Por que importa:** é literalmente a variável independente do seu experimento. Você vai comparar 3 (ou mais) configurações: nada congelado (baseline), congelar camadas 1-6, congelar camadas 7-12.

### 2.12 Backpropagation

🍳 **Analogia:** depois que o resultado do produto sai errado, você manda uma reclamação **de volta** pela linha de produção. Cada estação recebe a reclamação e ajusta um pouco o que faz. Quando a estação está **congelada**, ela recebe a reclamação mas **não muda**.

🧠 **Definição:** algoritmo que computa o gradiente da função de custo em relação a cada parâmetro, propagando da saída para a entrada via regra da cadeia.

**Por que importa:** congelar = "parar a propagação aqui". Você está editando o **grafo computacional** explicitamente.

### 2.13 Função de custo: Cross-Entropy Loss

🍳 **Analogia:** uma régua que mede o "quão errado" o modelo estava. Se ele tinha 90% de certeza que era "positivo" e era "positivo", o erro é pequeno. Se ele tinha 90% de certeza que era "positivo" e era "negativo", o erro é gigante.

🧠 **Definição:** $L = -\sum_i y_i \log(\hat{y}_i)$, onde $y$ é o rótulo verdadeiro (one-hot) e $\hat{y}$ é a distribuição predita. Mede divergência entre a distribuição predita e a real.

**Por que importa:** é a função que o AdamW vai minimizar. Quando você plotar a "curva de loss", isso é o que está sendo plotado.

### 2.14 Otimizador: AdamW

🍳 **Analogia:** o "gerente" que decide **quanto cada estação deve ajustar** depois de cada reclamação. AdamW é um gerente esperto — ele lembra do histórico de ajustes (momentum), normaliza pelo "ruído" típico de cada peso (Adam), e ainda aplica um **decaimento** nos pesos para evitar que cresçam demais (weight decay = o W do nome).

🧠 **Definição:** variante do Adam com weight decay desacoplado (Loshchilov & Hutter, 2019). É o otimizador padrão para fine-tuning de Transformers.

**Por que importa:** você vai usar AdamW com learning rate baixinho (tipicamente `2e-5` ou `3e-5`). Esses são os valores "canônicos" que funcionam bem com BERT/XLM-R.

### 2.15 F1-Score Macro

🍳 **Analogia:** imagine uma prova com 3 questões. Se 90% dos alunos acertam as questões 1 e 2 (fáceis) e 10% acertam a 3 (difícil), a média **simples** (accuracy) faz parecer que a prova foi tranquila. O F1-macro **dá peso igual a cada questão** — então a 3 puxa a média para baixo, revelando o problema.

🧠 **Definição:** F1 = média harmônica de precision e recall. F1-macro = média **aritmética** dos F1 por classe, sem ponderar por frequência. Ideal para classes desbalanceadas.

**Por que importa:** suas avaliações vão ter **muito mais notas 5 e 1 do que notas 3** (clientes só comentam quando estão muito felizes ou muito chateados). Se você usar accuracy, o modelo pode parecer "ótimo" só acertando as classes majoritárias. F1-macro força você a olhar performance em todas as classes.

### 2.16 Curva de Loss

🍳 **Analogia:** um gráfico do "humor do modelo" durante o treino. Idealmente o **loss de treino cai e fica baixo** (modelo aprendeu) e o **loss de validação também cai e fica próximo do de treino** (modelo generaliza). Quando o de validação **sobe enquanto o de treino cai**, é **overfitting** — o modelo decorou.

🧠 **Definição:** plot do loss (treino e validação) em função do passo/época.

**Por que importa:** o enunciado da sua proposta diz que vai usar análise gráfica de loss para "identificar o momento em que a rede perde capacidade de generalização". Esse é o sinal visual que você vai procurar.

### 2.17 Seed (Semente Aleatória)

🍳 **Analogia:** quando você embaralha o baralho, a ordem das cartas depende de uma **semente** invisível. Se você usar a mesma semente duas vezes, dá a mesma ordem. Em deep learning, a inicialização dos pesos da classification head, a ordem dos batches, dropout — tudo depende de seeds.

🧠 **Definição:** valor inicial dos geradores pseudo-aleatórios. Fixar a seed torna o experimento determinístico (na prática, quase — ainda há não-determinismo de GPU em alguns ops).

**Por que importa:** **um único experimento com uma seed não vale como evidência científica**. A literatura recomenda rodar cada configuração com pelo menos **3 seeds diferentes** e reportar **média ± desvio padrão**. Isso é o que separa um trabalho rigoroso de um "achismo".

### Você consegue explicar?

❓ O que é a diferença entre pré-treino e fine-tuning?
❓ Por que o XLM-R consegue "entender" português sem ser treinado especificamente para isso?
❓ Por que congelar as camadas iniciais poderia ajudar mais com Language Shift do que com Domain Shift?
❓ Por que usar F1-macro e não accuracy?
❓ Por que rodar com várias seeds?

---

## PARTE 3 — Análise crítica da sua proposta

> 📌 **Nota (lida em retrospecto):** esta Parte foi escrita **antes da implementação**, sobre a proposta original (Produto × Logística). Vários alertas aqui — a dificuldade de montar a célula minoritária e o viés do filtro por palavra-chave — foram **exatamente o que motivou as mudanças** do projeto (domínios → Eletrônicos × Beleza; filtro PT → categoria; balanceamento → desacoplado). O texto é mantido por valor pedagógico; o desfecho de cada ponto está em `DECISOES_E_DIFICULDADES.md`.

### 3.1 O que está **bem forte**

✅ **A pergunta é boa.** É específica, falsificável e prática. Tem uma **resposta esperada** (a teoria sugere uma direção) mas o resultado pode contrariar — isso é o que faz pesquisa de verdade.

✅ **O desenho 2×2 (idioma × domínio) é elegante.** Você consegue **isolar variáveis**, que é raro em projetos de aluno. Esse é o ponto que vai impressionar.

✅ **Os datasets existem e são abertos.** Não vai precisar fazer scraping nem rotular dado novo.

✅ **A base teórica (BERTology) sustenta a hipótese.** Você não está chutando — está testando uma previsão da literatura.

✅ **Cabe no Colab.** XLM-R base com fine-tuning é viável em GPU T4 gratuita.

### 3.2 O que precisa de **atenção** (não é problema, é alerta)

⚠️ **A célula "Logística/Inglês" é a mais difícil de montar.** A Amazon Customer Reviews tem milhões de avaliações, mas a maioria fala de produto. As que falam de entrega são uma fração. Você vai precisar de uma **estratégia de filtragem por palavras-chave** que recupere uma quantidade razoável (idealmente milhares de exemplos). Se não der, você precisa decidir: (a) usar um dataset complementar (ex: reviews de Uber/Lyft em inglês), ou (b) reconhecer essa célula como limitação e diminuir o claim. **Resolva isso antes de seguir para a fase 2.**

⚠️ **Filtragem por palavra-chave introduz viés.** Filtrar "entrega" pega "entrega rápida ❤️" mas também "produto chegou e a entrega foi ok, mas o produto..." (na verdade fala do produto). Isso ataca a **pureza** da separação domínio. Você precisa de uma **avaliação manual de uma amostra** (ex: 100 reviews) para medir a precisão do seu filtro. Reportar isso na metodologia mostra rigor.

⚠️ **"Camadas 1-6" vs "Camadas 7-12" é uma binarização forte.** A literatura mostra que a transição syntax→semantics é gradual. Considere incluir uma terceira configuração: congelar **só os embeddings** (camada 0) ou **só as primeiras 3 camadas**. Não precisa explodir o número de experimentos, mas três pontos no espaço dá uma curva, não só uma comparação.

⚠️ **Múltiplas seeds.** A proposta não menciona. Sem isso o resultado é fraco estatisticamente. **Inclua na metodologia: cada configuração roda com 3 seeds.**

⚠️ **A "análise gráfica de curvas de loss" precisa de método.** "Olhar o gráfico" não é critério científico. Defina o que você vai procurar: divergência treino-validação, ponto de overfitting, comparação entre configurações.

⚠️ **Olist tem rótulo numérico (1-5), não sentimento direto.** Você precisa decidir como mapear: 1-2 = negativo, 3 = neutro, 4-5 = positivo? Ou binário (≤3 vs ≥4)? **Essa decisão deve estar na metodologia.** O mesmo vale para Amazon (estrelas).

⚠️ **Tamanho computacional.** XLM-R base + Olist + Amazon + 3 configurações × 3 seeds = 9 experimentos. Cada um leva ~30 min a 2h no Colab gratuito. **Faça uma estimativa de tempo antes** para não levar susto.

### 3.3 Refinamentos sugeridos (opcional, mas recomendado)

🔧 **Adicionar baseline:** **modelo congelado por inteiro** (só treina a classification head). Isso é o piso de performance — se você bater ele, freezing parcial está ajudando; se não bater, há algo errado.

🔧 **Adicionar baseline:** **fine-tuning completo (nada congelado)**. Isso é o teto típico. Se freezing parcial chegar perto ou melhorar, é uma descoberta interessante.

🔧 **Mover de "congela 1-6 vs 7-12" para um conjunto mais informativo:**
- (a) Tudo congelado (só head)
- (b) Congela 0-3 (embeddings + 4 primeiras)
- (c) Congela 0-6 (embeddings + 6 primeiras) ← a sua hipótese A
- (d) Congela 7-12 ← a sua hipótese B
- (e) Nada congelado (fine-tuning completo)

Com 5 configurações × 3 seeds × 1 cenário de treino = 15 fine-tunings. Cada um avaliado nas 4 células = 60 avaliações. Avaliação é rápida (uma passada forward). Treinos são o gargalo. **Discuta com o grupo o quanto está disposto a rodar.**

🔧 **Padronizar os tamanhos.** Para a comparação ser justa, todos os subsets (Produto/EN, Produto/PT, Logística/EN, Logística/PT) deveriam ter **o mesmo número de exemplos** por classe. Se Logística/EN só tem 800 exemplos, **subamostra os outros** para 800 também. Isso evita atribuir diferença de performance a tamanho de teste.

### 3.4 Veredito

A proposta está **acima da média de projetos de matéria**. Tem rigor de engenharia, motivação prática e fundamentação teórica. Os pontos de atenção são todos resolvíveis. **Não mude o tema.** Refine.

### Você consegue explicar?

❓ Por que a célula Logística/Inglês é a mais difícil?
❓ Por que usar 3 seeds e não 1?
❓ Por que adicionar um baseline "tudo congelado"?

---

## PARTE 4 — O plano em fases — visão geral

> **Premissa:** você está no fim da semana 2. Vou supor um cronograma de mais 10–12 semanas até a entrega final. Ajuste conforme o calendário real da sua matéria.

| Fase | Semana | Foco | Entregável |
|------|--------|------|------------|
| 0 | 1 | Tema definido | ✅ proposta de tema (já feita) |
| **1** | **2 (agora)** | **Metodologia** | **documento de metodologia (amanhã)** |
| 2 | 3 | Aquisição + EDA | datasets baixados, EDA com gráficos, filtros validados |
| 3 | 4–5 | Baseline | pipeline rodando, F1 do baseline em todas as 4 células |
| 4 | 6–7 | Experimentos freezing | matriz completa de resultados, com seeds |
| 5 | 8 | Análise | gráficos, testes estatísticos, interpretação |
| 6 | 9–10 | Redação do paper | draft completo |
| 7 | 11–12 | Revisão + apresentação | slides + ensaio |

**Princípio de cada fase:** entrega algo **mostrável**. Não acumular trabalho invisível. Mesmo que seja um notebook bagunçado, **prove progresso semanal**.

---

## PARTE 5 — FASE 1 — Metodologia (foco para amanhã)

> **Esta é a fase em que estamos.** O resto do guia detalha as próximas fases, mas **leia esta com muito mais cuidado** porque é a entrega de amanhã.

### Objetivo da fase

Produzir um documento que descreva **com rigor de engenharia** como vocês vão responder à pergunta de pesquisa. O professor precisa ler e pensar: "ok, se eles executarem isso, vão ter uma resposta defensável".

### O que uma boa metodologia tem

1. Pergunta de pesquisa (revisada/refinada)
2. Hipóteses formais (o que esperamos observar)
3. Arquitetura escolhida + justificativa
4. Pipeline de dados (origem, filtros, splits)
5. Protocolo de treino (otimizador, learning rate, batch size, épocas, early stopping)
6. **Protocolo de congelamento** (o coração do experimento)
7. Matriz de experimentos (todas as combinações que vão rodar)
8. Métricas e análise estatística (seeds, intervalos)
9. Limitações conhecidas
10. Cronograma de execução

### Sub-fases (passo a passo)

#### 1.1 Revisar a pergunta de pesquisa

🛠️ **Ação:** sentar com o grupo e **escrever a pergunta em uma única frase** que cabe em um tweet. A da sua proposta atual está boa, mas teste se todos do grupo conseguem repetir de cabeça. Se não, refine.

🛠️ **Ação:** formular **hipóteses falsificáveis** explícitas. Exemplo:
> **H1:** Congelar as camadas iniciais (0–6) do encoder XLM-R durante o fine-tuning reduz a queda de F1-macro entre Eletrônicos/Inglês e Eletrônicos/Português em pelo menos 3 pontos, comparado ao fine-tuning completo.
>
> **H2:** Congelar as camadas finais (7–12) reduz a queda entre Eletrônicos/Inglês e Beleza/Inglês em pelo menos 3 pontos, comparado ao fine-tuning completo.
>
> **H0:** Ambas as estratégias têm efeito estatisticamente equivalente.

✅ **Entregável:** 3 linhas de hipóteses, prontas para colar na metodologia.

❓ **O professor pode te perguntar:** "se sua hipótese fosse falsificada, o que você concluiria?" — saiba responder.

#### 1.2 Travar a arquitetura

🛠️ **Ação:** decidir **xlm-roberta-base** (recomendação). Justificar: balanceia capacidade vs custo computacional; é o modelo padrão da literatura para esse tipo de estudo; tem 12 camadas (facilita partição 6+6).

🛠️ **Ação:** descrever a classification head: camada Linear de 768 → número de classes, com dropout 0.1, lendo o vetor de saída do `[CLS]` (na verdade `<s>` no XLM-R).

✅ **Entregável:** parágrafo "Arquitetura-Alvo" pronto.

#### 1.3 Travar o número de classes

🛠️ **Ação:** decidir o esquema de rótulos. Recomendação:
- **Binário (positivo/negativo)**: descarta os 3 (neutro). Mais simples, classes mais balanceadas, mais comum na literatura zero-shot.
- **Ternário (negativo/neutro/positivo)**: usa tudo, mas o 3 é ambíguo.

Sugestão para um projeto de matéria: **binário** (estrelas 1–2 = negativo; 4–5 = positivo; descarta 3). Mais simples, mais defensável, e ainda é desafiador.

✅ **Entregável:** decisão registrada com justificativa.

#### 1.4 Travar a estratégia de freezing

🛠️ **Ação:** definir as configurações que vão rodar. Recomendação mínima:

| ID | Nome | Camadas congeladas |
|----|------|-------------------|
| C1 | Full fine-tune | nenhuma (baseline alto) |
| C2 | Freeze lower | embeddings + camadas 0–5 |
| C3 | Freeze upper | camadas 6–11 |
| C4 | Frozen encoder | embeddings + todas as 12 camadas (só treina a head) |

(O XLM-R base indexa camadas de 0 a 11.)

🛠️ **Ação:** decidir o esquema de freeze: vai congelar **embeddings junto** ou só as camadas? Recomendação: **congelar embeddings junto com as camadas iniciais** (faz sentido pela hipótese — embeddings são onde a info léxica mora).

✅ **Entregável:** tabela das configurações com indicação exata de quais módulos PyTorch ficam com `requires_grad = False`.

❓ **O professor pode te perguntar:** "por que congelar embeddings com as camadas iniciais e não separado?" — Resposta: porque embeddings são onde mora a representação léxica, que é exatamente o que a teoria diz que precisa ser preservado para zero-shot cross-lingual.

#### 1.5 Travar os datasets e os subsets

🛠️ **Ação:** especificar de onde virá cada subset:

| Subset | Origem | Idioma | Domínio | Uso |
|--------|--------|--------|---------|-----|
| S1 | Amazon (keyword de eletrônicos) | EN | Eletrônicos | **Treino + validação** |
| S2 | Amazon (keyword de beleza) | EN | Beleza | Teste |
| S3 | B2W (categoria de eletrônicos) | PT | Eletrônicos | Teste |
| S4 | B2W (categoria de beleza) | PT | Beleza | Teste |

🛠️ **Ação:** decidir tamanhos com **balanceamento desacoplado**: o S1 (treino) usa todo o seu pool balanceado e é dividido 80/20; só as **4 células de teste** vão a um N comum (N = min entre as células de teste). Isso evita que um compartimento de teste escasso rebaixe o treino. *(Realizado: treino ≈ 10,7k; cada célula de teste ≈ 2,7k.)*

🛠️ **Ação:** definir a **atribuição de domínio** (abordagem híbrida — ver metodologia):

- **EN (Amazon, por palavra-chave):**
  - *eletrônicos:* `battery`, `usb`, `charger`, `wifi`, `bluetooth`, `smartphone`, `laptop`, `tablet`, `headphone`, `smartwatch`, `phone`, `router`, `hdmi`
  - *beleza:* `skin`, `perfume`, `fragrance`, `cream`, `lotion`, `shampoo`, `makeup`, `cosmetic`, `serum`, `sunscreen`
- **PT (B2W, por categoria `site_category_lv1`):** eletrônicos = `Celulares e Smartphones` + `Informática e Acessórios` + `TV e Home Theater`; beleza = `Beleza e Perfumaria`.

🛠️ **Ação:** definir o protocolo de **validação do filtro**. Só o lado **EN** (palavra-chave) é auditado — por um classificador zero-shot sobre 100 amostras de cada subconjunto EN. O lado **PT** vem da categoria (ground-truth, ~100%).

✅ **Entregável:** seção "Datasets e Filtros" pronta com tabela, palavras-chave, e protocolo de validação.

#### 1.6 Travar o protocolo de treino

🛠️ **Ação:** definir hiperparâmetros e congelar (não vão mudar entre experimentos, para a comparação ser justa):

- **Otimizador:** AdamW
- **Learning rate:** `2e-5` (padrão Transformer)
- **Batch size:** 16 (vai depender da GPU; ajuste para não estourar memória)
- **Épocas:** 3 com early stopping baseado em loss de validação (paciência 1)
- **Max sequence length:** 128 tokens (reviews tendem a ser curtas; medir histograma na EDA confirma)
- **Weight decay:** 0.01
- **Warmup steps:** 10% do total
- **Loss:** Cross-Entropy
- **Seeds:** {42, 123, 2024} — 3 seeds por configuração

✅ **Entregável:** seção "Protocolo de Treino" pronta.

#### 1.7 Travar métricas e análise

🛠️ **Ação:** especificar métricas:
- **Primária:** F1-macro
- **Secundárias:** F1 por classe, accuracy, loss
- **Análise:** média ± desvio padrão sobre 3 seeds; teste de significância (Mann-Whitney U ou t-test pareado) entre configurações
- **Visualizações:** curvas de loss treino/validação por configuração; heatmap F1 (configuração × célula de teste); barplot de Δ (queda em relação a Eletrônicos/EN) por configuração e por shift

✅ **Entregável:** seção "Métricas e Análise" pronta.

#### 1.8 Listar limitações

🛠️ **Ação:** anteceder críticas. Liste honestamente:
- Filtragem por keyword introduz viés (precisão do filtro < 100%)
- Apenas 3 seeds (limite computacional)
- XLM-R base e não large
- Atribuição de domínio assimétrica: EN por palavra-chave (precisão ~90–94%), PT por categoria (ground-truth ~100%)
- Não vamos testar variação de learning rate por configuração
- Mapeamento de estrelas → sentimento é uma simplificação

✅ **Entregável:** seção "Limitações" pronta.

#### 1.9 Cronograma das próximas semanas

🛠️ **Ação:** colocar 1 linha por semana até a entrega final. Cole isso no documento — mostra ao professor que vocês planejaram.

✅ **Entregável:** cronograma.

#### 1.10 Costurar o documento final

🛠️ **Ação:** abrir um arquivo `Metodologia_Semana_2.docx` (ou `.md`) e cumprir as 10 seções acima. **Use o rascunho que está em `Metodologia_Rascunho.md`** que está no mesmo diretório — copia e cola, ajusta.

✅ **Entregável principal:** documento de metodologia para entregar amanhã.

### Você consegue explicar?

❓ Quais são suas hipóteses formais?
❓ Por que escolheram xlm-roberta-base e não large?
❓ Quantos experimentos vão rodar no total? Por quê?
❓ Por que congelar embeddings junto com camadas iniciais?
❓ Como vão garantir que a separação eletrônicos/beleza é confiável?

---

## PARTE 6 — FASE 2 — Aquisição e exploração dos dados (EDA)

### Objetivo

Ter os 4 subsets (S1–S4) baixados, filtrados, validados, e com **histogramas, contagens, exemplos** documentados.

### Sub-fases

#### 2.1 Baixar os datasets brutos

🛠️ **Ação:** baixar via `datasets` da HuggingFace e/ou Kaggle.
- Amazon Reviews: existem várias versões; recomendo a versão "amazon_polarity" ou um subset de "amazon_reviews_multi" (a versão multi tem inglês e português, atenção).
- Olist: Kaggle (Olist Brazilian E-Commerce). Tem CSV separado de reviews e de orders — o de orders tem `order_status`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date` que ajudam a flaggar "atraso" objetivamente.
- B2W Reviews: Kaggle, ~130k avaliações.

🛠️ **Ação:** salvar tudo em `/data/raw/`.

✅ **Entregável:** datasets locais, README do que tem em cada arquivo.

#### 2.2 EDA exploratória

🛠️ **Ação:** notebook que mostra para cada dataset:
- Número de linhas
- Distribuição de estrelas (gráfico de barras)
- Distribuição de tamanho de review em tokens (histograma)
- 10 exemplos aleatórios de cada classe
- % de duplicatas, % de valores nulos

✅ **Entregável:** notebook `01_EDA.ipynb` com gráficos e observações.

#### 2.3 Aplicar e validar filtros

🛠️ **Ação:** aplicar os keyword filters definidos na metodologia.
🛠️ **Ação:** amostrar 100 exemplos de cada classe filtrada, ler **manualmente**, marcar se a classificação está certa.
🛠️ **Ação:** calcular **precisão** do filtro. Se < 80%, refine as keywords.
🛠️ **Ação:** documentar resultados.

✅ **Entregável:** tabela com precisão do filtro por subset, observações sobre falsos positivos.

#### 2.4 Construir os splits finais

🛠️ **Ação:** com base no mínimo entre os 4 subsets, decidir N. Subamostra todos para N. Garantir balanceamento de classes (positivo/negativo iguais).

🛠️ **Ação:** salvar em `/data/processed/` como `S1.parquet`, `S2.parquet`, `S3.parquet`, `S4.parquet`.

🛠️ **Ação:** dividir S1 em 80/20 treino/validação.

✅ **Entregável:** 4 arquivos prontos para alimentar o modelo, com tamanhos e composição documentados.

### Você consegue explicar?

❓ Qual o tamanho de cada subset e por quê?
❓ Qual a precisão do seu filtro de keywords?
❓ Por que truncar em 128 tokens?

---

## PARTE 7 — FASE 3 — Baseline (modelo sem freezing)

### Objetivo

Ter um pipeline que treina XLM-R em S1 com fine-tuning completo (configuração C1) e avalia nas 4 células. Isso é o **ponto de referência** contra o qual tudo é comparado.

### Sub-fases

#### 3.1 Setup do ambiente

🛠️ **Ação:** Colab notebook com:
- `pip install transformers datasets accelerate evaluate`
- Verificar GPU (`torch.cuda.is_available()`)
- Fixar seed
- Mount Google Drive para salvar checkpoints

✅ **Entregável:** notebook `02_setup.ipynb`.

#### 3.2 Tokenização e DataLoader

🛠️ **Ação:** carregar `XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')`.
🛠️ **Ação:** função de tokenização que retorna `input_ids`, `attention_mask`, `labels`.
🛠️ **Ação:** DataLoader com batch 16, shuffle no treino.

✅ **Entregável:** módulo `data.py`.

#### 3.3 Loop de treino

🛠️ **Ação:** usar `Trainer` da HuggingFace (mais fácil) ou loop manual (mais flexível).
🛠️ **Ação:** configurar early stopping (paciência 1) por loss de validação.
🛠️ **Ação:** salvar curvas de loss em CSV ou TensorBoard.

✅ **Entregável:** módulo `train.py` que recebe configuração de freezing como parâmetro.

#### 3.4 Avaliação nas 4 células

🛠️ **Ação:** função `evaluate(model, dataset)` que retorna F1-macro, F1 por classe, accuracy, confusion matrix.
🛠️ **Ação:** rodar nas 4 células: S1-val, S2, S3, S4. Salvar resultados.

✅ **Entregável:** tabela JSON com resultados do baseline C1.

#### 3.5 Sanity check

🛠️ **Ação:** verificar que:
- F1 em S1-val é alto (>0.85 esperado). Se for muito baixo, há bug.
- F1 em S2/S3/S4 é menor que em S1-val. Se for igual ou maior, tem coisa estranha (pode ser data leak).
- O modelo não está prevendo só uma classe (olha confusion matrix).

✅ **Entregável:** seção "Sanity Check" no notebook com observações.

### Você consegue explicar?

❓ Por que avaliar em S1-val também e não só S2/S3/S4?
❓ O que seria suspeito no resultado?

---

## PARTE 8 — FASE 4 — Experimentos de Layer Freezing

### Objetivo

Rodar as configurações C2, C3, C4 (e mais variantes se decidirem), cada uma com 3 seeds, e avaliar nas 4 células.

### Sub-fases

#### 4.1 Implementar o congelamento

🛠️ **Ação:** função que recebe o modelo e uma lista de camadas para congelar, e seta `requires_grad = False` corretamente.

```python
def freeze_layers(model, layers_to_freeze):
    # layers_to_freeze: ex ["embeddings", "encoder.layer.0", ..., "encoder.layer.5"]
    for name, param in model.named_parameters():
        for prefix in layers_to_freeze:
            if name.startswith("roberta." + prefix):
                param.requires_grad = False
```

🛠️ **Ação:** depois do freeze, imprimir contagem de parâmetros treináveis para confirmar.

✅ **Entregável:** função `freeze_layers` testada.

#### 4.2 Rodar a matriz

🛠️ **Ação:** loop sobre {C2, C3, C4} × {seed 42, 123, 2024}. 9 treinos.

🛠️ **Ação:** para cada treino, salvar:
- Curvas de loss
- F1 final em S1-val, S2, S3, S4
- Checkpoint final (opcional, ocupa espaço)

✅ **Entregável:** CSV com 9 linhas × 4 colunas de métricas + metadados (config, seed).

#### 4.3 Backup e organização

🛠️ **Ação:** salvar no Drive com naming consistente. Ex: `results_C2_seed42.json`.

🛠️ **Ação:** consolidar tudo em um DataFrame `results.csv`.

✅ **Entregável:** `results.csv` com tudo.

### Você consegue explicar?

❓ Quantos treinos vocês rodaram? Por quê?
❓ Quanto tempo cada um demorou? (registrem)
❓ Vocês ficaram limitados por GPU? Como contornaram?

---

## PARTE 9 — FASE 5 — Análise de resultados

### Objetivo

Transformar a `results.csv` em **história**. Esse é o momento em que você descobre se sua hipótese se confirmou.

### Sub-fases

#### 5.1 Médias e desvios

🛠️ **Ação:** para cada configuração e cada célula, calcular média e desvio padrão dos 3 seeds.

✅ **Entregável:** tabela "Configuração × Célula" com média ± std.

#### 5.2 Visualizações

🛠️ **Ação:** três gráficos principais:

- **Heatmap**: configurações nas linhas, células nas colunas, F1-macro no valor (cor mais escura = melhor).
- **Barras com erro**: cada barra é uma configuração, agrupado por célula, com error bar de std.
- **Curvas de loss**: treino × validação por configuração, overlay para comparar.

🛠️ **Ação:** plotar **Δ-shift**: a queda de F1 entre S1-val e cada célula, por configuração.

✅ **Entregável:** notebook `03_results.ipynb` com gráficos.

#### 5.3 Testes estatísticos

🛠️ **Ação:** comparar pares de configurações (ex: C1 vs C2) nas mesmas células, com teste t pareado ou Mann-Whitney U. Reportar p-valores.

🛠️ **Ação:** interpretar com cautela — n=3 é pouco para significância robusta. Reporte o p e o tamanho do efeito (Cohen's d).

✅ **Entregável:** tabela com comparações.

#### 5.4 Interpretação

🛠️ **Ação:** responder explicitamente:

- H1 foi suportada? Quanto?
- H2 foi suportada? Quanto?
- Houve resultado surpreendente?
- O que a curva de loss revela?
- Se você fosse rodar de novo, o que mudaria?

✅ **Entregável:** texto "Interpretação dos Resultados" — vai para o paper.

### Você consegue explicar?

❓ O que aconteceu com sua hipótese?
❓ Qual configuração teve melhor performance em qual célula?
❓ A diferença observada é estatisticamente significativa?

---

## PARTE 10 — FASE 6 — Redação do paper

### Objetivo

Aplicar os "10 simple rules for structuring papers" que o professor passou.

### Estrutura esperada (seguindo o paper de Mensh & Kording)

1. **Título** — uma frase, comunica a contribuição central. Já tem um ok: refine.
2. **Abstract** — C-C-C (Context-Content-Conclusion). 200 palavras.
3. **Introdução** — funil de problemas: campo amplo → subcampo → gap específico → o que vocês fizeram.
4. **Trabalhos Relacionados** — BERTology, transferência cross-lingual, domain adaptation. ~10 referências.
5. **Metodologia** — já estará 90% pronta da Fase 1, mas vai precisar atualizar com decisões finais.
6. **Resultados** — uma sequência de afirmações apoiadas por figuras, cada uma "fechada" no final do parágrafo.
7. **Discussão** — como o gap foi preenchido, limitações, contribuição para o campo.
8. **Conclusão + Trabalhos Futuros** — 1 parágrafo.
9. **Referências**.

### Sub-fases

#### 6.1 Outline

🛠️ **Ação:** **antes de escrever** uma palavra, fazer um outline com **uma frase por parágrafo planejado**. Isso é a Regra 9 do paper que o professor passou.

✅ **Entregável:** outline aprovado pelo grupo.

#### 6.2 Primeiro draft

🛠️ **Ação:** preencher o outline. Não polir ainda.

✅ **Entregável:** draft v0.1.

#### 6.3 Revisar para C-C-C

🛠️ **Ação:** em cada seção, cada parágrafo deve começar com contexto, ter conteúdo no meio, conclusão no fim. Faça isso explicitamente.

✅ **Entregável:** draft v0.2.

#### 6.4 Revisar título e abstract

🛠️ **Ação:** título e abstract são os pedaços mais lidos. Tempo desproporcional aqui.

✅ **Entregável:** draft v0.3.

#### 6.5 Revisão por colegas

🛠️ **Ação:** dar para alguém de fora ler. Perguntar: você consegue resumir a contribuição em uma frase? Se não, está ruim.

✅ **Entregável:** draft v1.0.

### Você consegue explicar?

❓ Qual a contribuição central do seu paper, em uma frase?
❓ Qual o "gap" da literatura que vocês preenchem?

---

## PARTE 11 — FASE 7 — Apresentação final

### Objetivo

Conseguir defender o trabalho em ~15 minutos + perguntas, sem decoreba.

### Sub-fases

#### 7.1 Estrutura dos slides (10–12 slides)

1. Capa — título, autores
2. Motivação — por que isso importa (1 frase em destaque)
3. Pergunta de pesquisa — destacada
4. Background mínimo — XLM-R, zero-shot, freezing (1 slide)
5. Hipóteses — H1, H2, H0
6. Desenho experimental — diagrama da matriz 2×2
7. Configurações de freezing — tabela
8. Resultados principais — 1 ou 2 gráficos
9. Interpretação — qual hipótese ganhou?
10. Limitações
11. Trabalhos futuros
12. Conclusão — 1 frase

#### 7.2 Ensaio

🛠️ **Ação:** ensaiar pelo menos 2 vezes inteirinho. Cronometrar.
🛠️ **Ação:** preparar respostas para as perguntas da Parte 12 deste guia.

#### 7.3 Ensaio com perguntas

🛠️ **Ação:** pedir para alguém de fora fazer perguntas difíceis (use a Parte 12 deste guia). Treinar resposta.

✅ **Entregável:** apresentação pronta.

---

## PARTE 12 — Perguntas que você precisa saber responder (defesa)

> Use essa seção como checklist. Se você trava em alguma, volte ao capítulo correspondente.

### Sobre motivação e contexto

1. Por que esse problema importa para uma empresa brasileira?
2. Por que não usar tradução automática (Google Translate) em vez de zero-shot?
3. Quem se beneficiaria do resultado do seu estudo?

### Sobre o modelo

4. O que é o XLM-R? Quantas camadas? Quantos parâmetros?
5. Como o XLM-R consegue "entender" português sem ter sido fine-tunado para isso?
6. Qual a diferença entre XLM-R e BERT?
7. O que é multi-head self-attention, em uma frase?

### Sobre fine-tuning e congelamento

8. O que muda matematicamente quando você "congela" uma camada?
9. Por que congelar embeddings junto com as primeiras camadas?
10. Por que essa estratégia poderia preservar a transferência cross-lingual?

### Sobre dados

11. Como vocês separaram "eletrônicos" de "beleza" (EN por palavra-chave, PT por categoria)? Que viés isso introduz?
12. Qual a precisão do filtro?
13. Por que igualar tamanhos dos subsets?
14. Por que truncar em 128 tokens?

### Sobre métricas

15. Por que F1-macro e não accuracy?
16. Por que 3 seeds e não 1?
17. Como vocês decidiram que uma diferença é "real" e não ruído?

### Sobre resultados

18. Qual a configuração que mais ajudou no Language Shift?
19. Qual a configuração que mais ajudou no Domain Shift?
20. Houve interação? (ex: a melhor para uma é a pior para a outra)
21. A hipótese foi confirmada?

### Sobre limitações

22. O que vocês não puderam testar e por quê?
23. Se tivessem mais 1 mês, o que fariam?
24. O resultado generaliza para outras tarefas de NLP?

### Sobre o paper

25. Qual a contribuição central em uma frase?
26. Qual o gap da literatura?

> **Dica de ouro:** quando o professor fizer uma pergunta difícil, **comece reformulando a pergunta**. Isso te dá 5 segundos para pensar e mostra que você entendeu o que ele perguntou.

---

## PARTE 13 — Recursos para estudar

### Papers essenciais (leitura prioritária)

1. **"Attention is All You Need"** (Vaswani et al., 2017) — onde Transformer foi criado.
2. **"Unsupervised Cross-lingual Representation Learning at Scale"** (Conneau et al., 2020) — paper original do XLM-R.
3. **"A Primer in BERTology"** (Rogers et al., 2020) — survey do que sabemos sobre o interior do BERT.
4. **"To Tune or Not to Tune?"** (Peters et al., 2019) — sobre estratégias de fine-tuning incluindo freezing.

### Tutoriais

- HuggingFace Course (gratuito, em português parcialmente): cobre tokenização, fine-tuning, Trainer.
- "The Illustrated Transformer" (Jay Alammar, blog post) — explicação visual do mecanismo de atenção.

### Para o desenho experimental e escrita

- "Ten Simple Rules for Structuring Papers" (Mensh & Kording, 2017) — já está em mãos.
- Deep Learning Modeling Steps do Neuromatch — o link que o professor passou.

### Sobre o domínio (e-commerce brasileiro)

- Documentação do Olist no Kaggle (descrição dos arquivos).
- B2W Reviews descrição no Kaggle.

---

## Encerramento

Você agora tem em mãos:

1. Uma análise crítica do que vocês propuseram (Parte 3).
2. Um vocabulário completo, do zero, com analogias (Parte 2).
3. Um plano em 7 fases (Partes 4–11).
4. Detalhe granular da Fase 1 — a entrega de amanhã (Parte 5).
5. Uma bateria de 26 perguntas de defesa (Parte 12).
6. Recursos para estudar (Parte 13).

O segundo documento (`Metodologia_Rascunho.md`) é um **rascunho pronto para colar e ajustar** com base nas decisões que vocês tomarem na Parte 5.

> **Princípio de fechamento:** se em qualquer momento você está copiando algo deste guia sem entender, **pare**. Releia. Procure a analogia. Tente reformular em voz alta. **Você precisa ser dono do que escreve.**
