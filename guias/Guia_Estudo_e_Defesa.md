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

**No projeto:** o tokenizer do XLM-R foi treinado para funcionar em 100 idiomas. Ele já sabe "partir" texto em português sem configuração extra.

---

### 4. XLM-RoBERTa

🍳 **Analogia (do Google IDE, é boa):** o XLM-R é um **supertradutor poliglota** que trabalha numa central internacional. Em vez de traduzir palavra por palavra, ele entende a *essência* de uma frase e a representa num "idioma mental universal" (o espaço vetorial compartilhado). "Bom" em português e "good" em inglês ficam **no mesmo ponto** desse mapa interno — é por isso que o modelo treinado em inglês consegue funcionar em português.

🧠 **Definição:** modelo Transformer pré-treinado pela Meta em ~2,5 TB de texto de 100 idiomas usando Masked Language Modeling. Versão **base**: 12 camadas, 768 dimensões, ~278M parâmetros.

**No projeto:** vocês usarão o **xlm-roberta-base** (12 camadas, divisão limpa de 6+6, roda no Colab T4 gratuito).

---

### 5. Camada (Layer) e a Hierarquia das 12 Camadas

🍳 **Analogia (da fábrica de 12 andares):**
- **Térreo e andares baixos (camadas 0–5):** os funcionários analisam a gramática básica, a ortografia e **em que idioma o texto está escrito**. Eles não se importam com o sentido profundo — só com a estrutura da língua.
- **Andares do meio e topo (camadas 6–11):** os funcionários já esqueceram qual era o idioma original e focam em entender o **sentimento profundo, a ironia, a intenção do cliente**. É aqui que o modelo decide "isso é positivo ou negativo".

🧠 **Definição (BERTology):** linha de pesquisa que estuda, por meio de *probing tasks*, o que cada camada do BERT/XLM-R representa. Resultado geral: camadas rasas → sintaxe/léxico/idioma; camadas profundas → semântica/tarefa.

**No projeto:** essa divisão é a **base teórica da hipótese**. Se as camadas baixas guardam info de idioma, congelar elas durante o treino deveria preservar o alinhamento multilíngue.

---

### 6. Classification Head (Cabeça de Classificação)

🍳 **Analogia:** o XLM-R é um intérprete super preparado, mas sem função específica. A *classification head* é a "crachá de trabalho" que você coloca nele — "agora você é um classificador de sentimento, sua saída é positivo ou negativo". É uma camada simples adicionada no topo.

🧠 **Definição:** camada Linear (768 → número de classes) que lê o vetor do token especial `<s>` (equivalente ao `[CLS]` do BERT) e produz logits por classe.

**No projeto:** a *head* é **sempre treinável** — ela é nova, não tem como funcionar congelada. O que vocês vão variar é se o **corpo** (o encoder) treina junto.

---

## Bloco B — O Treinamento

### 7. Pré-treino vs Fine-tuning

🍳 **Analogia:**
- **Pré-treino** = mandar alguém estudar 4 anos de faculdade lendo tudo que existe na internet. Ela vira "culta" de forma geral.
- **Fine-tuning** = pegar essa pessoa e treiná-la por 2 semanas no seu trabalho específico (ex: "atender SAC de e-commerce"). Ela já sabe português, então aprende a tarefa rapidamente.

🧠 **Definição:** pré-treino é o treinamento massivo, auto-supervisionado, em grande corpora. Fine-tuning é o ajuste supervisionado em uma tarefa específica, partindo dos pesos pré-treinados.

**No projeto:** vocês **não vão fazer pré-treino** (caro, levaria meses). Pegam o XLM-R já pronto e fazem só o fine-tuning para sentimento — e nesse fine-tuning, vão congelar partes da rede.

---

### 8. Backpropagation

🍳 **Analogia:** depois que o produto sai errado da fábrica, você manda uma reclamação **de volta** pela linha de produção. Cada estação recebe a reclamação e ajusta um pouco o que faz. Quando a estação está **congelada**, ela recebe a reclamação mas **não muda nada**.

🧠 **Definição:** algoritmo que calcula o gradiente da função de custo em relação a cada parâmetro, propagando o erro da saída para a entrada pela regra da cadeia.

**No projeto:** congelar = "parar a propagação aqui". Você está editando o grafo computacional explicitamente via `requires_grad = False`.

---

### 9. Cross-Entropy Loss (Função de Custo)

🍳 **Analogia:** uma régua que mede o "quão errado" o modelo estava. Se ele apostou 90% que era positivo e era positivo, o erro é pequeno. Se apostou 90% que era positivo e era negativo, o erro é enorme.

🧠 **Definição:** $L = -\sum_i y_i \log(\hat{y}_i)$, onde $y$ é o rótulo verdadeiro e $\hat{y}$ é a probabilidade predita. Mede a divergência entre as duas distribuições.

**No projeto:** é o que o otimizador vai minimizar. A "curva de loss" que vocês vão plotar é exatamente isso ao longo do treino.

---

### 10. Otimizador AdamW

🍳 **Analogia:** o "gerente" que decide quanto cada estação deve ajustar depois de cada reclamação. AdamW é um gerente esperto — lembra do histórico de ajustes (momentum), normaliza pelo ruído típico de cada peso (Adam) e ainda aplica um decaimento para evitar que os pesos cresçam demais (weight decay = o W do nome).

🧠 **Definição:** variante do Adam com weight decay desacoplado. Padrão para fine-tuning de Transformers.

**No projeto:** learning rate de `2e-5` — valor canônico para XLM-R. Não mude sem justificativa.

---

## Bloco C — O Experimento

### 11. Zero-Shot Cross-Lingual

🍳 **Analogia:** treinar uma pessoa em **inglês** para identificar avaliações positivas e negativas, e colocar ela para trabalhar **em português** sem nunca ter visto um exemplo em português. Ela não tem treino em português, mas como já sabe o idioma, "dá um jeito".

🧠 **Definição:** transferir um modelo treinado em uma tarefa no idioma A para aplicar a mesma tarefa no idioma B, sem usar nenhum dado rotulado de B no treino.

**No projeto:** essa é literalmente a configuração. Treina em inglês, testa em português. Sem usar rótulos em português.

---

### 12. Domain Shift vs Language Shift

🍳 **Analogia:**
- **Language Shift (mudança de idioma):** mesma receita de bolo, mas o livro de receitas mudou de inglês para português. Os ingredientes e o resultado são os mesmos — só a "embalagem linguística" mudou.
- **Domain Shift (mudança de assunto):** continua em português, mas saiu da cozinha (avaliando bolos) e foi para a mecânica (avaliando consertos de carro). O vocabulário é diferente, o que é "bom" é diferente, as queixas são diferentes.

🧠 **Definição:**
- **Language Shift:** mudança na distribuição léxico-sintática (idioma).
- **Domain Shift:** mudança na distribuição semântica/tópica (assunto, intenção, vocabulário de domínio).

**No projeto:** a matriz 2×2 isola isso:

| | Produto | Logística |
|---|---------|-----------|
| **Inglês** | Controle (nada) | Só Domain Shift |
| **Português** | Só Language Shift | Ambos juntos |

---

### 13. Layer Freezing (Congelamento de Camadas)

🍳 **Analogia:** sua fábrica de 12 andares. Você diz para os andares 0–5: "cruzem os braços, usem tampão de ouvido, não mudem nada durante o treinamento". Só os andares 6–11 se adaptam ao novo cliente. Eles ainda **processam** o texto (o forward pass continua), mas **não atualizam seus pesos** (o backward pass não os afeta).

🧠 **Definição:** marcar `requires_grad = False` nos parâmetros de camadas específicas durante o fine-tuning. O forward pass continua normalmente; o backward não atualiza esses parâmetros.

**No projeto:** as 4 configurações experimentais são variações do que está congelado:

| Config | Nome | O que congela |
|--------|------|---------------|
| C1 | Full Fine-Tuning | Nada (baseline alto) |
| C2 | Freeze Lower | Embeddings + camadas 0–5 |
| C3 | Freeze Upper | Camadas 6–11 |
| C4 | Frozen Encoder | Tudo (só treina a head) |

---

### 14. Seed (Semente Aleatória)

🍳 **Analogia:** quando você embaralha o baralho, a ordem das cartas depende de uma semente invisível. Se usar a mesma semente duas vezes, dá a mesma ordem. Em deep learning, a inicialização da *classification head*, a ordem dos batches e o dropout — tudo depende de seeds.

🧠 **Definição:** valor inicial dos geradores pseudo-aleatórios. Fixar a seed torna o experimento reprodutível (quase — GPU tem algum não-determinismo residual).

**No projeto:** **seeds = {42, 123, 2024}**, 3 rodadas por configuração. Isso permite reportar **média ± desvio padrão** e provar que o resultado vem da arquitetura, não de sorte. Total: 4 configs × 3 seeds = **12 treinos**.

---

## Bloco D — As Métricas

### 15. F1-Score Macro

🍳 **Analogia:** imagine uma prova com 3 questões onde 90% dos alunos acertam as questões fáceis (1 e 2) e só 10% acertam a difícil (3). A média simples (accuracy) faz parecer que a turma foi bem. O F1-macro **dá peso igual para cada questão** — então a difícil puxa a nota para baixo e revela o problema real.

🧠 **Definição:** média aritmética dos F1-Scores por classe, sem ponderar por frequência. F1 = média harmônica de precisão e recall.

**No projeto:** avaliações de e-commerce têm muito mais notas 5 e 1 do que notas 3. Se você usar accuracy, o modelo parece ótimo só acertando as classes majoritárias. F1-macro força a olhar para todas as classes igualmente.

---

### 16. Curva de Loss

🍳 **Analogia:** o "humor do modelo" ao longo do treino. O ideal é: loss de treino e de validação caem juntos. Quando o de validação **sobe enquanto o de treino cai**, o modelo começou a decorar os dados de treino em vez de aprender — isso é **overfitting**.

🧠 **Definição:** plot do valor da loss (treino e validação) em função da época ou do passo de treino.

**No projeto:** vocês vão gerar essas curvas para cada configuração (C1–C4) e compará-las. Se congelar camadas evita overfitting, as curvas de C2/C3 devem manter validação próxima do treino por mais tempo.

---

### 17. Δ-Shift (Delta de Degradação)

🍳 **Analogia:** se você tira 90 na prova de produto/inglês (seu ponto forte) e tira 70 na prova de logística/português (situação difícil), seu Δ = 20 pontos de queda. O Δ mede o **quanto você piorou**, não o quanto você foi bem em absoluto.

🧠 **Definição:** $\Delta(\text{shift}) = F1(\text{controle}) - F1(\text{cenário})$. Isola o impacto de cada tipo de shift independente do nível absoluto de performance.

**No projeto:** é a métrica principal de comparação entre configurações. A config que tiver menor Δ para Language Shift confirma H1; menor Δ para Domain Shift confirma H2.

---

# PARTE 2 — Perguntas de Defesa

> As perguntas marcadas com 💬 têm resposta pronta para memorizar (palavra por palavra). As demais têm pistas para você construir a resposta.

---

## Grupo 1: Motivação e Contexto

**1. Por que esse problema importa para uma empresa brasileira?** 💬

> *"Empresas como Mercado Livre, Magalu e iFood precisam classificar avaliações de clientes em português para monitorar satisfação em escala. Rotular dados em português é caro. A estratégia zero-shot promete resolver isso treinando em inglês — que tem dado de sobra — e aplicando diretamente em português. Nosso projeto mapeia por que essa estratégia falha e como mitigá-la estruturalmente."*

---

**2. Por que não usar tradução automática em vez de zero-shot?**

Pistas: tradução automática tem custo, latência, e introduz erros, especialmente em gírias e jargão de internet. Além disso, o XLM-R já aprendeu português — não faz sentido jogar fora esse conhecimento. A vantagem do zero-shot é ser mais rápido e mais robusto a variações regionais.

---

**3. Qual a diferença entre o que vocês fazem e simplesmente treinar o modelo em português?**

Pistas: treinar em português exigiria dados rotulados em português, que é exatamente o que queremos evitar. O custo de rotulação é o problema que motivou o projeto todo.

---

## Grupo 2: O Modelo

**4. O que é o XLM-R? Quantas camadas? Quantos parâmetros?** 💬

> *"O XLM-RoBERTa é um modelo Transformer pré-treinado pela Meta em 2,5 TB de texto de 100 idiomas simultâneos. Ele aprendeu a representar palavras de idiomas diferentes num mesmo espaço vetorial — 'bom' em português e 'good' em inglês ficam próximos nesse espaço. Usamos a versão base: 12 camadas, 768 dimensões, aproximadamente 278 milhões de parâmetros."*

---

**5. Como o XLM-R consegue funcionar em português sem ter sido fine-tunado para isso?**

Pistas: durante o pré-treino ele leu texto em português e alinhamento multilíngue emergiu naturalmente. Palavras com significado similar em idiomas diferentes ficam próximas no espaço vetorial compartilhado (espaço latente). É por isso que transferência cross-lingual funciona.

---

**6. Qual a diferença entre BERT e XLM-R?**

Pistas: BERT foi treinado só em inglês (e uma versão em inglês+alemão). XLM-R foi treinado em 100 idiomas com muito mais dados. XLM-R é baseado em RoBERTa (BERT com treinamento melhorado). Para tarefas multilíngues, XLM-R é o padrão.

---

**7. O que é self-attention em uma frase?**

Pistas: é o mecanismo que permite cada token da frase "olhar" para todos os outros tokens e decidir quais são relevantes para determinar seu significado no contexto. "Banco" em "fui ao banco sacar dinheiro" vs "sentei no banco do parque" — o self-attention usa o contexto para distinguir os dois.

---

## Grupo 3: Fine-Tuning e Congelamento

**8. O que muda matematicamente quando você "congela" uma camada?** 💬

> *"No PyTorch, marcamos `requires_grad = False` nos parâmetros dessa camada. Durante o forward pass, os dados continuam passando por ela normalmente. Durante o backward pass — quando o gradiente se propaga de volta para atualizar os pesos — essa camada é ignorada. Os pesos dela ficam exatamente como estavam no pré-treino. Ela contribui para o cálculo, mas não aprende nada novo."*

---

**9. Por que congelar embeddings junto com as camadas iniciais em C2?** 💬

> *"Os embeddings guardam a representação léxica básica das palavras — é onde mora a informação de qual idioma o token pertence. Se deixarmos os embeddings livres durante o fine-tuning em produto/inglês, o modelo pode distorcer esse espaço em direção ao inglês e perder o alinhamento multilíngue que o XLM-R construiu no pré-treino. Congelar embeddings junto com as camadas 0–5 preserva esse espaço compartilhado que é o que viabiliza a transferência cross-lingual."*

---

**10. Por que essa estratégia pode ajudar mais com Language Shift do que Domain Shift?**

Pistas: camadas baixas + embeddings representam o idioma. Congelá-las preserva o alinhamento inglês-português. Camadas altas representam a tarefa/semântica. Congelá-las (C3) preserva o que o modelo aprendeu sobre sentimento em geral — potencialmente ajudando a não over-especializar no domínio produto.

---

**11. Vocês testaram outros valores de learning rate por configuração?**

Pistas: não — fixamos LR = 2e-5 em todas as configurações intencionalmente para isolar a variável de freezing. Se mudássemos o LR por config, estaríamos comparando duas coisas ao mesmo tempo. Isso é uma limitação conhecida que declaramos explicitamente.

---

## Grupo 4: Dados

**12. Como vocês separaram "produto" de "logística"? Que viés isso introduz?** 💬

> *"Usamos filtragem por palavras-chave: uma avaliação é classificada como 'logística' se contém termos como 'delivery', 'shipping', 'entrega', 'atraso' e nenhum termo de produto, e vice-versa. Esse método é rápido, mas introduz viés: avaliações mistas como 'produto ótimo mas a entrega atrasou' podem cair em ambas as categorias e são descartadas. Para quantificar esse viés, amostramos 100 reviews de cada subconjunto e anotamos manualmente — encontramos [X]% de precisão, que declaramos abertamente na seção de limitações."*

---

**13. Por que igualar o tamanho dos subsets de teste?**

Pistas: se Logística/Inglês tem 800 exemplos e Produto/Português tem 5000, qualquer diferença de F1 pode vir do tamanho do conjunto e não do shift. Para a comparação ser válida, todos os 4 conjuntos de teste precisam ter N exemplos, onde N é o tamanho do menor.

---

**14. Por que truncar em 128 tokens?**

Pistas: avaliações de e-commerce são curtas — a maioria cabe em 128 tokens. Mostraremos o histograma de comprimentos na EDA para confirmar. Usar 512 (máximo do XLM-R) quadruplicaria o custo computacional sem ganho real.

---

**15. O Olist/B2W tem rótulo de sentimento ou só estrelas?**

Pistas: só estrelas (1–5). Mapeamos: 1–2 = negativo, 4–5 = positivo, 3 = descartado. Esse mapeamento é uma simplificação que declaramos como limitação. A nota 3 é genuinamente ambígua e incluí-la introduziria mais ruído do que sinal.

---

## Grupo 5: Métricas e Estatística

**16. Por que F1-macro e não accuracy?** 💬

> *"Datasets de avaliações de e-commerce têm desbalanceamento natural: muito mais notas extremas (1 e 5 estrelas) do que intermediárias. Um classificador que chuta sempre 'positivo' teria accuracy alta. O F1-macro calcula a média harmônica de precisão e recall para cada classe individualmente e depois faz a média aritmética — sem ponderar por frequência. Isso penaliza o modelo se ele ignorar a classe minoritária, forçando uma avaliação real do desempenho em todas as classes."*

---

**17. Por que 3 seeds e não 1?** 💬

> *"Modelos de deep learning inicializam a classification head com pesos aleatórios. Com apenas uma seed, não sabemos se um bom resultado vem da arquitetura ou da inicialização sortuda. Com 3 seeds diferentes, calculamos média e desvio padrão — se o desvio for pequeno, o resultado é estável e vem da configuração de freezing. Se for grande, a variância de inicialização domina e precisamos de mais seeds ou de uma conclusão mais cautelosa."*

---

**18. Como vocês decidem que uma diferença é "real" e não ruído?**

Pistas: usamos teste t pareado (ou Mann-Whitney U para n=3) entre pares de configurações. Reportamos p-valor e Cohen's d (tamanho do efeito). Com n=3 o threshold de significância é relaxado para p < 0.1. Se não houver significância, reportamos o tamanho do efeito e declaramos como resultado exploratório.

---

## Grupo 6: Resultados e Hipóteses

**19. Qual a sua hipótese principal e como você vai saber se ela foi confirmada?** 💬

> *"Temos duas hipóteses formais. H1: congelar as camadas iniciais (C2) mitiga mais o Language Shift do que o fine-tuning completo (C1) — especificamente, esperamos ver F1 maior de C2 no cenário Produto/Português. H2: congelar as camadas finais (C3) mitiga mais o Domain Shift — F1 maior no cenário Logística/Inglês. Confirmamos cada hipótese se a diferença for de pelo menos 3 pontos de F1-macro e com p < 0.1."*

---

**20. O que acontece se a hipótese for falsificada?**

Pistas: isso é um resultado igualmente válido. Significa que a separação léxico-sintático/semântico nas camadas do XLM-R não é tão limpa quanto a literatura de BERTology sugere, ou que Domain Shift e Language Shift estão mais entrelaçados do que o desenho experimental permite separar. Conclusão: freezing não é a solução para esse problema específico, e a literatura de BERTology pode não generalizar para cenários de domínio tão específicos quanto logística brasileira.

---

**21. Qual configuração vocês esperam que seja melhor em qual cenário?**

Pistas: C2 (freeze lower) para Language Shift; C3 (freeze upper) para Domain Shift. C1 provavelmente tem F1 mais alto no controle (Produto/Inglês). C4 (tudo congelado) provavelmente é o pior em todos os cenários — existe para mostrar o piso.

---

**22. Houve resultado surpreendente?**

Pistas: responda com o resultado real quando tiver. Se C3 tiver performado melhor do que C2 em Language Shift (contrário da hipótese), isso é interessante e vale investigar. Se C4 tiver performado inesperadamente bem, pode indicar que o XLM-R já resolve muito sem fine-tuning.

---

## Grupo 7: Limitações e Generalização

**23. Quais as principais limitações do estudo?** 💬

> *"Temos pelo menos quatro limitações explícitas. Primeiro, filtragem por palavras-chave tem precisão limitada — medimos isso manualmente. Segundo, rodamos apenas 3 seeds por restrição computacional, o que limita a robustez estatística. Terceiro, usamos XLM-R base e não large, então os resultados podem não generalizar para modelos maiores. Quarto, o domínio 'logística/inglês' é sub-representado na Amazon — a menor célula limita o tamanho total de teste."*

---

**24. Os resultados generalizam para outras tarefas de NLP?**

Pistas: não necessariamente. A conclusão é válida para classificação de sentimento em e-commerce. Para tarefas estruturalmente diferentes (NER, QA, tradução), o comportamento das camadas pode ser diferente. Isso é um trabalho futuro.

---

**25. Se tivessem mais 1 mês, o que fariam?**

Pistas: (a) mais seeds (5 a 10) para estatística mais robusta; (b) explorar granularidade de freezing mais fina (freeze 0–2, 0–4, etc.) para mapear a curva; (c) testar XLM-R large; (d) adicionar análise de atenção (visualizar o que o modelo está olhando em cada configuração); (e) incluir outros idiomas além de inglês-português.

---

## Grupo 8: O Paper

**26. Qual a contribuição central do paper em uma frase?** 💬

> *"Demonstramos empiricamente como o congelamento seletivo de camadas do encoder XLM-R afeta diferentemente a mitigação de Language Shift e Domain Shift em classificação de sentimento zero-shot cross-lingual, usando um desenho experimental 2×2 que isola os dois tipos de degradação simultaneamente."*

---

**27. Qual o gap da literatura que vocês preenchem?**

Pistas: a literatura confirma que Domain Shift e Language Shift degradam performance, mas poucos estudos isolam os **dois efeitos simultaneamente** e investigam como **intervenções arquiteturais específicas** (freezing de camadas distintas) afetam cada tipo de degradação de forma independente em modelos multilíngues.

---

> **Dica final de apresentação:** quando o professor fizer uma pergunta difícil, comece reformulando ela em voz alta — "Se entendi corretamente, você está perguntando sobre...". Isso te dá 5 segundos para pensar e mostra que você ouviu. Depois, use as analogias. Uma resposta com analogia é sempre mais convincente do que uma resposta só com jargão técnico.
