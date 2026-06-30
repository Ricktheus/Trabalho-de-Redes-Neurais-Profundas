# Roteiro de apresentação — falas por slide (5 apresentadores)

Texto para ser **falado** em cada slide (não é o conteúdo do slide, é a narração).
Divisão em blocos temáticos coerentes, ~3 slides por pessoa. Substituam
"Pessoa 1..5" pelos nomes reais do grupo. Tempo estimado: ~8 a 10 minutos.

| Apresentador | Slides | Bloco |
|---|---|---|
| **Pessoa 1** | 1, 2, 3 | Abertura e o problema |
| **Pessoa 2** | 4, 5, 6 | Hipóteses, desenho e dados |
| **Pessoa 3** | 7, 8, 9 | Resultados |
| **Pessoa 4** | 10, 11, 12 | Veredito e nuances |
| **Pessoa 5** | 13, 14, 15, 16 | Fronteira e fechamento |

---

## Pessoa 1 — Abertura e o problema

### Slide 1 — Título
Bom dia a todos. Nós somos o grupo responsável por este trabalho de Redes Neurais Profundas, intitulado "Análise Arquitetural do Congelamento de Camadas na Mitigação de Domain Shift e Language Shift em Transformers Multilíngues". Em termos simples: a gente investigou o XLM-RoBERTa, que é um modelo de linguagem multilíngue, e perguntou o que acontece quando congelamos parte das suas camadas durante o treino. Será que isso deixa o modelo mais robusto quando muda o domínio ou o idioma dos dados? Eu vou abrir a apresentação com o problema, e em seguida passo a palavra para os colegas.

### Slide 2 — A pergunta central
O trabalho gira em torno de uma única pergunta: até que ponto congelar seletivamente camadas do XLM-RoBERTa, durante o ajuste fino, preserva a robustez do modelo quando a distribuição de teste muda. A montagem é a seguinte: a gente treina um classificador de sentimento sempre em inglês, no domínio de eletrônicos, e depois observa o que acontece em duas direções. Quando muda o domínio, de eletrônicos para beleza; e quando muda o idioma, começando pelo português, que é próximo do inglês, até japonês e mandarim, que são línguas distantes. A robustez a línguas distantes é o caso mais exigente da nossa hipótese, e é justamente o horizonte que guia todo o desenho do trabalho.

### Slide 3 — Por que congelar camadas pode importar
Mas por que congelar camadas faria diferença? A literatura de interpretabilidade de Transformers mostra que as camadas têm papéis distintos. As camadas de baixo, da zero à cinco, cuidam de léxico e sintaxe, e é nelas que mora boa parte do alinhamento multilíngue aprendido no pré-treino. Já as camadas de cima, da seis à onze, cuidam das abstrações semânticas, mais ligadas à tarefa. A intuição do nosso trabalho nasce daí: congelando as camadas certas, a gente preservaria exatamente a parte do modelo responsável por generalizar. A base ajudaria a mudar de idioma; o topo ajudaria a mudar de domínio.

---

## Pessoa 2 — Hipóteses, desenho e dados

### Slide 4 — Hipóteses
A partir dessa hierarquia, formulamos duas hipóteses. A primeira é sobre idioma: congelar a base, que chamamos de configuração C2, ou Freeze Lower, preservaria o alinhamento multilíngue e reduziria a queda ao mudar de idioma, um efeito que deveria ser tanto maior quanto mais distante a língua. A segunda é sobre domínio: congelar o topo, a configuração C3, ou Freeze Upper, reduziria a especialização no domínio de treino. Para testar, comparamos quatro configurações: a C1, que é o ajuste fino completo; a C2, que congela a base; a C3, que congela o topo; e a C4, que congela o encoder inteiro e treina só a cabeça de classificação. Essa cabeça nasce do zero e é sempre treinável.

### Slide 5 — Desenho experimental
Sobre o desenho experimental: o modelo é sempre treinado em inglês e eletrônicos. A avaliação usa células que isolam cada tipo de mudança. A T1 é a baseline, mesmo idioma e mesmo domínio. A T2 muda só o domínio, para beleza, e mede o Domain Shift. A T3 muda só o idioma, para o português, e mede o Language Shift próximo. A T4 muda os dois ao mesmo tempo. E a célula de língua distante, com japonês e mandarim, é a fronteira do estudo. A métrica é o F1-macro, e cada configuração foi replicada com seis a oito seeds independentes, rodadas pelos integrantes do grupo. Nos testes estatísticos usamos o t de Welch, o Mann-Whitney e o Cohen's d, com limiar de significância de dez por cento. Um detalhe importante: os hiperparâmetros são idênticos entre as configurações, de propósito, para isolar a única variável que interessa, que é o congelamento.

### Slide 6 — Engenharia de dados
Uma palavra sobre os dados, que foi a etapa de maior cuidado metodológico. Como precisávamos de inglês e português nos mesmos domínios, usamos uma filtragem híbrida. No inglês, que não tem metadado de categoria, o domínio é atribuído por palavras-chave inequívocas. No português, usamos a categoria oficial do produto, que é ground-truth. E auditamos o filtro do inglês com um classificador zero-shot multilíngue, para conferir a precisão. Por fim, adotamos um balanceamento desacoplado: o treino usa todo o pool de eletrônicos em inglês, enquanto as células de teste vão a um número comum de exemplos. Isso garante comparação justa entre os cenários, sem encolher o treino. O resultado é um treino com milhares de exemplos e células de teste balanceadas em cinquenta por cento de cada classe.

---

## Pessoa 3 — Resultados

### Slide 7 — Resultado 1: mapa de desempenho
Agora os resultados. Este mapa de calor mostra o F1-macro de cada configuração em cada cenário, com as médias sobre as seeds. A primeira leitura é clara: C1 e C2 praticamente empatam em todos os cenários, com diferenças de até quatro décimos de ponto percentual. A C3 fica consistentemente abaixo das duas. E a C4, que treina apenas a cabeça, desaba, com os valores mais baixos do mapa, naquela faixa escura embaixo. Ou seja, já dá para perceber que congelar a base quase não custa, mas congelar o topo, ou o encoder inteiro, custa caro.

### Slide 8 — Resultado 2: quanto cada deslocamento custa
Este gráfico mostra quanto cada deslocamento custa, em pontos percentuais, em relação à baseline T1. Acima de zero é perda; abaixo de zero é ganho. Olhem para as barras azuis, que são a mudança de idioma do inglês para o português: elas são negativas. Isso quer dizer que o modelo vai igual ou até melhor em português, mesmo nunca tendo treinado nesse idioma. É um ganho zero-shot. Ou seja, no par de línguas próximas, não existe Language Shift. Já as barras vermelhas, da mudança de domínio para beleza, são positivas: aí sim há perda real, que cresce da C2 para a C3 e dispara na C4.

### Slide 9 — Resultado 3: significância vs. C1
Aqui estão os testes de significância de cada configuração contra a baseline C1. Com seis a oito seeds, os efeitos ficam bem nítidos, e o Mann-Whitney chega a um p na ordem de um milésimo. No domínio, a C2 fica empatada com a C1, sem diferença significativa; já a C3 é três pontos percentuais pior, com p menor que um milésimo, e a C4 é quarenta e seis pontos pior. No idioma próximo, a história se repete: a C2 empata, e a C3 e a C4 são significativamente piores. A mensagem central deste slide é que nenhuma configuração congelada supera a C1 de forma significativa; pelo contrário, a C3 e a C4 só pioram.

---

## Pessoa 4 — Veredito e nuances

### Slide 10 — Veredito das hipóteses
Com isso, chegamos ao veredito das hipóteses. A primeira hipótese, de que congelar a base mitigaria o Language Shift, foi refutada, mas por um motivo interessante: não há Language Shift a mitigar no par inglês-português, porque a baseline já ganha desempenho em português. A segunda hipótese, de que congelar o topo mitigaria o Domain Shift, foi não só refutada como invertida: congelar o topo, na C3, piora o domínio em três pontos percentuais, com efeito grande e p menor que um milésimo. E vale destacar a robustez do resultado: as configurações viáveis, C1, C2 e C3, replicaram entre execuções independentes com menos de um ponto percentual de diferença. Ou seja, as conclusões não são um acidente de seed.

### Slide 11 — O valor real da C2: eficiência, não regularização
Aqui vai um ponto de honestidade científica. Em análises preliminares, com poucas seeds, a C2 parecia proteger contra o Domain Shift. Quando refizemos o teste com seis a oito seeds, esse efeito simplesmente desapareceu: a diferença caiu para dois décimos de ponto percentual, com p de quarenta e dois por cento, ou seja, sem significância nenhuma. A conclusão revisada é que a C2 é estatisticamente equivalente à C1: nunca melhor, nunca pior. Mas o valor dela está em outro lugar: ela treina cerca de metade dos parâmetros sem nenhuma perda de desempenho. O ganho da C2 é eficiência, e não regularização. E o fato de mais seeds terem mudado a nossa leitura é exatamente como a ciência deve funcionar.

### Slide 12 — C4 não é um piso, é uma loteria de seed
Sobre a C4, que congela o encoder inteiro: mais do que fraca, ela é instável. Dependendo da seed, o F1 varia de trinta e seis a oitenta e um por cento. O desvio entre seeds é de cerca de dez pontos percentuais, umas dezesseis vezes maior que o das configurações viáveis. A lição é que fazer apenas probing linear do XLM-RoBERTa, nesta tarefa, é uma loteria de inicialização: não é reprodutível. Então congelar o encoder inteiro não é só ruim de desempenho, é arriscado.

---

## Pessoa 5 — Fronteira e fechamento

### Slide 13 — A fronteira: língua distante
E aqui chegamos ao próximo passo, que é o teste decisivo do trabalho: verificar se a ausência de Language Shift que vimos no par próximo se mantém em uma língua tipologicamente distante, como o japonês ou o mandarim. Se a queda aparecer ali, então o Language Shift existe e estava apenas escondido pela proximidade entre inglês e português. Se não aparecer, a robustez multilíngue do modelo se confirma como um achado forte. O experimento usa o corpus MARC, com células de japonês, de mandarim e uma âncora em inglês, de modo que a comparação isole a distância linguística. O carregamento é feito de forma robusta, conferindo o idioma de cada conjunto. E o melhor: dá para reaproveitar os modelos C1 a C4 que já estão treinados, bastando rodar a avaliação nas novas células, sem nenhum treino novo.

### Slide 14 — Conclusões
Resumindo as conclusões. Primeiro: os resultados são robustos a seed, replicados de forma independente pelo grupo. Segundo: as duas hipóteses originais foram refutadas, e a segunda com evidência forte, p menor que um milésimo. Terceiro: a C2 vale pela eficiência, treinando metade dos parâmetros, e não por regularização. Quarto: a C4 é um piso inviável e instável. E quinto: a fronteira do trabalho, o teste em língua distante, está desenhada e pronta para execução.

### Slide 15 — Próximos passos
Como próximos passos, o principal é executar o teste de língua distante, com japonês e mandarim separados, que responde à pergunta central do trabalho. Em seguida, pretendemos trocar o filtro de domínio do inglês, que hoje é por palavra-chave, por um classificador de domínio, para remover o confound de qualidade entre os conjuntos de inglês e português. E, por fim, avaliar mais línguas distantes, como árabe e hindi, para mapear a curva de degradação conforme a distância linguística aumenta.

### Slide 16 — Obrigado
É isso. Agradecemos a atenção de todos. O trabalho completo, com o notebook passo a passo, o relatório detalhado e a análise reproduzível, está disponível no nosso repositório. Ficamos à disposição para as perguntas.
