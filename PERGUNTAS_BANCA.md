# Preparação para a banca de qualificação — perguntas prováveis

Qualificação: 10/07/2026. Banca: César Lincoln (orientador), João Paulo Pordeus
(coorientador), José Antônio Macêdo, Diego Mesquita (FGV EMAp).

Cada pergunta vem com a resposta sugerida, sempre ancorada no que a dissertação
e o código realmente dizem. **Não improvise além disso.**

---

## Prof. Diego Mesquita (perfil mais técnico: métodos probabilísticos, geometric DL, inferência aproximada)

### D1. "Isso é geometria hiperbólica? Qual a relação com hyperbolic deep learning?"
**Resposta:** Não é embedding em espaço hiperbólico (Poincaré/Lorentz manifolds).
O bloco B(Δt) é uma *rotação hiperbólica* (boost de Lorentz): preserva a forma
bilinear de Minkowski, não a euclidiana. O nome "hiperbólico" vem das funções
cosh/sinh que substituem cos/sin no kernel rotacional, seguindo o HoPE. É uma
mudança de kernel de atenção, não de geometria do espaço latente.

### D2. "Expandindo, seu score é Σ_p [c_p(q,k)·e^{−(θ′−θ_p)|Δt|} + d_p(q,k)·e^{−(θ′+θ_p)|Δt|}]. Ou seja, uma mistura de exponenciais. O que a estrutura hiperbólica compra em relação a simplesmente multiplicar o score por e^{−β|Δt|} ou somar um viés tipo ALiBi?"
**Resposta (pergunta mais importante da defesa):**
1. Os **coeficientes dependem do conteúdo** (produtos q·k por par de dimensões),
   podendo ser positivos ou negativos — diferente do ALiBi, cuja penalidade é
   independente do conteúdo e igual para todos os pares.
2. O kernel gera um **espectro de taxas de decaimento** {θ′−θ_p, θ′+θ_p},
   p = 1..d/2: como θ_p varia de 1 até ~10⁻⁴, o modelo tem componentes que
   decaem quase na taxa mínima θ′−θ_1 (que pode ficar próxima de zero, retendo
   memória mais longa) e componentes rápidas θ′+θ_p. Um único e^{−β|Δt|}
   multiplicativo teria uma única escala temporal.
3. Mantém a **invariância temporal** do RoTHP (score depende só de Δt), o que o
   ALiBi discreto não dá em tempo contínuo.

### D3. "Se o kernel do RoTHP depende só de Δt, por que sequências mais LONGAS quebram o RoTHP? O Δt típico entre eventos consecutivos não muda."
**Resposta:** A atenção é sobre todos os pares (i, j) sob máscara causal. Com
mais eventos, o intervalo total t_i − t_j entre pares distantes cresce com o
comprimento da sequência, então o modelo encontra valores de Δt muito maiores
que os do treino. No kernel trigonométrico essas fases nunca vistas podem cair
em picos, dando score alto a eventos remotos; no hiperbólico o decaimento é
estrutural e vale para qualquer Δt. Além disso, o softmax passa a normalizar
sobre muito mais chaves, o que amplifica o efeito de scores espúrios.

### D4. "Você tem alguma garantia teórica de extrapolação, ou o argumento é só qualitativo + empírico?"
**Resposta honesta:** A garantia formal é sobre o *kernel*: com θ′ > max θ_p por
construção, o score decai monotonicamente para zero em |Δt| e é limitado — isso
é demonstrável pelas identidades das Eqs. de estabilidade (soma de exponenciais
com expoentes negativos). Não há bound sobre a NLL em extrapolação; a evidência
é empírica (Wilcoxon, 10 seeds). Análise teórica do θ′ aprendido está nos
trabalhos futuros.

### D5. "Por que as frequências θ_p continuam fixas (fórmula do RoPE) e só θ′ é aprendido? Por que um único θ′ global?"
**Resposta:** Escolha de desenho herdada do RoPE/HoPE, que mantém a comparação
controlada com o RoTHP (mesmas frequências). Um θ′ global já garante o
decaimento com 1 parâmetro; múltiplas taxas (por cabeça ou por par) estão
listadas como trabalho futuro. Aprendemos também que um fator de escala
temporal aprendido causou instabilidade (Seção 4.6), então fomos conservadores
no que tornar aprendível.

### D6. "Como a integral do compensador é calculada? Quantos pontos? Viés do estimador?"
**Resposta:** A mesma rotina para os 4 modelos, definida uma única vez na classe
base do EasyTPP: amostragem em grade fixa de 20 pontos por intervalo entre
eventos (grade determinística tipo linspace — na prática uma quadratura, embora
o texto chame de Monte Carlo). O ponto central para a comparação: **qualquer
viés do estimador é idêntico para todos os modelos**, então o ranking de NLL
não depende da aproximação da integral.
⚠️ *Se ele pressionar na palavra "Monte Carlo": admitir que é uma grade
determinística de 20 nós por intervalo e que a dissertação vai precisar dessa
precisão terminológica.*

### D7. "A normalização por sequência usa a média dos gaps DA SEQUÊNCIA DE TESTE. Isso não usa informação do futuro?"
**Resposta:** É um pré-processamento aplicado igualmente a todos os modelos, e
os timestamps são entrada (não alvo), então não há vazamento de rótulo. Mas sim,
em um cenário online estrito a média incluiria eventos futuros; a versão online
usaria média corrente ou a média do treino. Por isso a normalização é usada só
nos sintéticos e na ablação do Retweet, que estuda o efeito da escala — Taxi,
SO e Amazon usam timestamps crus (apenas shift para zero).

### D8. "NHP ganha em NLL na Amazon por margem enorme (0,64 vs 2,2 nats). A família Transformer é sequer competitiva? Qual a relevância prática?"
**Resposta (enquadramento já escrito na dissertação, Seção 5.2):** Esperado e
consistente com o benchmark unificado do EasyTPP, onde o NHP recorrente é o mais
forte em vários datasets (formulação em tempo contínuo modela a densidade entre
eventos nativamente). A pergunta de pesquisa é *within-family*: fixada a
arquitetura Transformer (mesma intensidade, mesmos hiperparâmetros), o kernel
posicional muda a extrapolação? Nessa comparação controlada o HoTHP é
consistentemente o melhor dos três; em acurácia de tipo ele supera o próprio NHP
na Amazon (0,340 vs 0,315) e no SO. E nos sintéticos, com 10 seeds e Wilcoxon,
o HoTHP vence *inclusive o NHP* em todos os fatores de extrapolação.

### D9. "3 seeds nos dados reais, sem teste estatístico. Quão confiáveis são essas conclusões?"
**Resposta:** Com N=3 o menor p-valor possível do Wilcoxon é 0,125 > 0,05, então
o teste seria inválido — preferimos não reportar significância a reportar
pseudo-significância. Reportamos média ± desvio (desvios pequenos, ex.
−0,651 ± 0,013). Ampliar seeds nos reais para viabilizar o Wilcoxon é item
explícito do cronograma até a defesa.

### D10. "Por que o HoTHP perde em 1× (in-distribution)?"
**Resposta:** O viés estrutural restringe o espaço de hipóteses: em 1× os outros
modelos podem ajustar padrões que o decaimento exponencial suprime. É o
trade-off clássico de viés indutivo: paga-se um pouco de ajuste em distribuição
para ganhar generalização fora dela. A diferença em 1× é pequena (ex. 1,545 vs
1,470) e reverte já em 2×.

---

## Prof. César Lincoln (orientador: GPs, ML probabilístico, reconhecimento de padrões)

### C1. "O processo de Cox aparece na fundamentação. Por que não comparar com uma intensidade modulada por GP (log-Gaussian Cox process)?"
**Resposta:** O LGCP modela a estocasticidade da intensidade por fator externo
latente; o Hawkes (e os modelos neurais aqui) a modelam pela própria história.
O objeto de estudo é o kernel posicional da atenção, então os baselines
relevantes compartilham a arquitetura de atenção. Comparação com processos de
Cox neurais/GP é extensão natural (posso citar como trabalho futuro).

### C2. "Wilcoxon pareado por seed: por que esse teste? Corrigiu para comparações múltiplas?"
**Resposta:** Não-paramétrico (sem hipótese de normalidade sobre NLL), pareado
porque a mesma seed gera os mesmos dados para todos os modelos, adequado a
N=10. Unicaudal (hipótese direcional: HoTHP melhor). Não aplicamos correção de
múltiplas comparações; mitigante: a maioria dos p-valores está no piso de
0,001 = 1/2¹⁰, então sobreviveriam a Bonferroni razoável — exceto o caso
slow-decay 10× vs NHP (p = 0,042), que reportamos explicitamente como a margem
mais apertada.

### C3. "Como você separa 'o viés ajuda' de 'o HoTHP só tem hiperparâmetros mais afinados'?"
**Resposta:** THP/RoTHP/HoTHP compartilham arquitetura, intensidade e
hiperparâmetros (exceto lr do HoTHP nos sintéticos, 5×10⁻⁴, por estabilidade
das funções hiperbólicas — posso rodar sensibilidade se a banca julgar
necessário). A única diferença estrutural é o kernel posicional.

### C4. "O que exatamente você entrega até a defesa?" 
**Resposta:** Cronograma (cap. 7): incorporar feedback da banca, mais seeds nos
reais p/ Wilcoxon, refinar ablações de normalização, completar conclusão/resumo,
defesa em agosto/2026.

---

## Prof. José Antônio Macêdo (Big Data, mineração, InsightLab, dados urbanos/trajetórias)

### M1. "O kernel é O(L²) por par de eventos. Isso escala para sequências realmente longas / cenário Big Data?"
**Resposta:** Atenção já é O(L²); o HoTHP adiciona fator constante (d/2 pares de
exponenciais). Medido: 1,07× (Taxi) a 2,44× (Retweet, sequências de 250) mais
lento que o RoTHP, e ainda mais barato que o NHP (o baseline mais forte em NLL).
A implementação usa chunking na dimensão de query para não materializar tensores
[B,L,L,D] (~1GB com L≈250). Aproximações low-rank são trabalho futuro. E o custo
é pago uma vez, no treino — na inferência o que importa é o resultado ser
válido (RoTHP a 5×: NLL 47,6; HoTHP: −0,651).

### M2. "Taxi é um dataset de mobilidade. Esse modelo se aplica a trajetórias/dados urbanos em geral?"
**Resposta:** Sim, a qualquer fluxo de eventos com timestamp e tipo em que
eventos recentes são mais informativos (demanda de corridas, ocorrências
urbanas, alertas de infraestrutura). A limitação prática é escala temporal
extrema (caso Retweet) — resolvível com a normalização por gap médio mostrada
na ablação. Marcas espaciais contínuas exigiriam estender o modelo (marks
categóricas hoje).

### M3. "Por que esses 4 datasets? Tamanhos? Split?"
**Resposta:** Benchmarks públicos padronizados do EasyTPP (mesmo framework dos
autores do NHP), o que garante comparabilidade com a literatura: Amazon (16
tipos, 6.454 seqs), SO (22, 1.401), Taxi (10, 1.400), Retweet (3, 20.000).
Sintéticos: 400/100/200 train/val/test. Protocolo train-short-test-long com
L_train fixo por dataset (18/20/7/50).

### M4. "Como isso vira produto/ferramenta? Alguém consegue reusar?"
**Resposta:** Implementado como modelo registrável no EasyTPP (subclasse, mesma
config YAML dos demais); qualquer usuário do framework troca `model_id` e roda.

---

## Prof. João Paulo Pordeus (coorientador: DSP, prognóstico de falhas, reconhecimento de padrões)

### J1. "O RMSE não separa os modelos. Se a predição pontual não melhora, qual o valor prático?"
**Resposta:** A NLL mede a qualidade da *distribuição* completa (quando E de que
tipo, com que incerteza), que é o que importa para simulação, detecção de
anomalia (eventos improváveis sob o modelo) e risco — não só o erro pontual do
próximo intervalo. Em acurácia de tipo o HoTHP também é o melhor da família e
supera o NHP em Amazon/SO. E no Retweet extrapolado a diferença não é de
métrica, é de validade: NLL 47,6 (RoTHP) vs −0,651 (HoTHP).

### J2. "O θ′ aprendido convergiu para quê? Ele se relaciona com o β do processo gerador?"
**Resposta preparada (verificar valor antes da defesa!):** A relação esperada é
que no cenário fast-decay o θ′ efetivo fique maior (memória curta) e no
slow-decay, menor. Ter esse número na ponta da língua fortalece muito a
resposta — extrair θ′ = softplus(raw) + 1 + 1e-4 dos checkpoints e anotar.
⚠️ *TODO antes da defesa: rodar célula no notebook e anotar os valores.*

### J3. "Aplicação em prognóstico de falhas (meu domínio): sequências de falhas são curtas e raras. O modelo serve?"
**Resposta:** É exatamente o cenário-alvo do train-short-test-long: treinar com
histórico curto disponível e operar em janelas longas. Falhas com influência de
manutenção preventiva são um caso de processo com inibição — e mostramos (Seção
4.8) que o viés de recência atua na atenção, não no sinal da intensidade, então
o modelo acomoda inibição (caso Amazon).

### J4. "Explique a estabilidade numérica sem slides."
**Resposta-elevator:** cosh e sinh explodem como e^{θ|Δt|}. Nunca computamos
cosh/sinh isolados: distribuímos o decaimento para dentro via
e^{−|Δt|θ′}·cosh(θ|Δt|) = ½(e^{−|Δt|(θ′−θ)} + e^{−|Δt|(θ′+θ)}). Como θ′ > θ por
construção, todos os expoentes são negativos — nenhum overflow, para qualquer
Δt. O sinal do sinh volta via sgn(Δt).

---

## Perguntas gerais de qualificação (qualquer membro)

### G1. "Qual é exatamente a SUA contribuição em relação ao HoPE? Não é só aplicar HoPE em TPP?"
**Resposta:** A adaptação não é trivial: (i) posições discretas → instantes
contínuos com |Δt| irregular e não limitado; (ii) parametrização de θ′ que
garante θ′ > max θ_p *por construção* (softplus + max + ε), não por sorte do
gradiente; (iii) decomposição em exponenciais puras para estabilidade em Δt
arbitrário; (iv) a análise de *onde* o viés atua (atenção vs intensidade), que
explica por que funciona em processos auto-inibitórios; (v) o protocolo
train-short-test-long adaptado a TPPs com validação estatística.

### G2. "Por que o nome 'Hawkes' se a atenção substitui o kernel de Hawkes?"
**Resposta:** Herança da linhagem THP/RoTHP (Transformer *Hawkes* Process). E o
HoTHP justamente *devolve* à atenção a propriedade central do kernel de Hawkes
(decaimento exponencial estrutural) que THP/RoTHP abriram mão — é o argumento
do slide do ALiBi.

### G3. "Por que sem conexões residuais?"
**Resposta:** Seguimos o desenho do RoTHP para manter a comparação controlada
(mesma arquitetura, só muda o kernel). `use_residual=False` para todos os três.

### G4. "Se distância grande ⇒ score ~0 para TODOS os pares, o softmax não fica uniforme sobre eventos distantes?" 
**Resposta:** O softmax é sobre a linha da query; os pares recentes dominam com
scores maiores, e os distantes recebem massa ≈ proporcionalmente igual e
desprezível — comportamento desejado (equivale a truncar a história suavemente,
como o Hawkes com kernel exponencial efetivamente faz).

---

## ⚠️ Pontos fracos conhecidos

### Antes da QUALIFICAÇÃO (10/07)

1. **Eq. da intensidade (4.x) ≠ código:** o texto escreve α_k(t−t_j)/(t_j+ε)
   (forma do paper do THP), mas o código implementado (herdado de torch_thp.py)
   usa α_k·τ linear no tempo decorrido, sem dividir por t_j. Recomendação:
   corrigir a equação na dissertação para refletir o implementado (é inclusive
   mais elegante: sem tempo absoluto, coerente com a invariância temporal).
   O slide da arquitetura no deck reproduz a forma da dissertação — alinhar os
   dois quando decidir. No mínimo, saber responder se perguntarem.
2. **"Monte Carlo" vs grade determinística:** o código usa grade fixa de pontos
   por intervalo (linspace). Ajustar a redação (ou trocar por "quadratura
   numérica com os mesmos nós para todos os modelos"), ou ao menos ter a
   resposta pronta (D6).
3. **Anotar o valor de θ′ aprendido** por dataset/cenário (perguntas J2/D5) —
   munição barata e de alto impacto para a arguição.

### Para a DEFESA FINAL (agosto) — não bloqueiam a qualificação

O capítulo de conclusão (6-conclusao.tex) está **comentado no documento.tex**
enviado à banca de qualificação, por orientação do César (conclusão só entra na
defesa final). Ao reativá-lo:

4. **Preencher a seção "Future work"**, que está vazia (o slide de limitações do
   deck já tem o conteúdo: mais seeds, múltiplas taxas de decaimento, low-rank,
   análise teórica de θ′).
5. **Atualizar números defasados da conclusão:** cap. 6 diz Retweet norm. 5×
   NLL = −0,542 ± 0,004 e custo 1,05×–2,24×; cap. 5 (atual) diz −0,651 ± 0,013
   e 1,07×–2,44×. Também "ties on Taxi" subestima o resultado within-family.

~~Figuras ausentes de figuras/~~ — resolvido (baixadas do Overleaf em
02/07/2026).
