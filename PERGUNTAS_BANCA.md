# Preparação para a banca — qualificação e defesa final

Mesma banca nas duas etapas: César Lincoln (orientador), João Paulo Pordeus
(coorientador), José Antônio Macêdo, Diego Mesquita (FGV EMAp).

Cada pergunta vem com a resposta sugerida, sempre ancorada no que a dissertação
e o código realmente dizem. **Não improvise além disso.**

A seção abaixo é a atualização para a **defesa final (agosto/2026)**, depois de
incorporado o retorno da qualificação (10/07/2026). Todas as perguntas D1-D10,
C1-C4, M1-M4, J1-J4 e G1-G4 mais abaixo continuam válidas (arquitetura e
experimentos centrais não mudaram) — leia-as primeiro, depois esta seção para o
que é novo.

---

## Atualização para a defesa final (agosto/2026)

### A. Pendências encontradas revisando o material — status atualizado

Estas três já foram corrigidas diretamente nos arquivos (2026-08-25):

1. ~~**Slide de backup com TODO literal não preenchido.**~~ **Corrigido.** O
   frame "Backup: para quanto o $\theta'$ convergiu?" tinha `[preencher]` nas
   três células da tabela. Extraí `hope_emb.theta_prime_raw` dos checkpoints
   salvos localmente (`dissertacao/final-material/HoTHP_validation_synthetic_dataset/`
   para os sintéticos, `checkpoints/{amazon,taxi,stackoverflow}/HoTHP/` para os
   reais) e preenchi com valores reais:

   | Cenário / conjunto | $\theta'$ aprendido |
   |---|---|
   | Sintético, decaimento rápido (10 seeds) | $1.665 \pm 0.105$ |
   | Sintético, decaimento lento (10 seeds) | $1.872 \pm 0.110$ |
   | Amazon (3 seeds) | $1.415 \pm 0.026$ |
   | Taxi (3 seeds) | $2.020 \pm 0.005$ |
   | StackOverflow (3 seeds) | $1.973 \pm 0.012$ |

   **Achado que contraria a expectativa que estava escrita no slide:** o
   decaimento *lento* aprende $\theta'$ **maior** (decaimento mais rápido) do
   que o decaimento *rápido* — o oposto do que a hipótese original previa.
   Reescrevi o slide para explicar isso: $\Delta t$ entra no kernel já
   normalizado pelo gap médio de cada sequência ($\approx 1$, Seção 4.7), então
   $\theta'$ mede o decaimento relativo a essa unidade normalizada, não a taxa
   física $\beta$ do gerador — a normalização remove justamente a escala
   absoluta que a intuição ingênua pressupunha. **Leve isso na ponta da língua
   para a arguição**, porque agora é um número real que a banca pode
   confrontar, não mais uma expectativa não testada.

   ⚠️ **Verificar antes da defesa:** os checkpoints encontrados usam seeds
   `2019`, `2020`, `2021` nos dados reais, enquanto o texto da dissertação
   (`4-metodologia.tex`, Seção "Compared models") diz que os experimentos
   principais usam seeds `42, 142, 242`. Os números acima são de checkpoints
   reais e treinados com a arquitetura correta, mas não confirmei que são
   exatamente as mesmas rodadas que geraram as Tabelas 5.x. Se não forem,
   reextrair com os checkpoints certos (mesmo procedimento, só trocar o
   caminho dos `.pt`).

2. **Contradição entre a conclusão e o capítulo de resultados sobre o Taxi
   (NÃO corrigida — dissertação já foi enviada à banca, não mexer no texto).**
   `6-conclusao.tex` diz "ties on Stackoverflow and Taxi", mas
   `5-resultados.tex` descreve o oposto para o Taxi dentro da família
   Transformer: "the HoTHP again has the best NLL of the three at every
   factor and is the most stable in extrapolation... while the HoTHP stays at
   $-0.266 \pm 0.007$" contra $-0.137 \pm 0.148$ do RoTHP em 5×. Como o
   documento já foi entregue, isso fica só como munição para a arguição: se
   perguntarem sobre o Taxi, a resposta correta é a do capítulo 5 (vitória
   dentro da família), não a frase resumida (e imprecisa) da conclusão. Vale
   avisar a banca da imprecisão de boa vontade se sobrar tempo, em vez de
   deixar que a percebam sozinhos.

3. **"Monte Carlo" no corpo do texto (NÃO corrigida, mesmo motivo).**
   `4-metodologia.tex` ainda diz "the same Monte Carlo estimate of the
   compensator integral". A resposta honesta (grade determinística de 20 nós
   por intervalo, não amostragem aleatória) é a D6, já escrita mais abaixo
   neste documento — leve-a pronta se alguém, principalmente o Diego,
   perguntar sobre o termo.

Esta ainda é uma decisão a tomar, não um erro para corrigir sozinho:

4. **Learning rate assimétrico nos sintéticos, sem grid search documentado
   para os baselines.** `4-metodologia.tex` explica que o HoTHP usa
   $5\times10^{-4}$ nos sintéticos "porque o kernel hiperbólico treinou de
   forma menos estável em $10^{-3}$", enquanto NHP/THP/RoTHP usam $10^{-3}$
   sem que o texto diga se essa taxa foi de fato testada para eles ou apenas
   herdada. Só há grid search documentado para o HoTHP na Amazon (dados
   reais). Não editei isso porque não sei se um grid search para os baselines
   nos sintéticos de fato foi rodado — só você sabe. Se foi, vale citar no
   texto; se não foi, a resposta em arguição é a que já está pronta na seção
   B mais abaixo.

### B. Perguntas novas geradas pelo conteúdo adicionado desde a qualificação

#### Do José Antônio Macêdo (dados urbanos/trajetórias — ele vai notar isso no Taxi)

**"Vocês escrevem uma seção inteira dizendo que o viés de recência falha em
processos periódicos, e citam explicitamente ritmo diário/semanal de corridas
de táxi como exemplo desse tipo de processo (Seção 5.5.2). Mas o HoTHP é
justamente o melhor da família Transformer no dataset Taxi. Isso não é uma
contradição?"**
Resposta sugerida: Não, porque a limitação vale para o alcance temporal
efetivamente representado na sequência, e o Taxi usa $L_{\text{train}}=7$ e
$L_{5\times}=38$ eventos — sequências curtas demais para que um ciclo
diário/semanal apareça dentro delas. A periodicidade da demanda de táxi é um
efeito de nível populacional (a cidade como um todo, agregada ao longo do
dia), não uma propriedade que preencha a janela de 7 a 38 eventos que o modelo
de fato observa por sequência. A limitação da Seção 5.5.2 é sobre um regime
que este benchmark específico não testa (dependências que abrangem centenas de
eventos), não uma contradição com o resultado observado.

#### Do César Lincoln (ele vai puxar o design experimental)

**"O protocolo train-short/test-long, por definição, favorece um modelo com
viés de recência — vocês mesmos escrevem isso na Seção 5.5.1 ('the protocol
also rewards a recency bias almost by construction'). Isso não enfraquece a
validade da comparação, já que é basicamente um benchmark desenhado para o
seu próprio modelo vencer?"**
Resposta sugerida: A honestidade sobre isso está no texto de propósito. Dois
contrapontos: (i) nos sintéticos, o viés de recência não é arbitrário, é a
definição matemática de um processo de Hawkes com kernel exponencial — o
"favorecimento" é o próprio objeto de estudo, não um artefato do desenho
experimental; (ii) nos dados reais, onde a estrutura geradora é desconhecida,
o resultado é misto (empates no Stackoverflow, perda para o NHP em NLL bruta
na Amazon/Taxi), o que mostra que a avaliação não está artificialmente armada
a favor do HoTHP — se estivesse, ele venceria em NLL absoluta em todos os
datasets reais também, e não vence.

#### Do Diego Mesquita (ele vai perguntar sobre o slide de replicação do THP)

**"O slide de backup menciona que vocês reproduziram o THP original no
MIMIC-II e encontraram uma LL inflada por causa da variante de solver Monte
Carlo do artigo. Se a LL reportada na literatura do THP é inflada, isso não
compromete usar o THP como baseline em toda a comparação da dissertação?"**
Resposta sugerida: Não, porque dentro da dissertação todos os quatro modelos
(NHP, THP, RoTHP, HoTHP) usam exatamente a mesma rotina de integração,
definida uma única vez na classe base do EasyTPP (`TorchBaseModel`) — qualquer
viés desse estimador é idêntico entre os modelos, então não afeta o ranking
interno. O achado do MIMIC-II é sobre outra coisa: LLs publicadas em artigos
diferentes, com implementações diferentes do integral, não são comparáveis
entre si. Foi exatamente esse achado que motivou reimplementar tudo sob um
único pipeline (EasyTPP) em vez de citar números de LL da literatura
diretamente — é o argumento a favor do desenho da própria dissertação, não uma
fragilidade dela. (Deixar claro que essa investigação é complementar, em uma
base de código separada do EasyTPP, e não um capítulo da dissertação.)

#### Do João Paulo Pordeus (ele vai comparar custo computacional com o mundo real de prognóstico)

**"Vocês dizem no capítulo de conclusão que a atenção remove o gargalo
sequencial das RNNs, mas a própria Tabela de tempo de treino mostra o NHP
recorrente sendo o mais lento de todos, e ele é o baseline mais forte em NLL.
Isso não contradiz o argumento de que Transformers são mais rápidos?"**
Resposta sugerida: O argumento sobre paralelismo é assintótico (custo por
época, por passo de otimização) e continua verdadeiro: dentro de uma época, a
atenção processa a sequência inteira em paralelo, enquanto a CT-LSTM processa
evento por evento, com sete equações de gate por passo (Equações 217-226 da
fundamentação teórica). O NHP ser o mais lento no relógio de parede reflete
esse custo por passo mais alto multiplicado pelo número de passos sequenciais,
não uma vitória da complexidade quadrática $O(L^2)$ da atenção sobre a linear
$O(L)$ da recorrência — em L pequeno (50 a 500 eventos), a constante da
recorrência sequencial domina; a vantagem assintótica da atenção só se
inverteria em sequências muito mais longas do que as usadas aqui.

### C. Perguntas gerais sobre o que mudou desde a qualificação

**"O que exatamente vocês fizeram com o nosso retorno da qualificação?"**
Resposta: os 14 itens da lista de feedback foram endereçados um a um (ver
slide "Desde a qualificação" da apresentação final) — a mais substantiva foi a
adição da Seção 2.4 (taxonomia Poisson-Hawkes), a formalização da ligação
verossimilhança Poisson→Hawkes ao final da 2.3.1, a discussão sobre o período
do RoPE ao final da 4.1 (por que um período maior não resolve a oscilação sem
perder resolução temporal), a tabela de RMSE nos dados reais, e as três novas
subseções de discussão no Capítulo 5 (viés de recência vs. memória de longo
alcance, processos periódicos, custo computacional vs. benefício), além da
conclusão e do capítulo de trabalhos futuros, que estavam propositalmente
comentados na qualificação por orientação do César.

**"Por que vocês não testaram nenhuma das alternativas de trabalho futuro,
como o Mamba Hawkes Process, se ele já resolve o custo quadrático?"**
Resposta: o escopo desta dissertação é uma pergunta específica sobre o kernel
posicional dentro do mecanismo de atenção, estabelecida a partir do RoTHP como
baseline direto. O Mamba Hawkes Process é uma família de arquitetura
totalmente diferente (state-space, não atenção) e muito recente (2024, dos
mesmos autores do RoTHP) — testar se o viés de recência hiperbólico transfere
para lá é uma pergunta de pesquisa natural, mas separada, e por isso foi
listada como trabalho futuro em vez de incorporada ao escopo já delimitado.

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
