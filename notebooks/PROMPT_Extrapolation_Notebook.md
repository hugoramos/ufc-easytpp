# Prompt: Notebook de Extrapolação e Análise de Atenção (RoTHP vs HoTHP)

Use este texto como contexto para entender o que o notebook **Extrapolation_and_Attention_Analysis.ipynb** faz e quais são os resultados até o momento.

---

## Objetivo do notebook

O notebook testa a **hipótese central do HoTHP** (Hyperbolic Transformer Hawkes Process):

- **RoTHP** usa RoPE (Rotary Position Encoding): rotação trigonométrica → padrões de atenção **oscilatórios** com a distância temporal.
- **HoTHP** usa HoPE (Hyperbolic Position Encoding): boosts hiperbólicos + decay multiplicativo → atenção **monotonicamente decrescente** com a distância temporal.

A hipótese é que, em **extrapolação temporal** (avaliar em lags além dos vistos no treino), o RoTHP degrada mais porque a atenção oscila de forma imprevisível, enquanto o HoTHP mantém um decay estável e generaliza melhor.

---

## Estrutura do notebook (4 partes)

### Part 1: Perfil teórico do attention score (sem treino)

- Fixa vetores q, k e varia a distância temporal Δt.
- Plota o **score de atenção** (kernel posicional) em função de Δt para RoTHP e HoTHP.
- **RoTHP:** curva oscilatória (cos/sin). **HoTHP:** decay monotônico exp(−Δt·θ′) × termos hiperbólicos.
- Confirma que o *desenho* dos encodings produz o contraste esperado (oscilação vs decay).

### Part 2: Dataset sintético para teste de extrapolação

- **Processo:** Hawkes multivariado (4 tipos), kernel misto (exponencial rápido + lento), mesmos parâmetros para todos os conjuntos.
- **Train / Val / Test-short:** sequências geradas **sem** gaps artificiais. Os intervalos entre eventos (deltas) têm distribuição “natural” (ex.: média ~0,25 em tempo bruto). Depois as sequências são **normalizadas** pelo tempo (divisão pela média dos deltas do treino), de modo que no treino/val/short os lags vistos pelo modelo ficam tipicamente na faixa de ~1 a ~15 (normalizado).
- **Test-extrapolation:** mesmas gerações Hawkes, mas após a geração aplica-se **inject_long_gaps**: são inseridos **3 gaps artificiais** por sequência, com comprimento uniforme em **[15, 35]** (mesmas unidades do processo). Após a normalização, esses gaps viram **lags muito maiores** (ex.: 60–140 em unidades normalizadas) do que os que o modelo viu no treino.
- **Cenário de extrapolação (resumo):** Treino e validação só veem sequências com lags “curtos”. No **test-extrapolation** avaliamos em sequências que contêm **distâncias temporais (lags) entre eventos muito maiores** do que as do treino. O modelo precisa **extrapolar** o comportamento da atenção (e da intensidade) para lags nunca ou raramente vistos — daí o nome “extrapolação temporal”.

### Part 3: Treino e avaliação (RoTHP vs HoTHP)

- **Modelos:** RoTHP e HoTHP com mesma arquitetura (hidden_size=32, 2 layers, 2 heads, etc.), a partir do repositório ufc-easytpp.
- **Treino:** múltiplas seeds (ex.: 5), mesmo número de épocas (ex.: 150–500), early stopping por val NLL. Opcionalmente registra-se **histórico** de val NLL e de test NLL (short e extrap) a cada N épocas.
- **Avaliação:** ao final de cada seed, calcula-se NLL no **test-short** (in-distribution) e no **test-extrapolation** (extrapolação). Define-se **degradação** = NLL(extrap) − NLL(short).
- **Análise de robustez:** runs do HoTHP com extrap NLL acima de um limiar (ex.: 15) são considerados “não convergidos”. Nas runs “convergidas” do HoTHP, compara-se degradação RoTHP vs HoTHP (mesmos seeds).
- **Testes estatísticos:** t-test pareado e Wilcoxon para testar se a degradação do HoTHP é menor que a do RoTHP (em todas as seeds ou só nas convergidas).
- **Gráficos:** barras (média ± dp) de NLL por modelo e split (short vs extrap); curvas de NLL por época (val, test short, test extrap) por seed e por modelo.

### Part 4: Perfis de atenção aprendidos (attention × lag)

- Extrai os **pesos de atenção** reais dos modelos treinados (última seed) sobre o test-extrapolation.
- Agrega por **lag temporal** (distância entre query e key) e plota **peso médio de atenção** em função do lag (e opcionalmente zoom em lags curtos).
- **Resultado esperado:** RoTHP → perfil **oscilatório** e com **maior variância** entre seeds; HoTHP → perfil **decrescente suave** e **mais estável**. Isso é a evidência empírica do viés indutivo (estável “attention weight × temporal lag”) do HoTHP.

---

## Cenário em que o HoTHP se beneficia

- **Regime:** avaliação em **extrapolação** (test-extrapolation), não em short range.
- **Condição:** o **treino do HoTHP precisa convergir** para a extrapolação (extrap NLL abaixo de um limiar razoável). Quando converge, a degradação (extrap − short) do HoTHP tende a ser **menor** que a do RoTHP nos mesmos seeds.
- **Evidência do perfil:** o gráfico “Learned Attention Profiles” (Part 4) mostra que o HoTHP aprende decay estável com o lag; o RoTHP aprende perfil oscilatório e mais variável. A inconsistência de NLL do HoTHP entre seeds é atribuída a **otimização/convérgencia**, não ao desenho da atenção.
- **Conclusão para a dissertação:** o cenário onde o perfil “attention weight × temporal lag” estável do HoTHP beneficia é **extrapolação temporal + runs em que o HoTHP converge**. A maior variância entre seeds do HoTHP é uma **limitação prática**; reportar resultados condicionados a convergência (e/ou múltiplas seeds) permite argumentar o benefício do viés.

---

## Resultados até o momento

- **Short range (in-distribution):** RoTHP e HoTHP performam de forma **muito similar** (NLL baixo e estável entre seeds).
- **Extrapolação:**  
  - **RoTHP:** degradação consistente (extrap NLL > short NLL), com variância moderada entre seeds.  
  - **HoTHP:** quando **converge** (extrap NLL razoável), a degradação é **menor** que a do RoTHP nos mesmos seeds; porém **nem todos os seeds convergem** — alguns apresentam extrap NLL muito alto (falha catastrófica) ou convergência lenta/instável. Com seeds próximos (42–46) e mais épocas (500), a inconsistência do HoTHP persiste (alguns seeds bons, outros ruins).
- **Curvas de NLL por época:** ajudam a distinguir “não convergiu a tempo” (curva de extrap ainda caindo) de “convergiu em mínimo ruim” (val/short bons, extrap alto e estável). Mais épocas podem ajudar em alguns runs lentos, mas não resolvem todos os casos.
- **Perfis de atenção aprendidos:** confirmam que o HoTHP aprende decay monotônico estável e o RoTHP aprende perfil oscilatório e mais variável, alinhado à teoria. O benefício em NLL aparece quando o HoTHP converge no regime de extrapolação.

---

## Arquivos e dependências

- **Notebook:** `notebooks/Extrapolation_and_Attention_Analysis.ipynb`
- **Modelos:** `easy_tpp/model/torch_model/torch_rothp.py`, `easy_tpp/model/torch_model/torch_hothp.py` (projeto ufc-easytpp)
- **Config:** definido no próprio notebook (ModelConfig com hidden_size, num_layers, num_heads, etc.). Constantes como `EPOCHS`, `N_SEEDS`, `EXTRAP_NLL_THRESHOLD` podem ser ajustadas no notebook.

---

## Resumo em uma frase

O notebook treina RoTHP e HoTHP em sequências de Hawkes com lags “curtos”, avalia em **short range** e em **extrapolação** (sequências com long gaps injetados), e mostra que o HoTHP só supera o RoTHP em extrapolação **quando o treino converge**, com evidência empírica nos perfis de atenção (decay estável vs oscilatório).
