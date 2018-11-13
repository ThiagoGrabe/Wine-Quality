# Wine-Quality
Este repositório contém um projeto de análise de dados e algoritmos de aprendizado de máquina para predizer/classificar a qualidade
de um vinho dado alguns parâmetros químicos.

## Introdução

Vinhos são produtos amplamente consumidos em qualquer cultura. Tanto o vinho branco quanto o vinho tinto possuem características peculiares
que podem ser apreciadas em diversos ambientes e ocasiões sociais. De acordo com alguns [sites especializados](https://vinepair.com/articles/chemical-compounds-wine-taste-smell/)
características químicas do vinho como teor alcólico e a presença de ácidos podem representar um bom vinho mas também caracterizá-lo como um vinho de baixa qualidade.

O projeto será abordado como um problema de classificação no qual a variável desejada é a qualidade, entre 0 e 10, dos vinhos.

## Objetivo

O objetivo deste trabalho consiste em analisar alguns atributos característicos de vinhos tintos e vinhos brancos e determinar a qualidade deste vinho.

## Atributos

Os atributos do conjunto de dados são:

| Atributo  | Tipo |
| ------------- | ------------- |
|type                     |object|
|fixed acidity           |float64|
|volatile acidity        |float64|
|citric acid             |float64|
|residual sugar          |float64|
|chlorides               |float64|
|free sulfur dioxide     |float64|
|total sulfur dioxide    |float64|
|density                 |float64|
|pH                      |float64|
|sulphates               |float64|
|alcohol                  |object|
|quality                   |int64|

Cada atributo foi analisado e teve a sua importância para a predição da qualidade do vinho estudada.

# Análise Exploratória dos dados e Feature Engineering

A análise dos dados se deu em alguns pontos cruciais para o desenvolvimento do trabalho. Analisar cada atributo e verificar a consistência dos dados apresentados.

## Exploração

A primeira etapa de exploração é entender possíveis discrepâncias em valores de algumas linhas, outliers e tipos de dados não númericos em campos que deveriam ser e/ou campos que não deveriam ser numéricos e o são.

### Variável Target

Pode-se notar que a variável target, qualidade do vinho, tem sua concentração maior em torno do valor 6.

![Boxplot](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/Countplot_quality%20map.png)

### Formato desconhecido de algumas linhas no atributo "Alcohol"

A coluna contendo informações do teor alcólico apresentou valores com excesso de pontos decimais.

```
131.333.333.333.333
```
As 40 linhas que apresentavam tais valores foram removidas do conjunto de dados. A decisão de excluir tais valores e não apenas substituir por algum valor padrão se deu pelo fato de que não os números não apresentaram nenhuma caracteristica como fator multiplicativo ou mesmo algum divisor que o faria ter sentido. Para não adicionar bias ao modelo, a decisão de excluir as linhas foi tomada.

### Boxplot

Fora realizado um conjunto de boxplot para análise da distribuição dos dados em função da variável target. Pode-se ainda observar alguns outliers, mas somente da variável densidade chamou a atenção, pois alguns valores ultrapassavam em cem vezes a densidade da água.

![Boxplot](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/boxplot_quality.png)

A solução para o atributo densidade foi dividir estes valores por múltiplos de dez para que o range entre 0 e 1,1 fosse atingido. Por fim, a densidade teve a seguinte distribuição:

![Boxplot_after](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/boxplot_alcohol_fixed.png)

Após a alteração os valores ficaram próximos do que é sensato para valores de densidade.

### Feature Engineering


