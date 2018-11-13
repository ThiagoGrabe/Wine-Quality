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

# Feature Engineering

A análise dos dados se deu em alguns pontos cruciais para o desenvolvimento do trabalho. Analisar cada atributo e verificar a consistência dos dados apresentados.

## Exploração

A primeira etapa de exploração é entender possíveis discrepâncias em valores de algumas linhas, outliers e tipos de dados não númericos em campos que deveriam ser e/ou campos que não deveriam ser numéricos e o são.

### Variável Target

Pode-se notar que a variável target, qualidade do vinho, tem sua concentração maior em torno do valor 6.

![Boxplot](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/Countplot_quality%20map.png)

Além disso, uma visualização da distribuição dos dados em função da variável alvo foi feita para entender se as distribuições seguiam algum padrão específico além de distribuições normais.

![Boxplot](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/kde_plot_distribution.png)

Pode-se observar uma distribuição em suma normal dada a variável target.

### Feature Engineering

A coluna contendo informações do teor alcólico apresentou valores com excesso de pontos decimais.

```
131.333.333.333.333
```
As 40 linhas que apresentavam tais valores foram removidas do conjunto de dados. A decisão de excluir tais valores e não apenas substituir por algum valor padrão se deu pelo fato de que não os números não apresentaram nenhuma caracteristica como fator multiplicativo ou mesmo algum divisor que o faria ter sentido. Para não adicionar bias ao modelo, a decisão de excluir as linhas foi tomada.

### Distribuição dos dados

Fora realizado um conjunto de boxplot para análise da distribuição dos dados em função da variável target. Pode-se ainda observar alguns outliers, mas somente da variável densidade chamou a atenção, pois alguns valores ultrapassavam em cem vezes a densidade da água.

![Boxplot](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/boxplot_quality.png)

A solução para o atributo densidade foi dividir estes valores por múltiplos de dez para que o range entre 0 e 1,1 fosse atingido. Por fim, a densidade teve a seguinte distribuição:

![Boxplot_after](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/boxplot_alcohol_fixed.png)

Após a alteração os valores ficaram próximos do que é sensato para valores de densidade.

### Agrupamentos

Uma estratégia para auxiliar os modelos de classificação fora a criação de colunas com agrupamentos específicos dependendo da abordagem que se deseja. Foram realizados dois agrupamentos:

1. Agrupamento em função do campo de valores de qualidade;
2. Utilizando [PCA - Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) foi também gerada uma nova coluna que se relaciona com o número de clusters que podemos descrever a variância dos dados. Em suma, 99% da variância associada à variável target pode ser explicada por duas componentes principais.

A figura abaixo demostra a análise do número de clusteres feita utilizando PCA para esta determinação.

![PCA](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/PCA.png)

### Matrix de Correlações

Foi feito ainda um estudo de correlação entre os atributos.

![PCA](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/Correlation%20map.png)

As correlações que se destacam em função da qualidade do vinho são:

* Volatilidade da acidez;
* Cloretos;
* Dióxido de enxofre total;

Destacam-se ainda as correlações inversas entre os vinhos branco e vinho tinto em todos atributos.

