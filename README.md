# Wine-Quality
Este repositório contém um projeto de análise de dados e algoritmos de aprendizado de máquina para predizer/classificar a qualidade
de um vinho dado alguns parâmetros químicos.

# Introdução

Vinhos são produtos amplamente consumidos em qualquer cultura. Tanto o vinho branco quanto o vinho tinto possuem características peculiares
que podem ser apreciadas em diversos ambientes e ocasiões sociais. De acordo com alguns [sites especializados](https://vinepair.com/articles/chemical-compounds-wine-taste-smell/)
características químicas do vinho como teor alcólico e a presença de ácidos podem representar um bom vinho mas também caracterizá-lo como um vinho de baixa qualidade.

O projeto será abordado como um problema de classificação no qual a variável desejada é a qualidade, entre 0 e 10, dos vinhos.

# Objetivo

O objetivo deste trabalho consiste em analisar alguns atributos característicos de vinhos tintos e vinhos brancos e determinar a qualidade deste vinho.

# Modelagem

Para o modelagem do problema proposto, primeiramente foi identificado que se trata de uma classificação em torno da variável target, dada a discretização (0 a 10) dos valores de qualidade dos vinhos.

A partir deste ponto, uma estratégia de feature engineering foi criada para entender correlações entre os atributos e possíveis outliers no conjunto de dados. Outliers são normalizados (atributo densidade) para um range aceitável. Feito esta abordagem de eliminação de  outliers, uma avaliação de dados faltantes e fora de padrão em cada linhas do dataset é realizado. É possível com essa abordagem eliminar dados não padronizados, como strings em lugar de números e caracteres desconhecidos, por exemplo. 

Após uma análise introdutória e correção de pontos no dataset, uma técnica de [PCA - Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) é aplicada para agrupar em função das componentes principais o conjunto de dados, adicionado assim atributos relevantes para auxiliar na tarefa de classificação dos modelos candidatos.

Um ponto chave para o trabalho é criar dois modelos: um para cada tipo de vinho. Esta abordagem permite um desempenho melhor para ambos os tipos de vinho durante a etapa de treino e teste dos modelos. Um indício desta boa prática é a correlação totalmente inversa entre os tipos de vinho.

Para avaliar estes modelos e por se tratar de uma tarefa de classificação, as seguintes métricas foram escolhidas:

* Precisão (**Precision**)
* Acurácia (**Accuracy**)
* Revocação (**Recall**)

Essas métricas permitem ter uma visão geral da classificação realizada pelo modelo. A acurácia permite avaliar a quantidade de acertos em relação ao número de observações. Essa métrica nos dá uma visão geral do quão o modelo está condizente na tarefa de classificação. Já a precisão tem o objetivo de estabelecer o quanto o seu modelo está desempenhando bem a função para a qual ele foi feito. Por fim, a revocação é importante quando falsos negativos estão presentes na predição dos valores.

Seis algoritmos de classificação foram testados no conjunto de dados para então se definir o melhor entre os seis que tiver um desempenho melhor nas métricas estabelecidas:

* [XGBClassifier](https://xgboost.readthedocs.io/en/latest/index.html)
* [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
* [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)


Dada as circunstâncias do problema de qualidade de vinhos, para se escolher o modelo a ser utilizado as métricas de **precisão** e **acurácia** serão predominantes na análise, pois falsos positivos tem um impacto maior do que falsos negativos. Esta linha de pensamento se dá, pois um cliente que adquiri um vinho de qualidade alta e acaba tendo um vinho de menor qualidade, tem uma experiência negativa com a empresa/marca. Por outro lado, um falso negativo atesta que um vinho classificado com menor qualidade na classificação do modelo, mas que apresenta uma qualidade superior, acaba proporcionando ao cliente uma melhor experiência com a marca.

Por fim, os hiperparâmetros serão refinados utilizando algoritmo [Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV). Os seguintes hiperparâmetros serão refinados:

* n_estimators
* learning_rate
* colsample_bytree
* max_depth
* subsample
* gamma

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

Cada atributo foi analisado e teve a sua importância para a predição da qualidade do vinho estudada. Esses dados, ainda crus, serão tratados na etapa de **Feature Engineering**

# Feature Engineering

A análise dos dados se deu em alguns pontos cruciais para o desenvolvimento do trabalho. Analisar cada atributo e verificar a consistência dos dados apresentados.

## Exploração

A primeira etapa de exploração é entender possíveis discrepâncias em valores de algumas linhas, outliers e tipos de dados não númericos em campos que deveriam ser e/ou campos que não deveriam ser numéricos e o são.

### Variável Target

Pode-se notar que a variável target, qualidade do vinho, tem sua concentração maior em torno do valor 6.

![countplot](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/Countplot_quality%20map.png)

Além disso, uma visualização da distribuição dos dados em função da variável alvo foi feita para entender se as distribuições seguiam algum padrão específico além de distribuições normais.

![KDE](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/kde_plot_distribution2.png)

Pode-se observar uma distribuição em suma normal dada a variável target.

## Feature Engineering

### Data cleaning

A coluna contendo informações do teor alcólico apresentou valores com excesso de pontos decimais.

```
131.333.333.333.333
```
As 40 linhas que apresentavam tais valores foram removidas do conjunto de dados. A decisão de excluir tais valores e não apenas substituir por algum valor padrão se deu pelo fato de que não os números não apresentaram nenhuma caracteristica como fator multiplicativo ou mesmo algum divisor que o faria ter sentido. Para não adicionar bias ao modelo, a decisão de excluir as linhas foi tomada.

Fora realizado um conjunto de boxplot para análise da distribuição dos dados em função da variável target. Pode-se ainda observar alguns outliers, mas somente da variável densidade chamou a atenção, pois alguns valores ultrapassavam em cem vezes a densidade da água.

![Boxplot](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/boxplot_quality.png)

A solução para o atributo densidade foi dividir estes valores por múltiplos de dez para que o range entre 0 e 1,1 fosse atingido. Por fim, a densidade teve a seguinte distribuição:

![Boxplot_after](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/boxplot_alcohol_fixed.png)

Após a alteração os valores ficaram próximos do que é sensato para valores de densidade.

### Clustering

Uma estratégia para auxiliar os modelos de classificação fora a criação de colunas com agrupamentos específicos dependendo da abordagem que se deseja. Foram realizados dois agrupamentos:

1. Agrupamento em função do campo de valores de qualidade;
2. Utilizando [PCA - Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) foi também gerada uma nova coluna que se relaciona com o número de clusters que podemos descrever a variância dos dados. Em suma, 99% da variância associada à variável target pode ser explicada por duas componentes principais.

A figura abaixo demostra a análise do número de clusteres feita utilizando PCA para esta determinação.

![PCA](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/PCA.png)

### Matriz de Correlações

Foi feito ainda um estudo de correlação entre os atributos.

![corrmap](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/Correlation%20map2.png)

As correlações que se destacam em função da qualidade do vinho são:

* Volatilidade da acidez;
* Cloretos;
* Dióxido de enxofre total;

Destacam-se ainda as correlações inversas entre os vinhos branco e tinto em todos atributos.

### Considerações Finais - Data Engineering

Após uma análise exploratória dos dados e uma limpeza em algumas *features*, foram gerados datasets para a criação dos modelos de Machine Learning considerando a classificação da qualidade dos vinhos.

A matriz de correlação traz uma fundamental informação sobre a correlação inversa dos vinhos tinto e branco em relação aos atributos do conjunto de dados.

Duas etapas de *Data Cleaning* foram feitas: a primeira na remoção de números não padronizados do atributo teor alcólico e a segunda foi a normalização de algumas densidades que apresentavam valores muito altos para os valores aceitáveis.

Além disso, as três principais *features* são o teor alcólico, a quantidade de sulfatos e a acidez fixa dos vinhos.

Pode-se ainda agrupar os dados conforme um algoritmo de *Principal Component Analysis* que mostrou que a 99% variância da qualidade do vinho pode ser explicada por três componentes. Realizando os *Clusteres* necessários foi possível melhorar o desempenho do algoritmo de aprendizagem de máquina.

## Machine Learning

Nesta etapa do trabalho, algoritmos de classificação são escolhidos e testados para entender qual tem o melhor desempenho segundo algumas métricas específicas. Uma estratégia ainda foi dividir o conjunto de dados original

### Modelos de classificação

Para a tarefa de classificação os seguintes modelos foram escolhidos como possíveis candidatos para classificar a qualidade dos vinhos tinto e branco em uma escala de 0 a 10:

* [XGBClassifier](https://xgboost.readthedocs.io/en/latest/index.html)
* [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
* [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

Esses modelos apresentam características distantas na forma de realizar a classificação dos dados. Destacam-se *XGBoost* e *Support Vector Machine* que apresentam ótimos resultados quando se pesquisa em algumas referências:

[Article 01](https://www.quora.com/What-is-better-k-nearest-neighbors-algorithm-k-NN-or-Support-Vector-Machine-SVM-classifier-Which-algorithm-is-mostly-used-practically-Which-algorithm-guarantees-reliable-detection-in-unpredictable-situations) - Comparison KNN and SVM

[Article 02](https://www.quora.com/What-are-the-advantages-disadvantages-of-using-Gradient-Boosting-over-Random-Forests) - Advantages and disadvantages of using Gradient Boosting over Random Forest.

### Métricas

Para avaliar estes modelos e por se tratar de uma tarefa de classificação, as seguintes métricas foram escolhidas:

* Precisão (**Precision**)
* Acurácia (**Accuracy**)
* Revocação (**Recall**)

Tanto a precisão quanto acurácia dos modelos serão avaliadas e a revocação será utilizada como auxiliar. Por fim, será estabelicido uma [matrix de confusão](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) para validar o modelo e analisar os falsos positivos e falsos negativos.

### Conjunto de dados final

Após toda a etapa de pre processamento do conjunto de dados, um dataset com alguns atributos adicionais será utilizado para treinar e testar o modelo. Os atributos são:

| Atributo  | Tipo |
| ------------- | ------------- |
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
|alcohol                 |float64|
|quality                   |int64|
|Red                       |int64|
|White                    | int64|
|group_quality             |int64|
|Clusters PCA              |int64|

Para treinamento foram utilizadas as seguintes proporções dos dados para treinamento e teste dos modelos:

| Treino  | Teste |
| ------------- | ------------- |
|80%          |20%|

Esta divisão foi feita utilizando um algoritmo chamado *[Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)* que desempenha esta função de dividir o conjunto de dados de forma a obedecer as proporções definidas. 

Para cada treino foi estabelecido uma *seed* para repetitibilidade do processo.

### Treinamento e teste

