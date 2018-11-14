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

Para cada modelo de classificação os treinamentos e testes foram realizados dentro de um mesmo método e os resultados são apresentados no formato tabular para cada modelo e métrica associada.

Como houveram duas abordagens para o problema (tipos de vinhos avaliados separadamente e modelo único para os vinhos), houveram três treinamentos e, consequentemente, três resultados que serão apresentados abaixo:

#### Resultados de treino para os dados de *Vinhos Brancos*

| Model for   White Wine | Precision (Training) | Accuracy Score (Training) | Recall (Training) | Precision (Test) | Accuracy Score (Test) | Recall (Test) | Training Set | Testing Set |
|:----------------------:|:--------------------:|:-------------------------:|:-----------------:|:----------------:|:---------------------:|:-------------:|:------------:|:-----------:|
|      XGBClassifier     |       0.870611       |          0.689026         |      0.746337     |     0.726032     |        0.620761       |    0.634778   |  (3891, 13)  |  (973, 13)  |
|   LogisticRegression   |        0.51987       |          0.561809         |      0.358286     |     0.478961     |        0.586845       |    0.411973   |  (3891, 13)  |  (973, 13)  |
|           SVC          |       0.777663       |          0.802621         |      0.507582     |     0.449035     |        0.558068       |    0.299136   |  (3891, 13)  |  (973, 13)  |
| DecisionTreeClassifier |           1          |             1             |         1         |     0.425347     |        0.611511       |    0.416755   |  (3891, 13)  |  (973, 13)  |
|  KNeighborsClassifier  |       0.552262       |          0.646877         |      0.374798     |     0.301815     |        0.459404       |    0.260629   |  (3891, 13)  |  (973, 13)  |
|       GaussianNB       |       0.617573       |          0.482652         |      0.603063     |     0.632883     |        0.508736       |    0.656236   |  (3891, 13)  |  (973, 13)  |

#### Resultados de treino para os dados de *Vinhos Tintos*

|  Model for   Red Wine  | Precision (Training) | Accuracy Score (Training) | Recall (Training) | Precision (test) | Accuracy Score (test) | Recall (test) | Training Set | Testing Set |
|:----------------------:|:--------------------:|:-------------------------:|:-----------------:|:----------------:|:---------------------:|:-------------:|:------------:|:-----------:|
|      XGBClassifier     |       0.918741       |          0.820251         |      0.809426     |     0.605677     |        0.68652        |    0.57752    |  (1274, 13)  |  (319, 13)  |
|   LogisticRegression   |       0.632669       |          0.598901         |      0.448074     |      0.49985     |        0.586207       |    0.422588   |  (1274, 13)  |  (319, 13)  |
|           SVC          |       0.552841       |          0.732339         |      0.350718     |      0.37491     |        0.583072       |    0.306665   |  (1274, 13)  |  (319, 13)  |
| DecisionTreeClassifier |           1          |             1             |         1         |     0.605483     |        0.648903       |    0.606059   |  (1274, 13)  |  (319, 13)  |
|  KNeighborsClassifier  |       0.672606       |          0.66248          |      0.387217     |     0.313748     |        0.501567       |    0.27319    |  (1274, 13)  |  (319, 13)  |
|       GaussianNB       |        0.64173       |          0.533752         |      0.659428     |     0.543269     |        0.539185       |    0.526828   |  (1274, 13)  |  (319, 13)  |

#### Resultados de treino para os dados dos vinhos branco e tinto

| Model   for Both Wine Types | Precision (Training) | Accuracy Score (Training) | Recall (Training) | Precision (test) | Accuracy Score (test) | Recall (test) | Training Set | Testing Set |
|:---------------------------:|:--------------------:|:-------------------------:|:-----------------:|:----------------:|:---------------------:|:-------------:|:------------:|:-----------:|
|        XGBClassifier        |       0.858103       |          0.66273          |      0.704499     |     0.636081     |        0.614551       |    0.526602   |  (5165, 15)  |  (1292, 15) |
|      LogisticRegression     |       0.662984       |          0.566699         |      0.371123     |     0.391837     |        0.571981       |    0.35126    |  (5165, 15)  |  (1292, 15) |
|             SVC             |       0.758254       |          0.75547          |      0.43015      |      0.53991     |        0.56192        |    0.257796   |  (5165, 15)  |  (1292, 15) |
|    DecisionTreeClassifier   |           1          |             1             |         1         |     0.595025     |        0.655573       |    0.535367   |  (5165, 15)  |  (1292, 15) |
|     KNeighborsClassifier    |       0.524822       |          0.638529         |      0.355194     |     0.280395     |        0.499226       |    0.229848   |  (5165, 15)  |  (1292, 15) |
|          GaussianNB         |       0.512314       |          0.468151         |      0.59695      |     0.513026     |        0.487616       |    0.486432   |  (5165, 15)  |  (1292, 15) |

### Considerações finais na escolha do modelo

Pode-se verificar que o *XGBClassifier* apresentou a melhor desempenhoao se analisar a precisão e acurácia. Um ponto bem importante é que o modelo *Decision Tree Classifier* apresentou um desempnho de acurácia superior para o modelo e que todos os vinhos estão contidos no conjunto de dados. Porém, pode-se observar um problema de *overfitting* em todos os casos, haja vista uma pontuação de treinamento igual a 1 em todos os casos e uma redução considerável no conjunto de testes. Vale ressaltar que, tendo em vista este cenário, um modelo com este comportamento de *overfitting* não teria um desempenho aceitável quando uma nova entrada de dados ocorrer.

#### Modelo escolhido: XGBoost


O *[XGBoost](https://github.com/dmlc/xgboost)* (eXtreme Gradient Boosting) é uma conhecida e eficiente implementação de código aberto do algoritmo baseado em árvores de aumento de gradiente. O aumento de gradiente é um algoritmo de aprendizagem supervisionada que tenta prever com precisão uma variável de destino. Para isso, combina as estimativas de um conjunto de modelos mais simples e mais fracos. O XGBoost tem excelente desempenho em competições de machine learning, pois é mais robusto ao lidar com uma variedade de tipos de dados, relacionamentos e distribuições, bem como com um grande número de hiperparâmetros que podem ser aperfeiçoados e ajustadas para um cenário mais apropriado. Essa flexibilidade faz do XGBoost uma escolha consistente para problemas de regressão, classificação (binária e multiclasse) e pontuação.
[Referência](https://docs.aws.amazon.com/pt_br/sagemaker/latest/dg/xgboost.html)

### Tuning dos modelos

Nesta etapa um aprimoramento dos modelos será realizado utilizando um algoritmo de busca dos melhores hiperparâmetros para um dado modelo. O algoritmo escolhido é o *[Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)*. Esta técnica realiza uma busca exaustiva entre uma lista especificada de hiperparâmetros para o modelo. Uma métrica deve ser associada para avaliação e, no caso deste trabalho, a métrica utilizada foi a acurácia.

#### Hyperparameters (Hiperparâmetros)

O *[XGBoost](https://github.com/dmlc/xgboost)* possui alguns hiperparâmetros interessantes para classificação. Os escolhidos para esete trabalho foram:

| Hiperparâmetro  | Valores Testados |
| ------------- | ------------- |
| n_estimators           |100, 150, 200|
| learning_rate        |0.05, 0.07, 0.4|
|colsample_bytree             |0.5, 0.7, 1|
|max_depth             |5, 7, 10|
|subsample             |0.6, 0.7, 0.8|
|gamma             |0, 0.1, 0.3|

A *[AWS](https://docs.aws.amazon.com/pt_br)* possui um pacote do *[XGBoost](https://github.com/dmlc/xgboost)* em sua solução de Machine Learning. Ela apresenta uma explicação interessante dos hiperparâmetros:

| Hiperparâmetro   | Descrição                                                                                                                                                                                                                                                         |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gamma            | A redução de perda mínima necessária para fazer uma partição adicional em um nó de folha da árvore. Quanto maior for o parâmetro, mais conservador será o algoritmo.Valores válidos: flutuante. Intervalo: [0,∞).Valor padrão: 0                                  |
| max_depth        | A profundidade máxima de uma árvore. Aumentar esse valor torna o modelo mais complexo e propenso a sofrer sobreajuste. 0 indica que não há limite. Um limite é necessário quando grow_policy=depth-wise.Valores válidos: inteiro. Intervalo: [0,∞)Valor padrão: 6 |
| subsample        | Taxa de subsampling da instância de treinamento. Se você configurá-la como 0,5, o XGBoost aleatoriamente coletará metade das instâncias de dados para expandir as árvores. Isso evita o sobreajuste.Valores válidos: flutuante. Intervalo: [0,1].Valor padrão: 1  |
| colsample_bytree | Taxa de subsampling de colunas ao criar cada árvore.Valores válidos: flutuante. Intervalo: [0,1].Valor padrão: 1                                                                                                                                                  |
| n_estimators     | Número de árvores no algoritmo. Intervalo: [0,∞) Valor padrão 100                                                                                                                                                                                                 |
| learning_rate    | Taxa de aprendizado ou peso para fator de correção às novas árvores. Intervalo [0,1]. Valor padrão 0.1                                                                                                                                                            |
|                  |                                                                                                                                                                                                                                                                   |

#### Resultado da otimização dos hiperparâmetros

Os valores ótimos encontrados para os hiperparâmetros foram:

##### Vinho Tinto

| Hiperparâmetro  | Valores Testados |
| ------------- | ------------- |
| n_estimators           |100|
| learning_rate        |0.07|
|colsample_bytree             |0.7|
|max_depth             |7|
|subsample             |0.8|
|gamma             |0|

#### Vinho Branco

| Hiperparâmetro  | Valores Testados |
| ------------- | ------------- |
| n_estimators           |200|
| learning_rate        |0.05|
|colsample_bytree             |0.5|
|max_depth             |10|
|subsample             |0.8|
|gamma             |0.1|

#### Vinhos Branco e Tinto

| Hiperparâmetro  | Valores Testados |
| ------------- | ------------- |
| n_estimators           |200|
| learning_rate        |0.05|
|colsample_bytree             |0.7|
|max_depth             |10|
|subsample             |0.8|
|gamma             |0|

### Modelos Finais

Após a otimização dos hiperparâmetros de cada modelo, os três modelos foram treinados e as métricas novamente avaliadas. Um ponto importante neste passo é verificar se o modelo final é suficientemente bom para atender requisitos de clientes. Para este ponto, uma matriz de confunção foi feita para entender a quantidade de falsos positivos e falsos negativos o modelo consegue captar.

#### Métricas finais: Modelo *Vinhos Tintos*

| Modelos | Precision | Accuracy | Recall |
|:-------:|:---------:|:--------:|:------:|
| Inicial |   0.6057  |  0.6865  | 0.5775 |
|  Final  |   0.6502  |  0.7398  | 0.6318 |

#### Métricas finais: Modelo *Vinhos Brancos*

| Modelos | Precision | Accuracy | Recall |
|:-------:|:---------:|:--------:|:------:|
| Inicial |   0.7260  |  0.6208  | 0.6348 |
|  Final  |   0.8106  |  0.7235  | 0.7010 |

#### Métricas finais: Modelo *Vinhos Tintos e Brancos*

| Modelos | Precision | Accuracy | Recall |
|:-------:|:---------:|:--------:|:------:|
| Inicial |   0.6361  |  0.6146  | 0.5266 |
|  Final  |   0.6948  |  0.7353  | 0.6198 |

Em todos os casos a otimização dos modelos obteve um ganho considerável em todas as métricas propostas.

Em um cenário de vendas de vinhos tintos e brancos, podemos pensar que falsos negativos (vinhos classificados como piores do que realmente são) possuem pesos menores do que falsos positivos (vinhos classificados como melhores do que realmente são). Esta abordagem é coerente com o seguinte pensamento:

>*"Enquanto cliente eu não gostaria de comprar ou degustar um vinho que seja classificado com ótimo e, na realidade, ele é ruim. Em contrapartida, o cliente não teria problemas em comprar ou degustar um vinho classificado como médio e, na realidade, ele é excelente"*

Esta ideia pode ser validade com matrizes de confusão para cada modelo:

| ![redwine_confusionmatrix_final](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/confusion_matrix_redwine_final.png) | 
|:--:| 
| *Matriz de confusão para o modelo de vinhos tintos* |


| ![whitewine_confusionmatrix_final](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/confusion_matrix_whitewine_final.png) | 
|:--:| 
| *Matriz de confusão para o modelo de vinhos brancos* |

| ![wines_confusionmatrix_final](https://github.com/ThiagoGrabe/Wine-Quality/blob/master/Images/confusion_matrix_wines_final.png) | 
|:--:| 
| *Matriz de confusão para o modelo de vinhos tintos e brancos* |

Pode-se observar pelas matrizes que para os todos os modelos, a classificação em sua maioria é feita de forma correta com alguns falsos positivos e falsos negativos.

### Conclusão

Pode-se concluir pelo estudo que os modelos criados apresentam boa representação dos dados pela análise das métricas escolhidas e, principalmente, pelos falsos positivos representarem uma fatia pequena do conjunto total de dados.

Ainda pelo estudo considero que, caso haja possibilidade, seja utilizado um modelo distinto para cada tipo de vinho. Ao utilizar o modelo geral para vinhos tintos e brancos em vinhos brancos temos um decréscimo de desempenho do modelo. Entretanto, deve-se entender como o modelo seria aplicado em ambientes de produção.

Por fim, há bastante espaço para melhorar o desempenho dos modelos. Acredito que uma acurácia em torno de 0.75 pontos seja ainda baixa para um modelo de classificação. Alguns pontos de melhoria estão listados abaixo:

* Uma análise dos atributos mais profunda. Entender *outliers* em todas as *features* e decidir se devemos trabalhar em cima destes valores. Enfim, realizar um *Feature Engineering* mais elaborado abordando várias hipóteses.
* Aumentar a base de dados é sempre um caminho que tende a ser favorável no desempenho do modelo. Sendo assim, adicionar mais dados e atributos relavantes seja uma ótima forma de melhorar os modelos.
* Realizar um estudo mais aprofundado de outro algoritmo de classificação. O *Support Vector Machine* se mostrou promissor e com uma otimização de hiperparâmetros e os selecionando bem, pode-se obter bons resultados.
* Realizar um outro estudo com outras métricas que podem trazer *insights* diferentes da abordagem dada ao problema. Pode-se pensar em *F1 Score*, *F-Beta* e *ROC Curve*.
