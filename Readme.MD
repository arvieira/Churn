# INFO
  - Detrator, Neutro, Promotor
    - 0 a 6 na coluna class detratores
    - 7 e 8 são neutros
    - 9 e 10 são promotores
  - Churn e non-churn
    - 0, 1 e 2 são churn
    - 8, 9 e 10 são no-churn

# DÚVIDAS
- O que são as diferentes separações da base?
  - Base completa: São as 190 variáveis
  - Base Carlos Alberto: São as 38 variáveis selecionadas pelo aluno
  - Base CFS: ??? (Base menor - 23 variáveis)
  - Base RELIEF F: ??? (Base média - 140 variáveis)
- Não foi feita seleção de variáveis. Apenas lançou a base com as 38 selecionadas.

# ROADMAP
- OK - Primeira fase
  - OK - Ler a base de dados
  - OK - Modificar a variável "classe" de 0, 1 e 2 para 0 ou 1 (no-churn e churn)
    - OK - 0 = churn
    - OK - 1 e 2 = no-churn
  - OK - Separar as variáveis de entrada e saída
  - OK - Balancear a base com undersampling aleatório para ter um V0
  - OK - Separar a base em 70/30 treino e teste
  - OK - Treinar os modelos
    - OK - XGBoost
    - OK - SVM
    - OK - SVM com AdaBoost
  - OK - Verificar o desempenho com a matriz de confução
    - OK - Acurácia
    - OK - Precision
    - OK - Recall
    - OK - F-score
    
- TODO: Fazer o GridSearchCV para SVM e SVM com AdaBoost
  - Inserir os resultados em preprocessamento_tomek_random_gridsearch_cv
- TODO: Melhorar o preprocessamento
  - Fazer seleção de variáveis
  - Remover outliers
  - Remover inconsistências
    
- Segunda fase (Comparar os resultados a cada etapa com a primeira fase)
  - Aplicar os préprocessamentos já feitos
    - OK - Missing Values: Não tinha
    - OK - Análise exploratória: Removeu linhas e colunas de (82618, 190) para (64292, 177), mas sem efeito no 
    resultado dos modelos
    - OK - Remoção de variáveis com mais de 40% de zeros, sem contar as binárias: Removeu colunas de (64292, 177) 
    para (64292, 120), mas sem efeito no resultado dos modelos
    - OK - Normalizar com o MinMax pq eu não removi os outliers ainda: Aparentemente melhorou um pouco, mas 
    tem que confirmar
    - Fazer seleção de variáveis
    - Remover outliers
    - Remover inconsistências
  - OK - Utilizar Tomek link para o undersampling
    - OK - Obteve uma melhora para o SVM. O SVM com AdaBoost ainda não está funcionando direito
    - OK - Observação sobre o tomeklink. Ele somente remove os casos da classe majoritária que estão próximos demais
    da minoritária. Caso não tenha elementos suficientes, ele retorna uma base desbalanceada. 
    - OK - Utilizar o random para terminar de excluir elementos da classe majoritária e balancear a base.
  - Usar o GridSearchCV para busca de hyperparâmetros
    - O CV do GridSearch já possui o cross-validation