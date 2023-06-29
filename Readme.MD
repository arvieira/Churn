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
- Primeira fase
  - OK - Ler a base de dados
  - OK - Modificar a variável "classe" de 0, 1 e 2 para 0 ou 1 (no-churn e churn)
    - OK - 0 = churn
    - OK - 1 e 2 = no-churn
  - OK - Separar as variáveis de entrada e saída
  - OK - Balancear a base com undersampling aleatório para ter um V0
  - OK - Separar a base em 70/30 treino e teste
  - Treinar os modelos com cross-validation (k-fold)
    - OK - XGBoost
      - Verificar o cross-validation quando fizer com o SVM
    - PAREI AQUI - SVM
    - SVM com AdaBoost
  - Verificar o desempenho com a matriz de confução
    - Acurácia
    - Precision
    - Recall
    - F-score
- Segunda fase (Comparar os resultados a cada etapa com a primeira fase)
  - Aplicar os préprocessamentos já feitos
  - Utilizar Tomek link para o undersampling
