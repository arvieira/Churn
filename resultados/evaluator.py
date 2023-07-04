import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# Calcula e imprime a matriz de confusão para um modelo
def evaluate(alg, real, predicted):
    print(f"-> Avaliando o {alg} com matriz de confusão...")
    print(pd.DataFrame(confusion_matrix(real, predicted), columns=['No Churn Pred', 'Churn Pred'],
                       index=['No Churn Real', 'Churn Real']))
    print(metrics.classification_report(real, predicted, digits=5))