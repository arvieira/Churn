import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn import metrics


# Desenha a curva ROC
def plot_roc(axis, y_test, pred_proba, title):
    fpr, tpr, thresh = metrics.roc_curve(y_test, pred_proba)
    auc = metrics.roc_auc_score(y_test, pred_proba)
    axis.plot(fpr, tpr, label=f"{title} AUC={auc:.3f}")

    axis.set_title('ROC Curve')
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.legend(loc=0)


# Calcula e imprime a matriz de confusão para um modelo
def evaluate(alg, real, predicted, pred_proba):
    # Matriz de confusão
    print(f"-> Avaliando o {alg} com matriz de confusão...")
    cm = confusion_matrix(real, predicted)
    print(pd.DataFrame(cm, columns=['No Churn Pred', 'Churn Pred'],
                       index=['No Churn Real', 'Churn Real']))
    print(metrics.classification_report(real, predicted, digits=5))

    # AUC
    # print(f'-> Acurácia: {accuracy_score(real, predicted)}')
    print(f"-> Valor AUC: {metrics.roc_auc_score(real, pred_proba)}")

    # Exibindo a matriz de confusão para colocar no artigo
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot()
    plt.show()

    # Plot all ROC into one graph
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plot_roc(ax, real, pred_proba, alg)
    plt.show()

    return accuracy_score(real, predicted), metrics.roc_auc_score(real, pred_proba)
