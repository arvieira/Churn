import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier


max_iter = 3


def best_model(ds, name):
    parameter_space = {
        # 'hidden_layer_sizes': [(1), (2), (3), (4), (5),(6), (7), (8), (9), (10), (11),(12)],
        'hidden_layer_sizes': [(1)],
        'activation': ['tanh', 'relu'],  # 'relu', 'logistic'],
        'solver': ['sgd', 'adam'],  # , 'adam', 'lbfgs'],
        'alpha': [0.001, 0.005],  # , 0.005],
        'learning_rate': ['constant', 'adaptative'],  # ,'adaptive'],
        'nesterovs_momentum': [True],  # , False],
        'beta_1': [0.95], 'beta_2': [0.999],
        'learning_rate_init': [0.01, 0.05],  # , 0.02, 0.05, 0.1],
        'momentum': [0.9, 0.95]  # 0.85, 0.9, 0.95]
    }

    # Criando e escrevendo em arquivos de texto (modo 'w').
    arquivo = open('saidaMLP.txt', 'w')

    print("\nlearning on dataset %s" % name)
    X_train, y_train = ds[0]
    X_test, y_test = ds[1]

    # for each dataset, plot learning for each learning strategy
    mlps = []

    print("\nlearning on dataset %s" % name)
    X, y = ds[0]
    X_test, y_test = ds[1]
    iteracao = 100

    # print("training: %s" % label)
    mlp = MLPClassifier(validation_fraction=0.2, early_stopping=True,
                        n_iter_no_change=10, max_iter=iteracao,
                        hidden_layer_sizes=(1),
                        activation='tanh',  # 'relu', 'logistic'],
                        solver='adam',  # 'sgd','adam', 'lbfgs'],
                        alpha=0.001,  # 0.001, 0.005],
                        learning_rate='adaptative',  # 'constant', 'adaptive'],
                        nesterovs_momentum=True,  # , False],
                        beta_1=0.95, beta_2=0.999,
                        learning_rate_init=0.01,  # 0.01, 0.02, 0.05, 0.1],
                        momentum=0.9  # 0.85, 0.9, 0.95]
                        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                module="sklearn")
        for i in range(0, max_iter):
            mlp.fit(X, y)
            mlps.append(mlp)
    ypred = []
    acc = []

    for i in range(len(mlps)):
        ypred.append(mlps[i].predict(X))
        acc.append(accuracy_score(y, ypred[i]))

    # retorna indice relativo ao modelo com a melhor acurácia
    ind_acc = np.argmax(acc)

    # armazena a melhor instancia do modelo
    mlps_best = (mlps[ind_acc])
    ypred_best = ypred[ind_acc]

    # mlps.append(mlp)
    print("training")
    # print("training: %s" % params)
    print("Training set accuracy: %f" % mlps_best.score(X, y))
    # print("Training set loss: %f" % mlps[ind_acc].loss_)
    acc = accuracy_score(y, ypred_best, normalize=False)
    precision = precision_score(y, ypred_best)
    f1 = f1_score(y, ypred_best, average='micro')
    recall = recall_score(y, ypred_best, average='weighted')
    cm = confusion_matrix(y, ypred_best)
    print("accuracy", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1", f1)
    print("cm", cm)
    print('------------------------')

    print('-------RESULTADOS TESTE---------')
    X_test, y_test = ds[1]
    y_test_pred = mlps_best.predict(X_test)

    print("testing")
    # print("testing: %s" % p)
    # invert data transforms on forecast
    acc = accuracy_score(y_test, y_test_pred, normalize=False)
    precision = precision_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='micro')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    cm = confusion_matrix(y_test, y_test_pred)
    print("accuracy", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1", f1)
    print("cm", cm)

    np.set_printoptions(precision=2)
    class_names = []
    class_names.append("Class 0")
    class_names.append("Class 1")
    class_names.append("Class 2")

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.ax_.set_title(title)

        print(title)
        print(disp.plot())

    plt.show()
