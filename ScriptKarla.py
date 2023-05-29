# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:29:48 2021

@author: Karla
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from keras import callbacks, optimizers

# métricas
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# outros importes
from sklearn.datasets import make_circles
from sklearn.metrics import precision_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from itertools import combinations
from math import factorial
from scipy.stats import t
from pickle import dump

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping
from sklearn.metrics import fbeta_score, make_scorer

import requests
import json
import warnings

warnings.filterwarnings("ignore")


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


def floresta(ds, name):
    arquivo = open('saidaRF.txt', 'w')
    print("\nlearning on dataset %s" % name)
    X_train, y_train = ds[0]
    X_test, y_test = ds[1]

    # scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    scorers = {
        # 'precision_macro': make_scorer(precision_score),
        # 'recall_macro': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    param_search = {'min_samples_split': [10, 20],
                    'min_samples_leaf': [10, 20],
                    'max_depth': [10, 15, 20],
                    'max_features': [20, 30, 38]
                    }

    estimators = []

    flr = RandomForestClassifier(n_estimators=100)
    # estimators.append(('RF', RandomForestClassifier(n_estimators=100)))
    # model = Pipeline(estimators)

    refit_score = 'accuracy_score'

    # skf = StratifiedKFold(n_splits=5)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    scoring = 'accuracy'

    # efit an estimator using the best found parameters on the whole dataset.
    search = GridSearchCV(estimator=flr, param_grid=param_search,
                          scoring=scoring, return_train_score=True, cv=cv, verbose=1)

    search_result = search.fit(X_train, y_train)
    dump(search_result, open('BestClassifierModel_RF.pkl', 'wb'))

    y_train_pred = search_result.predict(X_train)

    print("Best parameters set found on development set:")
    print()
    print(search_result.best_params_)
    print(search_result.best_score_)
    arquivo.write("Best parameters set found on development set:")
    arquivo.write(str("----------------------------------------"))
    arquivo.write(str(search_result.best_params_))
    arquivo.write(str(search_result.best_score_))

    print('\nConfusion matrix of MLP optimized for {} on the TRAIN DATA:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_train, y_train_pred),
                       columns=['class1', 'class2'], index=['0', '1']))
    arquivo.write(str('Confusion matrix of MLP optimized for {} on the TRAIN DATA:'.format(refit_score)))

    arquivo.write(str("----------------------------------------"))
    arquivo.write(str(pd.DataFrame(confusion_matrix(y_train, y_train_pred),
                                   columns=['class1', 'class2'], index=['0', '1'])))

    # Print the precision and recall, among other metrics
    print("Metrics Classification Report", metrics.classification_report(y_train, y_train_pred, digits=4))
    arquivo.write("Metrics Classification Report")
    arquivo.write(str("----------------------------------------"))
    arquivo.write(str(metrics.classification_report(y_train, y_train_pred, digits=4)))

    results_df = pd.DataFrame(search_result.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("max_depth")
    # results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]

    # create df of model scores ordered by performance
    model_scores = results_df.filter(regex=r"split\d*_test_score")

    # plot 30 examples of dependency between cv fold and AUC scores
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:30],
        dashes=False,
        palette="Set1",
        marker="o",
        alpha=0.5,
        ax=ax,
    )
    ax.set_xlabel("CV test fold", size=12, labelpad=6)
    ax.set_ylabel("Model AUC", size=12)
    ax.tick_params(direction='out', bottom=True, labelbottom=False)
    plt.show()
    # print correlation of AUC scores across folds
    print(f"Correlation of models:\n {model_scores.transpose().corr()}")

    # ------------------------------------------------------------
    # AVALIA SE OS MODELOS SÃO SEMELHANTES
    # SE p<0.05 eles são difentes

    # avaliacao da correlacao entre os resultados
    model_1_scores_RF = model_scores.iloc[0].values  # scores of the best model
    model_2_scores_RF = model_scores.iloc[1].values  # scores of the second-best model

    differences = model_1_scores_RF - model_2_scores_RF

    n = differences.shape[0]  # number of test sets
    df = n - 1
    n_train = len(list(cv.split(X_train, y_train))[0][0])
    n_test = len(list(cv.split(X_test, y_test))[0][1])

    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    print(f"Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}")
    arquivo.write(str(f"Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}"))
    arquivo.write(str("----------------------------------------"))

    # ------------------------------------------------------------
    # AVALIA SE OS MODELOS SÃO SEMELHANTES
    # SE p<0.05 eles são difentes
    t_stat_uncorrected = np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
    p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)

    print(
        f"Uncorrected t-value: {t_stat_uncorrected:.3f}\n"
        f"Uncorrected p-value: {p_val_uncorrected:.3f}"
    )

    arquivo.write("Uncorrected t-value:")
    arquivo.write(str(t_stat_uncorrected))
    arquivo.write("Uncorrected p-value: ")
    arquivo.write(str(p_val_uncorrected))

    # ------------------------------------------------------------
    n_comparisons = factorial(len(model_scores)) / (
            factorial(2) * factorial(len(model_scores) - 2)
    )
    pairwise_t_test = []

    for model_i, model_k in combinations(range(len(model_scores)), 2):
        model_i_scores = model_scores.iloc[model_i].values
        model_k_scores = model_scores.iloc[model_k].values
        differences = model_i_scores - model_k_scores
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_t_test.append(
            [model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val]
        )

    pairwise_comp_df = pd.DataFrame(
        pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
    ).round(3)
    pairwise_comp_df

    # ------------------------------------------------------------

    # make the predictions test set
    y_pred = search_result.predict(X_test)

    print('test recall_micro = ', recall_score(y_test, y_pred, average='micro'))
    print('test recall_macro = ', recall_score(y_test, y_pred, average='macro'))

    arquivo.write('test recall_micro = ')
    arquivo.write(str(recall_score(y_test, y_pred, average='micro')))

    arquivo.write('test recall_macro = ')
    arquivo.write(str(recall_score(y_test, y_pred, average='macro')))

    print('test precision_micro = ', precision_score(y_test, y_pred, average='micro'))
    print('test precision_macro = ', precision_score(y_test, y_pred, average='macro'))

    arquivo.write('test precision_micro = ')
    arquivo.write(str(precision_score(y_test, y_pred, average='micro')))

    arquivo.write('test precision_macro = ')
    arquivo.write(str(precision_score(y_test, y_pred, average='macro')))

    print('Best params for {}'.format(refit_score))
    print(search_result.best_params_)

    arquivo.write(str('Best params for {}'.format(refit_score)))
    arquivo.write(str(search_result.best_params_))

    # confusion matrix on the test data.
    print('\nConfusion matrix of MLP optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       columns=['class1', 'class2'], index=['0', '1']))

    arquivo.write(str('\nConfusion matrix of MLP optimized for {} on the test data:'.format(refit_score)))
    arquivo.write(str(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                   columns=['class1', 'class2'], index=['0', '1'])))

    # Print the precision and recall, among other metrics

    print('Metrics Classification Report')
    print(metrics.classification_report(y_test, y_pred, digits=2))

    arquivo.write('Metrics Classification Report')
    arquivo.write(str(metrics.classification_report(y_test, y_pred, digits=2)))

    return search_result


def support_vector(ds, name):
    arquivo = open('saidaRF.txt', 'w')
    print("\nlearning on dataset %s" % name)
    X_train, y_train = ds[0]
    X_test, y_test = ds[1]
    # scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    scoring = {'Accuracy': make_scorer(accuracy_score)}
    scorers = {
        # 'precision_macro': make_scorer(precision_score),
        # 'recall_macro': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)}
    param_search = [
        {'kernel': ['rbf'], 'C': [1], 'gamma': [1]}
        # {'kernel': ['linear'], 'C': [1]},
        # {'kernel': ['poly'],   'C': [1], 'coef0': [0,1,2,5], 'gamma':[1],'degree': [2]}
        # ]

        #  [
        # {'kernel': ['rbf'],    'C': [0.01, 0.1, 1, 5, 10], 'gamma':[0.001, 0.01, 0.1, 1, 5]}
        # {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 5, 10]}]
        # {'kernel': ['poly'],   'C': [0.01, 0.1, 1, 5, 10], 'gamma':[0.001, 0.01, 0.1, 1, 5],'degree': [1,2,3]}
        # {'kernel': ['poly'],   'C': [1], 'gamma':[1],'degree': [2,3]}
    ]
    svc = SVC(random_state=0)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)
    # refit an estimator using the best found parameters on the whole dataset.
    refit_score = 'accuracy_score'
    search = GridSearchCV(estimator=svc, param_grid=param_search, scoring=scorers, return_train_score=True,
                          refit=refit_score, cv=cv)  # refit=refit_score, return_train_score=True,
    search_result = search.fit(X_train, y_train)
    dump(search_result, open('BestClassifierModel_SVC.pkl', 'wb'))
    y_train_pred = search_result.predict(X_train)
    print('\nConfusion matrix of MLP optimized for {} on the TRAIN DATA:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_train, y_train_pred),
                       columns=['class1', 'class2'], index=['0', '1']))
    print("Best parameters set found on development set:")
    print()
    print(search_result.best_params_)
    print(search_result.best_score_)
    results_df = pd.DataFrame(search_result.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("kernel")
    # results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]
    results_df = pd.DataFrame(search_result.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")
    # results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]
    # create df of model scores ordered by performance
    model_scores = results_df.filter(regex=r"split\d*_test_score")
    # plot 30 examples of dependency between cv fold and AUC scores
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:30],
        dashes=False,
        palette="Set1",
        marker="o",
        alpha=0.5,
        ax=ax,
    )
    ax.set_xlabel("CV test fold", size=12, labelpad=10)
    ax.set_ylabel("Model AUC", size=12)
    ax.tick_params(bottom=True, labelbottom=False)
    plt.show()
    # print correlation of AUC scores across folds
    print(f"Correlation of models:\n {model_scores.transpose().corr()}")
    # avaliacao da correlacao entre os resultados
    model_1_scores = model_scores.iloc[0].values  # scores of the best model
    model_2_scores = model_scores.iloc[1].values  # scores of the second-best model    
    differences = model_1_scores - model_2_scores
    n = differences.shape[0]  # number of test sets
    df = n - 1
    n_train = len(list(cv.split(X_train, y_train))[0][0])
    n_test = len(list(cv.split(X_test, y_test))[0][1])
    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    print(f"Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}")
    t_stat_uncorrected = np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
    p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)
    print(
        f"Uncorrected t-value: {t_stat_uncorrected:.3f}\n"
        f"Uncorrected p-value: {p_val_uncorrected:.3f}"
    )
    n_comparisons = factorial(len(model_scores)) / (
            factorial(2) * factorial(len(model_scores) - 2)
    )
    pairwise_t_test = []
    for model_i, model_k in combinations(range(len(model_scores)), 2):
        model_i_scores = model_scores.iloc[model_i].values
        model_k_scores = model_scores.iloc[model_k].values
        differences = model_i_scores - model_k_scores
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_t_test.append(
            [model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val]
        )
    pairwise_comp_df = pd.DataFrame(
        pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
    ).round(3)
    pairwise_comp_df
    # make the predictions test set
    y_pred = search_result.predict(X_test)

    # confusion matrix on the test data.
    print('\nConfusion matrix of MLP optimized for {} on the Test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       columns=['class1', 'class2'], index=['0', '1']))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_test, y_pred, digits=2))


def melhor_modelo(ds, name):
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
        warnings.filterwarnings("ignore", category=ConvergenceWarning,
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
    print("training: %s" % params)
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

    print("testing: %s" % p)
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
        disp = plot_confusion_matrix(mlps_best, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


def modeloMLP(ds, name):
    GRID = [
        {
            'scaler': [StandardScaler()],
            'estimator': [MLPClassifier(random_state=RANDOM_SEED)],
            'estimator__solver': ['adam'],
            'estimator__learning_rate_init': [0.0001, 0.001, 0.01],
            'estimator__max_iter': [500],
            'estimator__hidden_layer_sizes': [(5), (10), (20), (30), (40), (50), (60), (70), (80), (90), (100)],
            'estimator__activation': ['logistic', 'tanh', 'relu'],
            'estimator__alpha': [0.0001, 0.001, 0.005],
            'estimator__early_stopping': [True]
        }
    ]

    PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])

    grid_search = GridSearchCV(estimator=PIPELINE, param_grid=GRID,
                               scoring=make_scorer(accuracy_score),  # average='macro'),
                               n_jobs=-1, cv=split, refit=True, verbose=1,
                               return_train_score=False)

    grid_search.fit(X, y)

    param_search = {
        'hidden_layer_sizes': [(10), (15), (20), (25), (30), (40), (50)],  #
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'alpha': [0.001, 0.005],
        'learning_rate': ['constant', 'adaptive'],
        'nesterovs_momentum': [True],
        'beta_1': [0.95], 'beta_2': [0.999],
        'learning_rate_init': [0.0001, 0.001],  # , 0.02, 0.05, 0.1],
        'momentum': [0.9, 0.85, 0.95]
    }

    # Criando e escrevendo em arquivos de texto (modo 'w').
    arquivo = open('saidaMLP.txt', 'w')
    print("\nlearning on dataset %s" % name)
    X_train, y_train = ds[0]
    X_test, y_test = ds[1]
    # scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    scoring = 'accuracy'
    refit_score = 'accuracy_score'
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

    my_callbacks = [callbacks.EarlyStopping(monitor='val_accuracy',
                                            patience=100, mode='max',
                                            restore_best_weights=True)]
    # refit an estimator using the best found parameters on the whole dataset.
    mlp = MLPClassifier(
        batch_size='auto',
        max_iter=500,
        shuffle=True,
        tol=0.0001,
        nesterovs_momentum=True,
        validation_fraction=0.2,
        early_stopping=True,
        n_iter_no_change=15)

    # mlp.out_activation_ = 'softmax'
    # efit an estimator using the best found parameters on the whole dataset.
    search = GridSearchCV(estimator=mlp,
                          param_grid=param_search,
                          scoring=scoring,
                          return_train_score=True,
                          refit=True,
                          cv=cv)

    search_result = search.fit(X_train, y_train)
    dump(search_result, open('BestClassifierModel_MLP2classes.pkl', 'wb'))
    print("Best parameters set found on development set:")
    print()
    print(search_result.best_params_)
    print(search_result.best_score_)
    arquivo.write("Best parameters set found on development set:")
    arquivo.write("____________________________________________")
    print()
    arquivo.write("Best Parameters")
    arquivo.write(str(search_result.best_params_))
    arquivo.write("Best Scores")
    arquivo.write(str(search_result.best_score_))

    y_train_pred = search_result.predict(X_train)
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_train, y_train_pred, digits=3))
    print('\nConfusion matrix of MLP optimized for {} on the TRAIN DATA:'.format(refit_score))
    arquivo.write('\nConfusion matrix of MLP optimized for {} on the TRAIN DATA:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_train, y_train_pred),
                       columns=['class1', 'class2'], index=['0', '1']))
    arquivo.write(str(pd.DataFrame(confusion_matrix(y_train, y_train_pred),
                                   columns=['class1', 'class2'], index=['0', '1'])))

    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_train, y_train_pred, digits=4))
    arquivo.write(str(pd.DataFrame(confusion_matrix(y_train, y_train_pred),
                                   columns=['class1', 'class2'], index=['0', '1'])))
    results_df = pd.DataFrame(search_result.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("hidden_layer_sizes")
    # results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]
    report(search_result.cv_results_)
    print('results mean_test_score', results_df['mean_test_score'].round(3).head())
    arquivo.write('results mean_test_score')
    arquivo.write(str(results_df['mean_test_score'].round(3).head()))
    arquivo.write('----------------------------------------------------------')
    print('results std_test_score', results_df['std_test_score'].round(3).head())
    arquivo.write('results std_test_score')
    arquivo.write(str(results_df['std_test_score'].round(3).head()))
    arquivo.write('----------------------------------------------------------')
    arquivo.write("Reports of CV Results")
    arquivo.write(str(report(search_result.cv_results_)))
    arquivo.write('----------------------------------------------------------')
    # create df of model scores ordered by performance
    model_scores = results_df.filter(regex=r"split\d*_test_score")
    # plot 30 examples of dependency between cv fold and AUC scores
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:30],
        dashes=False,
        palette="Set1",
        marker="o",
        alpha=0.5,
        ax=ax,
    )
    ax.set_xlabel("CV test fold", size=12, labelpad=6)
    ax.set_ylabel("Model AUC", size=12)
    ax.tick_params(direction='out', bottom=True, labelbottom=False)
    plt.show()
    # print correlation of AUC scores across folds
    print(f"Correlation of models:\n {model_scores.transpose().corr()}")
    arquivo.write(f"Correlation of models:\n {model_scores.transpose().corr()}")
    # ------------------------------------------------------------
    # AVALIA SE OS MODELOS SÃO SEMELHANTES
    # SE p<0.05 eles são difentes   
    # avaliacao da correlacao entre os resultados
    model_1_scores_RF = model_scores.iloc[0].values  # scores of the best model
    model_2_scores_RF = model_scores.iloc[1].values  # scores of the second-best model    
    differences = model_1_scores_RF - model_2_scores_RF
    n = differences.shape[0]  # number of test sets
    df = n - 1
    n_train = len(list(cv.split(X_train, y_train))[0][0])
    n_test = len(list(cv.split(X_test, y_test))[0][1])
    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    print(f"Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}")
    arquivo.write(f"Corrected t-value: {t_stat:.3f}\nCorrected p-value: {p_val:.3f}")
    # ------------------------------------------------------------
    # AVALIA SE OS MODELOS SÃO SEMELHANTES
    # SE p<0.05 eles são difentes
    t_stat_uncorrected = np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
    p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)
    print(
        f"Uncorrected t-value: {t_stat_uncorrected:.3f}\n"
        f"Uncorrected p-value: {p_val_uncorrected:.3f}"
    )
    arquivo.write(f"Uncorrected t-value: {t_stat_uncorrected:.3f}\n"
                  f"Uncorrected p-value: {p_val_uncorrected:.3f}")

    # ------------------------------------------------------------
    n_comparisons = factorial(len(model_scores)) / (
            factorial(2) * factorial(len(model_scores) - 2)
    )
    pairwise_t_test = []
    for model_i, model_k in combinations(range(len(model_scores)), 2):
        model_i_scores = model_scores.iloc[model_i].values
        model_k_scores = model_scores.iloc[model_k].values
        differences = model_i_scores - model_k_scores
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_t_test.append(
            [model_scores.index[model_i], model_scores.index[model_k], t_stat, p_val]
        )
    pairwise_comp_df = pd.DataFrame(
        pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
    ).round(3)
    print(str(pairwise_comp_df))
    arquivo.write("Pairwise Comparting")
    arquivo.write(str(pairwise_comp_df))
    # ------------------------------------------------------------
    # make the predictions test set
    y_pred = search_result.predict(X_test)
    arquivo.write("-------------------------------------------------------")
    arquivo.write("Test Results")
    print('Test recall_micro = ', recall_score(y_test, y_pred, average='micro'))
    print('Test recall_macro = ', recall_score(y_test, y_pred, average='macro'))
    arquivo.write('Test recall_micro = ')
    arquivo.write(str(recall_score(y_test, y_pred, average='micro')))
    arquivo.write('Test recall_macro = ')
    arquivo.write(str(recall_score(y_test, y_pred, average='macro')))
    print('Test precision_micro = ', precision_score(y_test, y_pred, average='micro'))
    print('Test precision_macro = ', precision_score(y_test, y_pred, average='micro'))
    arquivo.write('Test precision_micro = ')
    arquivo.write(str(precision_score(y_test, y_pred, average='micro')))
    arquivo.write('Test precision_macro = ')
    arquivo.write(str(precision_score(y_test, y_pred, average='micro')))
    print('Test Best params for refit {}'.format(refit_score))
    print(search_result.best_params_)
    # confusion matrix on the test data.
    print('\nConfusion matrix of MLP optimized for {} on the Test data:'.format(refit_score))
    arquivo.write('\nConfusion matrix of MLP optimized for {} on the Test data:'.format(refit_score))
    print(str(pd.DataFrame(confusion_matrix(y_test, y_pred),
                           columns=['class1', 'class2'], index=['0', '1'])))
    arquivo.write(str(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                   columns=['class1', 'class2'], index=['0', '1'])))

    print(metrics.confusion_matrix(y_test, y_pred))
    arquivo.write("Confusion Matrix")
    arquivo.write(str(metrics.confusion_matrix(y_test, y_pred)))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_test, y_pred, digits=2))
    arquivo.write('metrics')
    arquivo.write(str(metrics.classification_report(y_test, y_pred, digits=2)))
    return search_result


# pd.set_option('display.max_columns', None)

# criando o DataFrame com import do CSV


# criando o DataFrame com import do CSV
basecsv = pd.read_csv('base_carlos_csv.csv', sep=',')

# #base completa
# dataset = basecsv[[
# 'detratores',
# 'pct_uso_2g_vivo',
# 'pct_uso_3g_vivo',
# 'pct_uso_4g_vivo',
# 'pct_uso_3g_nextel',
# 'pct_uso_4g_nextel',
# 'pct_uso_2g',
# 'trafego_vivo_over_quota',
# 'trafego_total',
# 'avg_changes_vivo_nextel',
# 'avg_changes_3g_4g',
# 'bytes_exceeded_m0',
# 'bytes_exceeded_m1',
# 'days_over_quota_m0',
# 'days_over_quota_m1',
# 'mean_worst_rtt_nextel_3g',
# 'mean_worst_rtt_nextel_4g',
# 'mean_worst_rtt_vivo_2g',
# 'mean_worst_rtt_vivo_3g',
# 'mean_worst_rtt_vivo_4g',
# '11',
# '21',
# 'other',
# 'tmcode',
# 'safra_geracao',
# 'vl_fatura_avg_3m',
# 'vl_fatura_max_3m',
# 'vl_fatura_r1',
# 'vl_fatura_r2',
# 'vl_fatura_r3',
# 'fl_fatura_zerada',
# 'qt_gbytes_quota_total_atual',
# 'vl_plano_atual',
# 'vl_plano_oferta',
# 'qt_gb_total_oferta',
# 'pc_uso_dados_avg_3m',
# 'pc_uso_dados_m1',
# 'qt_gbvivo_m1',
# 'qt_dias_restante_fidelizacao',
# 'fl_fidelizado',
# 'fl_credit_score_c',
# 'fl_credit_score_null',
# 'fl_credit_score_d',
# 'fl_credit_score_a',
# 'fl_credit_score_e',
# 'fl_credit_score_b',
# 'fl_credit_score_j',
# 'fl_credit_score_i',
# 'fl_credit_score_f',
# 'fl_credit_score_g',
# 'fl_credit_score_h',
# 'fl_credit_score_u',
# 'fl_servico_ativo_e_contrato_familia',
# 'vl_preco_gb_truncado',
# 'percentil_por_plano',
# 'percentil_geral',
# 'port_in_flag',
# 'm1_qt_involuntary_suspension',
# 'm1_qt_voluntary_suspension',
# 'm1_rate_plan_amt',
# 'm1_service_amt',
# 'm1_air_amt',
# 'm1_occ_amt',
# 'nxt_3g_traffic_volume_m1',
# 'nxt_4g_traffic_volume_m1',
# 'vivo_traffic_volume_m1',
# 'm1_call_term_err',
# 'm1_nextel_long_dist_99_sec',
# 'm1_nextel_long_dist_99_qty',
# 'm1_vivo_long_dist_99_sec',
# 'm1_vivo_long_dist_99_qty',
# 'm1_nextel_long_dist_non_99_sec',
# 'm1_nextel_long_dist_non_99_qty',
# 'm1_vivo_long_dist_non_99_sec',
# 'm1_vivo_long_dist_non_99_qty',
# 'm1_nextel_landline_moc_secs',
# 'm1_nextel_landline_moc_qty',
# 'm1_nextel_landline_mtc_secs',
# 'm1_nextel_landline_mtc_qty',
# 'm1_vivo_landline_moc_secs',
# 'm1_vivo_landline_moc_qty',
# 'm1_vivo_landline_mtc_secs',
# 'm1_vivo_landline_mtc_qty',
# 'm1_nextel_mobile_moc_secs',
# 'm1_nextel_mobile_moc_qty',
# 'm1_nextel_mobile_mtc_secs',
# 'm1_nextel_mobile_mtc_qty',
# 'm1_vivo_mobile_moc_secs',
# 'm1_vivo_mobile_moc_qty',
# 'm1_vivo_mobile_mtc_secs',
# 'm1_vivo_mobile_mtc_qty',
# 'm1_nextel_nextel_moc_secs',
# 'm1_nextel_nextel_moc_qty',
# 'm1_nextel_nextel_mtc_secs',
# 'm1_nextel_nextel_mtc_qty',
# 'm1_vivo_nextel_moc_secs',
# 'm1_vivo_nextel_moc_qty',
# 'm1_vivo_nextel_mtc_secs',
# 'm1_vivo_nextel_mtc_qty',
# 'fl_rclpcn_max_3m_at',
# 'qt_rclpcn_sum_3m_at',
# 'qt_slcatv_sum_3m_at',
# 'fl_errctrpctexr_max_3m_at',
# 'fl_ctrpctexr_max_3m_at',
# 'qt_ctrpctexr_sum_3m_at',
# 'qt_ctrpctexr_sum_1m_at',
# 'vl_ctrpctexr_sum_3m_at',
# 'fl_ngdctrpctexr_max_3m_at',
# 'fl_ctrpctrcr_max_3m_at',
# 'qt_slcviabol_sum_3m_at',
# 'fl_slcviabol_max_3m_at',
# 'fl_slcviabol_max_1m_at',
# 'qt_cnt_sum_3m_at',
# 'fl_cnt_sum_3m_at',
# 'fl_cnt_max_1m_at',
# 'qt_acsapp_sum_3m_at',
# 'qt_acsapp_sum_1m_at',
# 'fl_ftrele_bn',
# 'qt_diavdactr_ct',
# 'qt_spspag_sum_3m_ct',
# 'qt_sps_sum_3m_ct',
# 'qt_diasps_max_3m_ct',
# 'qt_diaspspag_max_3m_ct',
# 'pc_vdachnivl_med_3_6m_vd',
# 'qt_vda_sum_3_6m_vd',
# 'fl_debaut_max_1m_tt',
# 'fl_debaut_max_2r_tt',
# 'qt_diaatr_max_3m_tt',
# 'fl_neg_max_3m_tt',
# 'fl_nvrpad_max_3m_tt',
# 'vl_pag_sum_3m_tt',
# 'vl_atr_sum_3m_tt',
# 'vl_pagdpsvnc_sum_3m_tt',
# 'qt_diaatratl_max_3m_tt',
# 'pc_pagdpsvcn_pag_3m_tt',
# 'pc_pag_3m_tt',
# 'fl_degree',
# 'cd_sale_channel_67',
# 'cd_sale_channel_63',
# 'cd_sale_channel_66',
# 'cd_sale_channel_missing',
# 'cd_sale_channel_62',
# 'churn_score',
# 'pct_dom_particulares_permanentes',
# 'pct_dom_particulares_improvisados',
# 'pct_dom_rend_percap_mensal_0',
# 'pct_dom_rend_percap_mensal_ate_1_8_sm',
# 'pct_dom_rend_percap_mensal_1_8_a_1_4_sm',
# 'pct_dom_rend_percap_mensal_1_4_a_1_2_sm',
# 'pct_dom_rend_percap_mensal_1_2_a_1_sm',
# 'pct_dom_rend_percap_mensal_1_a_2_sm',
# 'pct_dom_rend_percap_mensal_2_a_3_sm',
# 'pct_dom_rend_percap_mensal_3_a_5_sm',
# 'pct_dom_rend_percap_mensal_5_a_10_sm',
# 'pct_dom_rend_percap_mensal_10_plus_sm',
# 'rendimento_medio_dom',
# 'inar_rate_plan_Y',
# 'brand_name_m1_ASUS',
# 'brand_name_m1_Apple',
# 'brand_name_m1_LG',
# 'brand_name_m1_Motorola',
# 'brand_name_m1_Samsung',
# 'brand_name_m1_other',
# 'operating_system_m1_Android',
# 'operating_system_m1_iOS',
# 'operating_system_m1_other',
# 'bluetooth_m1_N',
# 'bluetooth_m1_Not Known',
# 'bluetooth_m1_Y',
# 'wlan_m1_N',
# 'wlan_m1_Not Known',
# 'wlan_m1_Y',
# 'device_type_m1_Connected Computer',
# 'device_type_m1_Dongle',
# 'device_type_m1_Handheld',
# 'device_type_m1_Mobile Phone/Feature phone',
# 'device_type_m1_Module',
# 'device_type_m1_Portable(include PDA)',
# 'device_type_m1_Smartphone',
# 'device_type_m1_Tablet',
# 'device_type_m1_Vehicle',
# 'device_type_m1_WLAN Router',
# 'total_pct_airtime',
# 'std_tp',
# 'worst_tp_mean',
# 'pct_0_400',
# 'pct_400_700',
# 'pct_700_1000',
# 'pct_1000_2000',
# 'pct_2000_plus'
# ]]

# #BASE CARLOS ALBERTO #38
dataset = basecsv[[
    'detratores',
    'qt_dias_restante_fidelizacao',
    'avg_changes_3g_4g',
    'avg_changes_vivo_nextel',
    'mean_worst_rtt_nextel_3g',
    'mean_worst_rtt_nextel_4g',
    'mean_worst_rtt_vivo_2g',
    'mean_worst_rtt_vivo_3g',
    'mean_worst_rtt_vivo_4g',
    'total_pct_airtime',
    'churn_score',
    'trafego_total',
    'trafego_vivo_over_quota',
    'qt_gbvivo_m1',
    'pct_uso_4g_nextel',
    'pct_uso_2g_vivo',
    'pct_uso_3g_vivo',
    'pct_uso_4g_vivo',
    'pct_uso_3g_nextel',
    'pct_uso_2g',
    'qt_diavdactr_ct',
    'm1_call_term_err',
    'nxt_3g_traffic_volume_m1',
    'nxt_4g_traffic_volume_m1',
    'vivo_traffic_volume_m1',
    'pc_uso_dados_avg_3m',
    'pc_uso_dados_m1',
    'qt_acsapp_sum_3m_at',
    'qt_acsapp_sum_1m_at',
    'pct_dom_rend_percap_mensal_5_a_10_sm',
    'pct_0_400',
    'pct_400_700',
    'pct_700_1000',
    'pct_1000_2000',
    'pct_2000_plus',
    'vl_pag_sum_3m_tt',
    'vl_atr_sum_3m_tt',
    'worst_tp_mean'
]]

# base CFS

# dataset = basecsv[[
# 'detratores',
# 'pct_uso_3g_nextel',
# 'pct_uso_4g_nextel',
# 'trafego_vivo_over_quota',
# 'trafego_total',
# 'avg_changes_3g_4g',
# 'bytes_exceeded_m1',
# 'mean_worst_rtt_nextel_3g',
# 'fl_fatura_zerada',
# 'qt_gbvivo_m1',
# 'qt_dias_restante_fidelizacao',
# 'fl_credit_score_null',
# 'nxt_3g_traffic_volume_m1',
# 'vivo_traffic_volume_m1',
# 'qt_acsapp_sum_3m_at',
# 'qt_diasps_max_3m_ct',
# 'fl_neg_max_3m_tt',
# 'vl_atr_sum_3m_tt',
# 'churn_score',
# 'brand_name_m1_LG',
# 'operating_system_m1_iOS',
# 'total_pct_airtime',
# 'pct_0_400'

# ]]


# BASE RELIEF F
# dataset = basecsv[[
# 'detratores',
# 'port_in_flag',
# 'pct_2000_plus',
# 'cd_sale_channel_missing',
# 'fl_credit_score_i',
# 'fl_credit_score_d',
# 'fl_servico_ativo_e_contrato_familia',
# 'fl_credit_score_b',
# 'fl_fidelizado',
# 'fl_credit_score_null',
# 'churn_score',
# #'var_21',
# 'fl_credit_score_c',
# 'fl_credit_score_u',
# #'var_11',
# 'm1_call_term_err',
# 'days_over_quota_m1',
# 'qt_dias_restante_fidelizacao',
# 'qt_diavdactr_ct',
# 'mean_worst_rtt_nextel_3g',
# 'fl_cnt_sum_3m_at',
# 'fl_credit_score_e',
# 'pct_0_400',
# 'mean_worst_rtt_nextel_4g',
# 'fl_credit_score_a',
# 'fl_credit_score_h',
# 'fl_credit_score_g',
# 'pct_dom_rend_percap_mensal_3_a_5_sm',
# 'vl_preco_gb_truncado',
# 'pct_dom_rend_percap_mensal_1_2_a_1_sm',
# 'pct_dom_rend_percap_mensal_1_4_a_1_2_sm',
# 'days_over_quota_m0',
# 'fl_cnt_max_1m_at',
# 'm1_service_amt',
# 'other',
# 'fl_credit_score_f',
# 'fl_credit_score_j',
# 'pct_dom_rend_percap_mensal_5_a_10_sm',
# 'rendimento_medio_dom',
# 'pct_dom_rend_percap_mensal_2_a_3_sm',
# 'pct_1000_2000',
# 'avg_changes_3g_4g',
# 'avg_changes_vivo_nextel',
# 'vl_fatura_r2',
# 'fl_neg_max_3m_tt',
# 'pct_dom_rend_percap_mensal_1_a_2_sm',
# 'std_tp',
# 'brand_name_m1_LG',
# 'mean_worst_rtt_vivo_4g',
# 'trafego_total',
# 'pct_uso_4g_vivo',
# 'vl_fatura_r3',
# 'brand_name_m1_Samsung',
# 'pc_pag_3m_tt',
# 'operating_system_m1_other',
# 'vl_plano_atual',
# 'mean_worst_rtt_vivo_3g',
# 'vl_fatura_max_3m',
# 'operating_system_m1_Android',
# 'pct_dom_rend_percap_mensal_1_8_a_1_4_sm',
# 'brand_name_m1_Motorola',
# 'pct_dom_rend_percap_mensal_10_plus_sm',
# 'worst_tp_mean',
# 'vl_fatura_r1',
# 'pc_vdachnivl_med_3_6m_vd',
# 'wlan_m1_N',
# 'qt_vda_sum_3_6m_vd',
# 'qt_acsapp_sum_3m_at',
# 'pct_uso_3g_vivo',
# 'pct_700_1000',
# 'pct_uso_2g',
# 'pct_uso_2g_vivo',
# 'qt_diasps_max_3m_ct',
# 'fl_fatura_zerada',
# 'vl_pag_sum_3m_tt',
# 'fl_debaut_max_1m_tt',
# 'wlan_m1_Y',
# 'qt_cnt_sum_3m_at',
# 'nxt_4g_traffic_volume_m1',
# 'qt_diaatr_max_3m_tt',
# 'brand_name_m1_other',
# 'vl_pagdpsvnc_sum_3m_tt',
# 'fl_debaut_max_2r_tt',
# #'wlan_m1_Not_Known',
# #'bluetooth_m1_Not_Known',
# 'm1_nextel_nextel_mtc_qty',
# 'm1_vivo_nextel_mtc_qty',
# 'm1_vivo_nextel_moc_qty',
# 'm1_nextel_nextel_moc_qty',
# 'm1_nextel_nextel_moc_secs',
# 'vl_fatura_avg_3m',
# 'qt_acsapp_sum_1m_at',
# 'qt_spspag_sum_3m_ct',
# 'total_pct_airtime',
# 'm1_nextel_nextel_mtc_secs',
# 'qt_sps_sum_3m_ct',
# 'm1_nextel_landline_mtc_secs',
# 'm1_nextel_long_dist_99_sec',
# 'device_type_m1_Handheld',
# 'nxt_3g_traffic_volume_m1',
# 'm1_nextel_landline_moc_qty',
# 'm1_vivo_landline_moc_qty',
# 'device_type_m1_Smartphone',
# 'tmcode',
# 'pct_400_700',
# 'qt_diaatratl_max_3m_tt',
# 'mean_worst_rtt_vivo_2g',
# 'vl_ctrpctexr_sum_3m_at',
# 'fl_ctrpctexr_max_3m_at',
# 'qt_gbytes_quota_total_atual',
# 'm1_nextel_landline_moc_secs',
# 'bytes_exceeded_m1',
# 'm1_vivo_long_dist_99_qty',
# 'm1_nextel_long_dist_99_qty',
# 'brand_name_m1_ASUS',
# 'm1_nextel_mobile_mtc_secs',
# 'm1_nextel_landline_mtc_qty',
# 'm1_vivo_landline_mtc_qty',
# 'pct_uso_4g_nextel',
# 'pct_dom_rend_percap_mensal_0',
# 'qt_diaspspag_max_3m_ct',
# 'pct_dom_particulares_improvisados',
# 'percentil_geral',
# 'trafego_vivo_over_quota',
# #'device_type_m1_Mobile_Phone',
# 'bytes_exceeded_m0',
# 'm1_nextel_long_dist_non_99_qty',
# 'm1_vivo_long_dist_non_99_qty',
# 'm1_vivo_mobile_mtc_qty',
# 'm1_nextel_mobile_mtc_qty',
# 'm1_nextel_long_dist_non_99_sec',
# 'qt_ctrpctexr_sum_3m_at',
# 'm1_nextel_mobile_moc_secs',
# 'qt_rclpcn_sum_3m_at',
# 'fl_rclpcn_max_3m_at',
# #'device_type_m1_Portable_PDA',
# 'fl_nvrpad_max_3m_tt',
# 'm1_rate_plan_amt',
# 'pc_pagdpsvcn_pag_3m_tt',
# 'operating_system_m1_iOS'
# ]]


dataset.describe()
dataset.columns

# undersimpling
df = dataset.describe()
# df_class_1 = dataset.loc[dataset.classe == 0].sample(10468)
# df_class_2 = dataset.loc[dataset.classe == 1].sample(10468)
# df_class_3 = dataset.loc[dataset.classe == 2]
# dataset = df_c7lass_1.append(df_class_2).append(df_class_3)


# df_class_1 = dataset.loc[dataset.detratores == 0].sample(17000)
# df_class_2 = dataset.loc[dataset.detratores == 1].sample(17000)
# base = df_class_1.append(df_class_2)

# sem qualquer equalização
df_class_1 = dataset.loc[dataset.detratores == 0]
df_class_2 = dataset.loc[dataset.detratores == 1]
base = df_class_1.append(df_class_2)

base.detratores.value_counts()

# creating input features and target variables
X = base.iloc[:, 1:38]
y = base.iloc[:, 0]
print("y:", y)

# standardizing the input feature
sc = StandardScaler()
X = sc.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# base estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)

# Equalizacao com SMOTE
sm = SMOTE(random_state=42)
X_res_train, y_res_train = sm.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_res_train))
X_train = X_res_train
y_train = y_res_train

# show the distribution
print('y_train class distribution')
print(y_train.value_counts(normalize=True))
print('y_test class distribution')
print(y_test.value_counts(normalize=True))
data_sets = [(X_train, y_train), (X_test, y_test)]

# MODELO MLP
# name=['Classificação de Churn - MLP']
# grid_search_clf = modelo(data_sets, name=name)
# results = pd.DataFrame(grid_search_clf.cv_results_)
# results = results.sort_values(by='rank_test_score', ascending=False)
# print('results mean_test_score', results['mean_test_score'].round(3).head())
# print('results std_test_score', results['std_test_score'].round(3).head())
# report(grid_search_model.cv_results_)


# RANDOM FOREST
# name=['Classificação de Churn - Random Forest']
# grid_search_model = floresta(data_sets, name=name)
# results = pd.DataFrame(grid_search_model.cv_results_)
# results = results.sort_values(by='rank_test_score', ascending=False)
# print('results mean_test_score', results['mean_test_score'].round(3).head())
# print('results std_test_score', results['std_test_score'].round(3).head())
# report(grid_search_model.cv_results_)


# MLP
name = ['Classificação de Churn - MLP']
grid_search_model = modeloMLP(data_sets, name=name)
results = pd.DataFrame(grid_search_model.cv_results_)
results = results.sort_values(by='rank_test_score', ascending=False)
print('results mean_test_score', results['mean_test_score'].round(3).head())
print('results std_test_score', results['std_test_score'].round(3).head())
report(results.cv_results_)

# grid_search_clf = grid_search_wrapper(refit_score='accuracy_score')
# results = pd.DataFrame(grid_search_clf.cv_results_)
# results = results.sort_values(by='mean_test_precision_score', ascending=False)
# results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()

# melhor_modelo(data_sets, name=name)


# classifier = Sequential()
# #First Hidden Layer 
# classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=35))
# #Second  Hidden Layer
# #classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))Layer
# #Output 
# classifier.add(Dense(3, activation='softmax', kernel_initializer='random_normal'))
# #Compiling the neural network
# classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
# #Fitting the data to the training dataset
# classifier.fit(X_train,pd.get_dummies(y_train), batch_size=10, epochs=10)
# pd.get_dummies(y_train)
# print(classifier.fit)

# eval_model=classifier.evaluate(X_train, y_train)
# eval_model

# y_pred=classifier.predict(X_test)
# #y_pred_bin = np.where(y_pred>0.5,1,0)
# #testar outras possibilidades de 0.5

# print(y_pred)

# # Use trained model to predict output of test dataset
# val = classifier.predict(X_test)


# lb = preprocessing.LabelBinarizer()
# lb.fit(y_test)
# #lb.fit(y_test)

# y_test_lb = lb.transform(y_test)
# #y_test_lb = lb.transform(y_test)
# #val_lb = lb.transform(val)
# #val_lb = lb.transform(val)

# #roc_auc_score(y_test_lb, val, average='macro')


# print(y_test_lb)


# y_pred_bin = np.argmax(y_pred,axis=1)
# print(y_pred_bin)

# #print métricas
# print('Accuracy Score =', accuracy_score(y_test, y_pred_bin))
# print('Precision Score =',precision_score(y_test, y_pred_bin, average='macro'))
# print('Recall Score =',recall_score(y_test, y_pred_bin,average='macro'))
# print('F1 Score =',f1_score(y_test, y_pred_bin,average='macro'))
# print('ROC = ',roc_auc_score(y_test_lb, val, average='macro'))

# cm = confusion_matrix(y_test_lb, val)
# print(cm)

# print(val)
# print(y_test)
