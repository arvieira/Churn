from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import t

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

from pickle import dump

from utils.util import compute_corrected_ttest


def random_florest_model(ds, name):
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
