from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import t
from sklearn import metrics

from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.svm import SVC

from pickle import dump

from karla_models.utils import compute_corrected_ttest


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