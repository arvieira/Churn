from itertools import combinations
from math import factorial

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import t

from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics

from keras import callbacks
from pickle import dump

from utils.util import compute_corrected_ttest, report


RANDOM_SEED = 132453


def mlp_model(ds, name, X, y):
    grid = [
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

    pipeline = Pipeline([('scaler', None), ('estimator', MLPClassifier())])

    split = 5

    grid_search = GridSearchCV(estimator=pipeline, param_grid=grid,
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
