import os

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from models.constraints import PREP_BASE
from sklearn import metrics

from preprocessing.balance_split import base_split
from results.evaluator import evaluate


# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def modelfit(alg, df_x_train, df_y_train, df_x_test, df_y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(df_x_train.values, label=df_y_train.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(df_x_train, df_y_train)

    # Predict training set:
    dtrain_predictions = alg.predict(df_x_train)
    dtrain_predprob = alg.predict_proba(df_x_train)[:, 1]
    dtest_predictions = alg.predict(df_x_test)
    dtest_predprob = alg.predict_proba(df_x_test)[:, 1]

    print(f'CV best iteration: {alg.best_iteration}')

    df_y_pred = alg.predict(df_x_test)
    pred_proba = alg.predict_proba(df_x_test)[::, 1]
    evaluate('XGBoost', df_y_test, df_y_pred, pred_proba)

    # Print model report:
    print("\nModel Report")
    print("Accuracy Train: %.4g" % metrics.accuracy_score(df_y_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(df_y_train, dtrain_predprob))
    print("Accuracy Test: %.4g" % metrics.accuracy_score(df_y_test.values, dtest_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(df_y_test, dtest_predprob))

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    # plt.show()


def search(df_x_train, df_y_train):
    param_test1 = {
        # 'max_depth': range(3, 10, 2), # 3
        # 'min_child_weight': range(1, 6, 2) # 1

        # 'gamma': [i/10.0 for i in range(0, 5)] # 0.0

        # 'subsample': [i / 10.0 for i in range(6, 10)], # 0.9
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)] # 0.9

        # 'subsample': [0.85, 0.9, 0.95, 1], # 0.9
        # 'colsample_bytree': [0.85, 0.9, 0.95, 1] # 0.9

        # 'reg_alpha': [1e-4, 1e-3, 1e-2, 0.1, 1] # 0.0001
        # 'reg_alpha': [1e-4, 1e-5, 1e-6] # 0.00001
        # 'reg_alpha': [0.00005, 0.0001, 0.00015] # 0.00005

        # 'reg_lambda': [0.1, 1, 10]  # 1
        'reg_lambda': [0.95, 1, 1.05]  # 1

        # 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5] # 0.3
        # 'learning_rate': [0.25, 0.3, 0.35] # 0.15

        # 'objective': ['binary:logistic', 'multi:softmax', 'binary:hinge'], # softmax
        # 'num_class': [2]

        # 'booster': ['gbtree', 'dart'] # gbtree
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(booster='gbtree', learning_rate=0.35, n_estimators=53, max_depth=3,
                                                    min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.9,
                                                    objective='multi:softmax', num_class=2, nthread=4,
                                                    reg_alpha=5e-05, eval_metric='logloss',
                                                    seed=27),
                            param_grid=param_test1, scoring='accuracy', n_jobs=4, cv=5, verbose=3)
    gsearch1.fit(df_x_train, df_y_train)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)


data = pd.read_csv(PREP_BASE)
X_train, X_test, y_train, y_test = base_split(data)

# Choose all predictors except target & IDcols
preditores = X_train.columns
xgb1 = XGBClassifier(
    booster='gbtree',
    learning_rate=0.35,
    n_estimators=53,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='multi:softmax',
    num_class=2,
    nthread=4,
    reg_alpha=5e-05,
    eval_metric='logloss',
    seed=27)
modelfit(xgb1, X_train, y_train, X_test, y_test)
# search(X_train, y_train)

