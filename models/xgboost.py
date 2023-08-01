from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate

from models.constraints import SEED, K_CV, N_REPEATS
from results.evaluator import evaluate
from xgboost import XGBClassifier, plot_importance

import pandas as pd


# Treinando e avaliando o XGBoost
def create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test, cv_type='simple'):
    if cv_type == 'grid_cv':
        # Com cross-validation e busca de parâmetros
        print("-> Treinando o XGBoost com GridSearch e cross-validation...")
        alg = "XGBoost com GridSearch e cross-validation"
        model = XGBClassifier()

        # params = {
        #     'booster': ['gbtree'],
        #     'objective': ['multi:softmax'],
        #     'num_class': [2],
        #     'eval_metric': ['logloss'],
        #     'nthread': [4],
        #     'learning_rate': [0.01, 0.1],
        #     'n_estimators': [53, 55],
        #     'max_depth': [3, 4],
        #     'min_child_weight': [1, 2],
        #     'gamma': [0, 0.1],
        #     'subsample': [0.5, 0.7],
        #     'colsample_bytree': [0.7, 0.9],
        #     'reg_alpha': [0.00005, 0.0001],
        #     'seed': [SEED]
        # }

        # params = {
        #     'booster': ['gbtree'],
        #     'objective': ['multi:softmax'],
        #     'num_class': [2],
        #     'eval_metric': ['logloss'],
        #     'nthread': [4],
        #     'learning_rate': [0.15, 0.1],
        #     'n_estimators': [54, 55, 56],
        #     'max_depth': [4],
        #     'min_child_weight': [3, 2],
        #     'gamma': [0, 0.01],
        #     'subsample': [0.5, 0.6, 0.4],
        #     'colsample_bytree': [0.8, 0.9, 1],
        #     'reg_alpha': [0.00005, 0.00001],
        #     'seed': [SEED]
        # }

        # params = {
        #     'booster': ['gbtree'],
        #     'objective': ['multi:softmax'],
        #     'num_class': [2],
        #     'eval_metric': ['logloss'],
        #     'nthread': [4],
        #     'learning_rate': [0.15, 0.2],
        #     'n_estimators': [55, 56, 57],
        #     'max_depth': [4],
        #     'min_child_weight': [3, 4],
        #     'gamma': [0.1, 0.01, 0.001],
        #     'subsample': [0.6, 0.7],
        #     'colsample_bytree': [0.7, 0.8],
        #     'reg_alpha': [0.00005],
        #     'seed': [SEED]
        # }

        # params = {
        #     'booster': ['gbtree'],
        #     'objective': ['multi:softmax'],
        #     'num_class': [2],
        #     'eval_metric': ['logloss'],
        #     'nthread': [4],
        #     'learning_rate': [0.15, 0.2, 0.25],
        #     'n_estimators': [56, 57, 58],
        #     'max_depth': [4],
        #     'min_child_weight': [3],
        #     'gamma': [0.1, 0.01],
        #     'subsample': [0.6],
        #     'colsample_bytree': [0.8, 0.9],
        #     'reg_alpha': [0.00005],
        #     'seed': [SEED]
        # }

        # params = {
        #     'booster': ['gbtree'],
        #     'objective': ['multi:softmax'],
        #     'num_class': [2],
        #     'eval_metric': ['logloss'],
        #     'nthread': [4],
        #     'learning_rate': [0.25, 0.3],
        #     'n_estimators': [55, 56, 57],
        #     'max_depth': [4],
        #     'min_child_weight': [3],
        #     'gamma': [0.01, 0.001],
        #     'subsample': [0.6],
        #     'colsample_bytree': [0.9, 1],
        #     'reg_alpha': [0.00005],
        #     'seed': [SEED]
        # }

        params = {
            'booster': ['gbtree'],
            'objective': ['multi:softmax'],
            'num_class': [2],
            'eval_metric': ['logloss'],
            'nthread': [4],
            'learning_rate': [0.3, 0.35],
            'n_estimators': [56, 57, 58],
            'max_depth': [4],
            'min_child_weight': [3],
            'gamma': [0.001, 0.0001],
            'subsample': [0.6],
            'colsample_bytree': [1],
            'reg_alpha': [0.00005],
            'seed': [SEED]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring='accuracy',
            cv=5,
            verbose=3
        )

        grid.fit(df_X_train, df_y_train)
        xgboost = grid.best_estimator_

        print(grid.best_estimator_)
        print(grid.best_params_)
    elif cv_type == 'cv':
        # Somente cross-validation sem o grid search
        print("-> Treinando o XGBoost com Cross-validation...")
        alg = "XGBoost com Cross-validation"

        xgboost = XGBClassifier(
            booster='gbtree',
            objective='multi:softmax',
            num_class=2,
            eval_metric='logloss',
            learning_rate=0.1,
            n_estimators=55,
            max_depth=4,
            min_child_weight=1,
            gamma=0,
            subsample=0.5,
            colsample_bytree=0.9,
            reg_alpha=5e-05,
            seed=SEED
        )

        cv = RepeatedStratifiedKFold(n_splits=K_CV, n_repeats=N_REPEATS)
        results = cross_validate(
            xgboost, df_X_train, df_y_train, cv=cv, scoring='accuracy', return_train_score=True, return_estimator=True)

        estimator = results['estimator'][results['train_score'].argmax(axis=0)]

        plot_importance(
            estimator,
            title='Importância das Variáveis para o XGBoost',
            xlabel='F-score',
            ylabel='Variáveis',
            grid=False
        )

        df_y_pred = estimator.predict(df_X_test)
        pred_proba = estimator.predict_proba(df_X_test)[::, 1]
        return evaluate(alg, df_y_test, df_y_pred, pred_proba)
    else:
        # Sem cross-validation
        print("-> Treinando o XGBoost...")
        alg = "XGBoost"

        xgboost = XGBClassifier(
            booster='gbtree',
            objective='multi:softmax',
            num_class=2,
            eval_metric='logloss',
            learning_rate=0.1,
            n_estimators=55,
            max_depth=4,
            min_child_weight=1,
            gamma=0,
            subsample=0.5,
            colsample_bytree=0.9,
            reg_alpha=5e-05,
            seed=SEED
        )

        xgboost.fit(df_X_train, df_y_train)

    df_y_pred = xgboost.predict(df_X_test)
    pred_proba = xgboost.predict_proba(df_X_test)[::, 1]
    return evaluate(alg, df_y_test, df_y_pred, pred_proba)
