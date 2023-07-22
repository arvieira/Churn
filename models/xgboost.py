from sklearn.model_selection import GridSearchCV

from models.constraints import SEED
from results.evaluator import evaluate
from xgboost import XGBClassifier


# Treinando e avaliando o XGBoost
def create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=False):
    if grid_search:
        # Com cross-validation e busca de parâmetros
        print("-> Treinando o XGBoost com GridSearch e cross-validation...")
        model = XGBClassifier()

        # params = {
        #     'booster': ['gbtree'],
        #     'objective': ['multi:softmax'],
        #     'num_class': [2],
        #     'eval_metric': ['logloss'],
        #     'nthread': [4],
        #     'seed': [SEED],
        #     'learning_rate': [0.35],
        #     'n_estimators': [53],
        #     'max_depth': [3],
        #     'min_child_weight': [1],
        #     'gamma': [0],
        #     'subsample': [0.9],
        #     'colsample_bytree': [0.9],
        #     'reg_alpha': [0.00005],
        # }

        params = {
            'booster': ['gbtree'],
            'objective': ['multi:softmax'],
            'num_class': [2],
            'eval_metric': ['logloss'],
            'nthread': [4],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'n_estimators': [53, 55, 57],
            'max_depth': [1, 2, 3, 4],
            'min_child_weight': [1, 2, 3],
            'gamma': [0, 0.1, 0.01],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'reg_alpha': [0.00005, 0.0001, 0.0002],
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
    else:
        # Sem cross-validation
        print("-> Treinando o XGBoost...")

        xgboost = XGBClassifier(
            booster='gbtree',
            learning_rate=0.1,
            n_estimators=55,
            max_depth=4,
            min_child_weight=1,
            gamma=0,
            subsample=0.5,
            colsample_bytree=0.9,
            objective='multi:softmax',
            num_class=2,
            nthread=4,
            reg_alpha=5e-05,
            eval_metric='logloss',
        )

        xgboost.fit(df_X_train, df_y_train)

    df_y_pred = xgboost.predict(df_X_test)
    evaluate('XGBoost', df_y_test, df_y_pred)
