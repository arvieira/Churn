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

        params = {
            'n_estimators': [45, 50, 55],
            'subsample': [0.25, 0.5, 1],
            'max_depth': [3, 4, 5],
            'eta': [0.05, 0.1, 0.3],
            'booster': ['gbtree'],
            'eval_metric': ['logloss'],
            'nthread': [4],
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
    else:
        # Sem cross-validation
        print("-> Treinando o XGBoost...")

        xgboost = XGBClassifier(
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
            eval_metric='logloss'
        )

        xgboost.fit(df_X_train, df_y_train)

    df_y_pred = xgboost.predict(df_X_test)
    evaluate('XGBoost', df_y_test, df_y_pred)
