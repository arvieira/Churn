from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from results.evaluator import evaluate


# Treinando e avaliando o XGBoost
def create_mlp(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=False):
    if grid_search:
        # Com cross-validation e busca de parâmetros
        print("-> Treinando o MLP com GridSearch e cross-validation...")

        mlp = MLPClassifier(
            batch_size='auto',
            max_iter=500,
            shuffle=True,
            tol=0.0001,
            nesterovs_momentum=True,
            validation_fraction=0.2,
            early_stopping=True,
            n_iter_no_change=15)

        # params = {
        #     'hidden_layer_sizes': [10, 15, 20, 25, 30, 40, 50],
        #     'activation': ['relu', 'tanh', 'logistic'],
        #     'solver': ['adam', 'sgd', 'lbfgs'],
        #     'alpha': [0.001, 0.005],
        #     'learning_rate': ['constant', 'adaptive'],
        #     'nesterovs_momentum': [True],
        #     'beta_1': [0.95], 'beta_2': [0.999],
        #     'learning_rate_init': [0.0001, 0.001],  # , 0.02, 0.05, 0.1],
        #     'momentum': [0.9, 0.85, 0.95]
        # }

        params = {
            'hidden_layer_sizes': [5, 10, 15],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.001, 0.005],
            'learning_rate': ['constant', 'adaptive'],
            'nesterovs_momentum': [True],
            'beta_1': [0.95],
            'beta_2': [0.999],
            'learning_rate_init': [0.0001, 0.001],  # , 0.02, 0.05, 0.1],
            'momentum': [0.9, 0.85]
        }

        grid = GridSearchCV(
            estimator=mlp,
            param_grid=params,
            scoring='accuracy',
            cv=5,
            refit=True,
            verbose=3
        )

        grid.fit(df_X_train, df_y_train)
        mlp = grid.best_estimator_

        print(grid.best_estimator_)
        print(grid.best_params_)
    else:
        # Sem cross-validation
        print("-> Treinando o MLP...")
        mlp = MLPClassifier(
            batch_size='auto',
            max_iter=500,
            shuffle=True,
            tol=0.0001,
            nesterovs_momentum=True,
            validation_fraction=0.2,
            early_stopping=True,
            n_iter_no_change=15)
        mlp.fit(df_X_train, df_y_train)

    df_y_pred = mlp.predict(df_X_test)
    evaluate('MLP', df_y_test, df_y_pred)
