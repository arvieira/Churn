from sklearn import svm
from sklearn.model_selection import GridSearchCV

from models.constraints import SEED
from results.evaluator import evaluate


# Treinando e avaliando o SVM
def create_svm(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=False):
    if grid_search:
        # Com cross-validation e busca de parâmetros
        print("-> Treinando o SVM com GridSearch e cross-validation...")
        model = svm.SVC()

        params = {
            'C': [45, 55],
            'gamma': [1, 0.0001],
            'kernel': ['linear'],
            'max_iter': [5000, 10000],
            'random_state': [SEED]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring='accuracy',
            cv=5,
            verbose=3
        )

        grid.fit(df_X_train, df_y_train)
        svm_model = grid.best_estimator_

        print(grid.best_estimator_)
        print(grid.best_params_)
    else:
        # Sem cross-validation
        print("-> Treinando o SVM...")
        svm_model = svm.SVC(kernel='linear', C=55, gamma=1, random_state=SEED, max_iter=10000)
        svm_model.fit(df_X_train, df_y_train)

    df_y_pred = svm_model.predict(df_X_test)
    pred_proba = svm_model.predict_proba(df_X_test)[::, 1]
    evaluate('SVM', df_y_test, df_y_pred, pred_proba)
