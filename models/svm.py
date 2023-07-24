from sklearn import svm
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate

from models.constraints import SEED, K_CV, N_REPEATS
from results.evaluator import evaluate


# Treinando e avaliando o SVM
def create_svm(df_X_train, df_X_test, df_y_train, df_y_test, cv_type=False):
    if cv_type == 'grid_cv':
        # Com cross-validation e busca de parâmetros
        print("-> Treinando o SVM com GridSearch e cross-validation...")
        alg = "SVM com GridSearch e cross-validation"
        model = svm.SVC()

        params = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear', 'rbf'],
            'max_iter': [10000],
            'random_state': [SEED],
            'probability': True
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
    elif cv_type == 'cv':
        # Somente cross-validation sem o grid search
        print("-> Treinando o SVM com Cross-validation...")
        alg = "SVM com Cross-validation"

        model = svm.SVC()

        cv = RepeatedStratifiedKFold(n_splits=K_CV, n_repeats=N_REPEATS)
        results = cross_validate(
            model, df_X_train, df_y_train, cv=cv, scoring='accuracy', return_train_score=True, return_estimator=True)

        estimator = results['estimator'][results['train_score'].argmax(axis=0)]
        df_y_pred = estimator.predict(df_X_test)
        pred_proba = estimator.predict_proba(df_X_test)[::, 1]
        return evaluate(alg, df_y_test, df_y_pred, pred_proba)
    else:
        # Sem cross-validation
        print("-> Treinando o SVM...")
        alg = "SVM"
        svm_model = svm.SVC(kernel='rbf', C=1, gamma=1, random_state=SEED, max_iter=10000, probability=True)
        svm_model.fit(df_X_train, df_y_train)

    df_y_pred = svm_model.predict(df_X_test)
    pred_proba = svm_model.predict_proba(df_X_test)[::, 1]
    return evaluate('SVM', df_y_test, df_y_pred, pred_proba)
