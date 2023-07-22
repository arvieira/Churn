from models.mlp import create_mlp
from models.svm import create_svm
from models.svm_adaboost import create_svm_adaboost
from models.xgboost import create_xgboost


# Executa o treino e a avaliação dos modelos escolhidos
def train_models(df_X_train, df_X_test, df_y_train, df_y_test, models=None):
    # Colocando os valores padrões
    if models is None:
        models = ['xgboost', 'mlp', 'svm', 'svm_ada']

    # Treinando e avaliando o modelo XGBoost no modo raw
    if 'xgboost' in models:
        return create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test)
    if 'xgboost_cv' in models:
        return create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test, cv_type='cv')
    if 'xgboost_grid' in models:
        return create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test, cv_type='grid_cv')

    # Treinando e avaliando o modelo SVM no modo raw
    if 'svm' in models:
        create_svm(df_X_train, df_X_test, df_y_train, df_y_test)
    if 'svm_cv' in models:
        create_svm(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=True)

    # Treinando e avaliando o modelo SVM no modo raw com AdaBoost
    if 'svm_ada' in models:
        create_svm_adaboost(df_X_train, df_X_test, df_y_train, df_y_test)
    if 'svm_ada_cv' in models:
        create_svm_adaboost(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=True)

    # Treinando e avaliando o modelo MPL no modo raw
    if 'mlp' in models:
        create_mlp(df_X_train, df_X_test, df_y_train, df_y_test)
    if 'mlp_cv' in models:
        create_mlp(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=True)
