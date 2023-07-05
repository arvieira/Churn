import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.model_selection import train_test_split

from models.constraints import SEED
from models.svm_adaboost import create_svm_adaboost
from models.xgboost import create_xgboost
from models.svm import create_svm
from preprocessing.util import preprocessing


# Lê a base de dados, cria a coluna de churn e no-churn como detratores
# Criando a coluna detratores
# Essa coluna será:
# 1 - churn
# 0 - no-churn
def read_database(filename):
    print("-> Lendo a base de dados e mapeando classes...")

    # Lendo a base
    df = pd.read_csv(filename, sep=',')

    # O atributo classe pode receber três valores: 0, 1 e 2
    # somente os 0s são detratores e geram churn. Os números 1 e 2 são no-churn
    df['classe'] = df['classe'].map(lambda x: 1 if x == 0 else 0)

    return df


# Balanceamento da base e separação em treino e teste
def balance_train_test(df, alg='random'):
    # Separando as variáveis de entrada e saída em X e y
    X = df.drop(columns=['classe'])
    y = df['classe']
    # print(X)
    # print(y)

    # Verificando o balanceamento da base
    # Balanceando a base de dados com undersampling aleatório como passo inicial
    # Verificando o balanceamento da base após o undersampling
    # print(df['classe'].value_counts())
    X, y = balance(X, y, alg)
    # print(y.value_counts())

    # Separando a base em treino e teste, mantendo a base de test balanceada pelo y
    print("-> Separando a base em treino e teste com proporção de 70/30...")
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    # print(f"X_train: {df_X_train.shape}")
    # print(f"X_test: {df_X_test.shape}")
    # print(f"y_train: {df_y_train.shape}")
    # print(f"y_test: {df_y_test.shape}")
    # print(f"y_train counts: {df_y_train.value_counts()}")
    # print(f"y_test counts: {df_y_test.value_counts()}")

    return df_X_train, df_X_test, df_y_train, df_y_test


# Balanceia a base com o algoritmo escolhido
def balance(df_X, df_y, alg):
    if alg == 'random':
        print(f"-> Baleanceando a base. Método escolhido: Random...")
        return RandomUnderSampler(sampling_strategy='majority').fit_resample(df_X, df_y)
    elif alg == 'tomek':
        print(f"-> Baleanceando a base. Método escolhido: TomekLink + Random...")
        tomek_sampler = TomekLinks(sampling_strategy='majority')
        tomek_X, tomek_y = tomek_sampler.fit_resample(df_X, df_y)
        return RandomUnderSampler(sampling_strategy='majority').fit_resample(tomek_X, tomek_y)
    else:
        print('Algoritmo de sampling não conhecido')
        return df_X, df_y


# Executa o treino e a avaliação dos modelos escolhidos
def train_models(df_X_train, df_X_test, df_y_train, df_y_test, models=None):
    # Colocando os valores padrões
    if models is None:
        models = ['xgboost', 'svm', 'svm_ada']

    # Treinando e avaliando o modelo XGBoost no modo raw
    if 'xgboost' in models:
        create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test)
    if 'xgboost_cv' in models:
        create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=True)

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


if __name__ == '__main__':
    # Lendo a base de dados e agregando os valores da classe em churn e no-churn
    data = read_database('base.csv')
    # print(data)

    #################################################################
    #             Padrão sem preprocessamento para comparar         #
    #################################################################
    # Balanceando a base e separando treino e teste
    # X_train, X_test, y_train, y_test = balance_train_test(data)

    # Treinando os modelos raw
    # train_models(X_train, X_test, y_train, y_test)
    #################################################################
    #             Padrão sem preprocessamento para comparar         #
    #################################################################

    # Realizando os preprocessamentos
    data = preprocessing(data, 60)

    # Balanceando a base e separando treino e teste
    X_train, X_test, y_train, y_test = balance_train_test(data, 'tomek')

    # Treinando os modelos com preprocessamento
    train_models(X_train, X_test, y_train, y_test, models=['svm_ada'])
