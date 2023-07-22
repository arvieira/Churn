import os.path
import pandas as pd
from models.constraints import BASE, PREP_BASE
from models.modeler import train_models
from preprocessing.balance_split import base_balance, base_split
from preprocessing.preprocessor import preprocessing


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


if __name__ == '__main__':
    if not os.path.exists(PREP_BASE):
        print('-> Não foi encontrada uma base já processada.')
        # Lendo a base de dados e agregando os valores da classe em churn e no-churn
        data = read_database(BASE)
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
        data = preprocessing(data, zero_percentage=0.4, n_features=30, epsilon=400, normalizer='Z-SCORE')

        # Balanceando a base
        data = base_balance(data, 'tomek')

        # Salvando a base trabalhada para uso futuro
        data.to_csv(PREP_BASE, index=False)
    else:
        print('-> Base preprocessada encontrada. Carregando base...')
        data = pd.read_csv(PREP_BASE)

    # Separando treino e teste
    X_train, X_test, y_train, y_test = base_split(data)

    # Treinando os modelos com preprocessamento
    train_models(X_train, X_test, y_train, y_test, models=['xgboost_cv'])
