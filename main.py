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
    # Código para buscar o número de variáveis ótimo
    # df = []
    # for i in range(132):
    #     temp = data
    #
    #     # Realizando os preprocessamentos
    #     data = preprocessing(data, zero_percentage=0.6, n_features=i+1, epsilon=400, normalizer='Z-SCORE')
    #
    #     # Separando treino e teste
    #     X_train, X_test, y_train, y_test = base_split(data)
    #
    #     # Treinando os modelos com preprocessamento
    #     accuracy, auc = train_models(X_train, X_test, y_train, y_test, models=['xgboost'])
    #     print(f'Número de Variávies: {i+1}')
    #     df.append([i+1, accuracy, auc])
    #
    #     data = temp
    #
    # df = pd.DataFrame(df, columns=['Número de Variáveis', 'Acurácia', 'AUC'])
    # df.to_csv('bases/SelecaoVariaveis.csv', index=False)

    if not os.path.exists(PREP_BASE):
        print('-> Não foi encontrada uma base já processada.')

        # Lendo a base de dados e agregando os valores da classe em churn e no-churn
        data = read_database(BASE)

        # Realizando os preprocessamentos
        data = preprocessing(data, zero_percentage=0.6, n_features=29, epsilon=400, normalizer='MIN_MAX_STD')

        # Salvando a base trabalhada para uso futuro
        data.to_csv(PREP_BASE, index=False)
    else:
        print('-> Base preprocessada encontrada. Carregando base...')
        data = pd.read_csv(PREP_BASE)

    # Separando treino e teste
    X_train, X_test, y_train, y_test = base_split(data)

    # Balanceando a base
    X_train, y_train = base_balance(X_train, y_train, 'smote')

    # Treinando os modelos com preprocessamento
    accuracy, auc = train_models(X_train, X_test, y_train, y_test, models=['svm_ada'])
