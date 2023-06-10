import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.mlp_model import mlp_model, report
from preprocessing.bases_variables import CARLOS_ALBERTO
from preprocessing.sampling import mount_unbalanced_base, smote_equalizer
from preprocessing.transforming import transform_data
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import mrmr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def original_calls(basecsv):
    # Selecionando as colunas referentes ao trabalho Carlos Alberto
    dataset = basecsv[CARLOS_ALBERTO]
    df = dataset.describe()
    # print(dataset.describe())
    # print(dataset.columns)

    # Montando a base para rodar
    base = mount_unbalanced_base(dataset)

    # Separando as variáveis de entrada e a de saída
    X, y = base.iloc[:, 1:38], base.iloc[:, 0]
    # print(f'y:{Y}')

    # Transformações dos dados
    X_train, X_test, y_train, y_test = transform_data(X, y)

    # Equalizando com SMOTE
    X_train, y_train = smote_equalizer(X_train, y_train)

    # Imprimindo a distribuição
    # print('y_train class distribution')
    # print(y_train.value_counts(normalize=True))
    # print('y_test class distribution')
    # print(y_test.value_counts(normalize=True))

    # Guardando as divisões na variável data_sets
    data_sets = [(X_train, y_train), (X_test, y_test)]

    # Usando classificador de churn com MLP (Multilayer Perceptron)
    name = ['Classificador de Churn - MLP']
    grid_search_model = mlp_model(data_sets, name, X, y)
    results = pd.DataFrame(grid_search_model.cv_results_)
    results = results.sort_values(by='rank_test_score', ascending=False)
    print('results mean_test_score', results['mean_test_score'].round(3).head())
    print('results std_test_score', results['std_test_score'].round(3).head())
    report(results.cv_results_)


# Lê a base de dados, cria a coluna de churn e no-churn como detratores
def read_database(filename):
    # Lendo a base de dados
    _df = pd.read_csv(filename, sep=',')

    # Criando a coluna detratores
    # Essa coluna será:
    # 1 - churn
    # 0 - no-churn
    #
    # O atributo classe pode receber três valores: 0, 1 e 2
    # somente os 0s são detratores e geram churn. 1 e 2 são no-churn
    _df['classe'] = _df['classe'].map(lambda x: 1 if x == 0 else 0)

    return _df


# Procura por missing_values e retorna true se encontrar
def missing_values(_df):
    # Verifica se tem algum NaN ou Null no dataframe
    return _df.isna().any().any() or _df.isnull().any().any()


# Serve para realizar uam análise exploratória e retirar linhas e colunas problemáticas da base
def exploratory_data_analysis(_df):
    # LINHAS:
    # A coluna safra_geracao só possui dois valores possíveis 0 e 201907.
    # As linhas que apresentam 0 aqui, também apresentam zeros em várias outras colunas na forma de um padrão.
    # Podemos descartar as linhas com safra_geracao 0 e, posteriormente, a própria coluna safra_geracao
    indexes = list(_df.loc[_df['safra_geracao'] == 0].index)

    # Procurando por linhas que são zeros de ponta a ponta para remover
    indexes = indexes + list(_df.loc[(_df == 0).all(axis=1)].index)

    # Removendo linhas problemáticas
    _df = _df.drop(indexes)

    # COLUNAS
    # Adicionando o safra_geracao para remoção
    remove_columns = ['safra_geracao']

    # Colunas com um único valor para todas as linhas e que não discrimina nada
    for column in _df.columns[:-1]:
        unique = _df[column].unique()
        if len(unique) == 1:
            remove_columns.append(column)

    # Removendo colunas
    _df = _df.drop(columns=remove_columns, axis='columns')

    print(_df.shape)
    return _df


# Separando as variáveis por tipo e retornando um dicionário
def separate_vars(_df):
    _variables = {
        'output': ['classe'],
        'binary': [],
        'category': ['tmcode'],
        'num': [],
        'num_discrete': [],
        'num_continuous': []
    }

    # Separando as variáveis categóricas binárias
    for column in _df.columns[:-1]:
        unique = _df[column].unique()
        if len(unique) == 2:
            _variables['binary'].append(column)

    # Separando as variáveis categóricas
    # Buscando por unique values até 100 diferentes, só foram encontradas quantidades e amounts.
    # A única que ofereceu interesse foi a tmcode que separa a amostra em 48 grupos não ordenados.
    # for column in _df.columns[:-1]:
    #     if column not in _variables['binary']:
    #         unique = _df[column].unique()
    #         if len(unique) < 100:
    #             print(f'{column}: {unique}')

    # Separando as variáveis numéricas
    _variables['num'] = [column for column in _df.columns if
                         column not in _variables['output'] and
                         column not in _variables['binary'] and column
                         not in _variables['category']]

    # Separando as variáveis discretas e contínuas
    for var in _variables['num']:
        if isinstance(_df[var][0], np.int64):
            _variables['num_discrete'].append(var)
        else:
            _variables['num_continuous'].append(var)

    return _variables


# Removendo colunas por porcentagem de zeros
def remove_zero_columns(_df, _variables, percentage):
    zeros = []
    for column in _variables['num']:
        if len(_df[_df[column] == 0]) / len(_df) > percentage:
            zeros.append(column)

    _df = _df.drop(columns=zeros, axis='columns')

    for i in zeros:
        if i in _variables['num']:
            _variables['num'].remove(i)
        if i in _variables['num_discrete']:
            _variables['num_discrete'].remove(i)
        if i in _variables['num_continuous']:
            _variables['num_continuous'].remove(i)

    return _df, _variables


# Imprime quantidades de variáveis
def print_variables(_variables):
    print(f'Tamanho: {len(_variables["output"])} - {_variables["output"]}')
    print(f'Tamanho: {len(_variables["binary"])} - {_variables["binary"]}')
    print(f'Tamanho: {len(_variables["category"])} - {_variables["category"]}')
    print(f'Tamanho: {len(_variables["num"])} - {_variables["num"]}')
    print(f'Tamanho: {len(_variables["num_discrete"])} - {_variables["num_discrete"]}')
    print(f'Tamanho: {len(_variables["num_continuous"])} - {_variables["num_continuous"]}')


# Removendo outliers com o DBSCAN
def dbscan_remove_outliers(_df, _variables):
    selected = _variables['num']

    # Calculando o epsilon
    # neigh = NearestNeighbors(n_neighbors=len(selected))
    # nbrs = neigh.fit(_df[selected])
    #
    # distances, indices = nbrs.kneighbors(_df[selected])
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    #
    # # Procurar o cotovelo da função, aonde ela já está ficando vertical
    # plt.figure(figsize=(20, 10))
    # plt.plot(distances)
    # plt.title('K-distance Graph', fontsize=20)
    # plt.xlabel('Data Points sorted by distance', fontsize=14)
    # plt.ylabel('Epsilon', fontsize=14)
    # # plt.ylim(0, 10 ** 10)
    # # plt.xlim(63000, 65000)
    # plt.show()

    model = DBSCAN(eps=6 * 10 ** 9, min_samples=len(selected) + 1).fit(_df[selected])

    outliers_continuas = _df[model.labels_ == -1]
    print(len(outliers_continuas))


# Realiza a normalização pelo método selecionado
def normalize(_df, norm_type="MIN_MAX"):
    scaler = None

    if norm_type == 'MIN_MAX':
        scaler = MinMaxScaler()
    elif norm_type == 'Z-SCORE':
        scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(_df))


# Seleção de variáveis
def variable_selection(_df, selection_type='mrmr', number_of_features=30):
    # TODO Para o mrmr usar somente as contínuas
    # TODO Para o chi2 usar binárias e discretas
    # TODO Retirar o if, pq eu terei de usar os dois métodos
    # TODO Mudar o número de seleção de variáveis. Definir quantas quero de cada tipo
    X = _df.iloc[:, 0:-1]
    y = _df.iloc[:, -1]

    if selection_type == 'mrmr':
        # Input contínua/Saída discreta
        # TODO tenho que selecionar somente as colunas contínuas
        return mrmr.mrmr_classif(X, y, K=number_of_features)
    if selection_type == 'chi2':
        # Input discreta/Saída discreta
        # TODO usar o chi quadrado para todas as outras
        test = SelectKBest(score_func=chi2, k=number_of_features)
        fit = test.fit(X, y)

        np.set_printoptions(precision=3)
        print(fit.scores_)

        features = fit.transform(X)
        print(features.columns)
    else:
        print('Método não reconhecido')
        return None


if __name__ == '__main__':
    # Lendo a base de dados e agregando os valores da classe em churn e no-churn
    df = read_database('base.csv')

    # Procurando missing values
    if not missing_values(df):
        print("Não foram encontrados Missing Values")
    else:
        print("Missing value encontrado. Realize o tratamento")

    # Descartando linhas e colunas que não servem com análise exploratória
    df = exploratory_data_analysis(df)

    # Separando as variáveis
    variables = separate_vars(df)

    # Remover colunas com mais de 40% de zeros. Não pode fazer isso para as binárias
    df, variables = remove_zero_columns(df, variables, 0.4)

    # Imprimir quantidades de variáveis
    print_variables(variables)

    # Remover inconsistências
    # WEKA?

    # Normalização pelo min_max para não retirar os outliers
    # df = normalize(df)

    # Seleção de variáveis pelos métodos mRMR e chi2
    variable_selection(df)

    # Removendo outliers pelo DBSCAN
    # dbscan_remove_outliers(df, variables)

    print(df)


