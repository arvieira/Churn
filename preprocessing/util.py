import mrmr
from kydavra import ReliefFSelector
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Procura por missing values
from models.constraints import SEED


def missing_values(df):
    if df.isna().any().any() or df.isnull().any().any():
        print("\t-> Missing value encontrado. Realize o tratamento")
        return True
    else:
        print(f"\t-> Não foram encontrados Missing Values.")
        return False


# Realiza uma análise exploratória e retira linhas e colunas problemáticas
def exploratory_data_analysis(df):
    print("\t-> Realizando análise exploratória.")

    # LINHAS:
    # A coluna safra_geracao só possui dois valores possíveis 0 e 201907.
    # As linhas que apresentam 0 aqui, também apresentam zeros em várias outras colunas na forma de um padrão.
    # Podemos descartar as linhas com safra_geracao 0 e, posteriormente, a própria coluna safra_geracao
    indexes = list(df.loc[df['safra_geracao'] == 0].index)

    # Procurando por linhas que são zeros de ponta a ponta para remover
    indexes = indexes + list(df.loc[(df == 0).all(axis=1)].index)

    # Removendo linhas problemáticas
    df = df.drop(indexes)

    # COLUNAS
    # Adicionando o safra_geracao para remoção
    remove_columns = ['safra_geracao']

    # Colunas com um único valor para todas as linhas e que não discrimina nada
    for column in df.columns[:-1]:
        unique = df[column].unique()
        if len(unique) == 1:
            remove_columns.append(column)

    # Removendo colunas
    df = df.drop(columns=remove_columns, axis='columns')

    return df


# Separando as variáveis por tipo e retornando um dicionário
def separate_vars(df):
    print("\t-> Separando variáveis.")

    variables = {
        'output': ['classe'],
        'binary': [],
        'category': ['tmcode'],
        'num': [],
        'num_discrete': [],
        'num_continuous': []
    }

    # Separando as variáveis categóricas binárias
    for column in df.columns[:-1]:
        unique = df[column].unique()
        if len(unique) == 2:
            variables['binary'].append(column)

    # Separando as variáveis categóricas
    # Buscando por unique values até 100 diferentes, só foram encontradas quantidades e amounts.
    # A única que ofereceu interesse foi a tmcode que separa a amostra em 48 grupos não ordenados e já está
    # adicionada a lista na inicialização da variables
    # for column in _df.columns[:-1]:
    #     if column not in _variables['binary']:
    #         unique = _df[column].unique()
    #         if len(unique) < 100:
    #             print(f'{column}: {unique}')

    # Separando as variáveis numéricas
    variables['num'] = [column for column in df.columns if
                        column not in variables['output'] and
                        column not in variables['binary'] and
                        column not in variables['category']]

    # Separando as variáveis discretas e contínuas
    for var in variables['num']:
        if isinstance(df[var][0], np.int64):
            variables['num_discrete'].append(var)
        else:
            variables['num_continuous'].append(var)

    return variables


# Removendo variáveis com mais do que X% de zeros
def remove_zero_columns(df, variables, percentage):
    print(f"\t-> Removendo colunas com {percentage*100}% de registros com zeros.")

    zeros = []
    for column in variables['num']:
        if len(df[df[column] == 0]) / len(df) >= percentage:
            zeros.append(column)

    df = df.drop(columns=zeros, axis='columns')

    for i in zeros:
        if i in variables['num']:
            variables['num'].remove(i)
        if i in variables['num_discrete']:
            variables['num_discrete'].remove(i)
        if i in variables['num_continuous']:
            variables['num_continuous'].remove(i)

    return df, variables


# Seleção de variáveis
def variable_selection(df, variables, n_features=30):
    print("\t-> Selecionando variáveis...")
    # A ideia é que a dissertação utilizou 27 variáveis com essa base, então tentaremos usar
    # 30 para ver se fica melhor.
    x = df[variables['num_continuous'] + variables['num_discrete'] + variables['binary']]
    y = df[variables['output']]

    # mRMR
    # Instalar
    # pip install mrmr_selection
    # pip install polars
    selected_features = mrmr.mrmr_classif(x, y, K=n_features)
    print(f"\t-> {n_features} variáveis selecionadas: {selected_features}")

    # Separando as variáveis por tipo para o retorno
    selected_variables = {
        'output': ['classe'],
        'category': ['tmcode'],
        'binary': [value for value in variables['binary'] if value in selected_features],
        'num': [],
        'num_discrete': [value for value in variables['num_discrete'] if value in selected_features],
        'num_continuous': [value for value in variables['num_continuous'] if value in selected_features]
    }
    selected_variables['num'] = selected_variables['num_discrete'] + selected_variables['num_continuous']

    # Removendo as variáveis da base
    return_df = df[selected_features]
    return_df[variables['output']] = y

    # Knowledge base
    # chi2 para entradas discretas positivas com saída discreta
    # Tem valores negativos nas discretas que eu não posso utilizar no chi2
    # print('chi2:')
    # test = SelectKBest(score_func=chi2, k=number_of_features)
    # fit = test.fit(x_discretas, y)
    # np.set_printoptions(precision=3)
    # print(fit.scores_)
    # features = fit.transform(x_discretas)
    # print(features.columns)

    # ReliefF para entradas DISCRETAS com saída discreta
    # Não funciona devido ao excesso de registros
    # fs = ReliefFSelector(n_neighbors=20, n_features=10)
    # x_discretas['classe'] = y
    # selected_discrete = fs.select(x_discretas, 'classe')
    #
    # fs = ReliefF(n_neighbors=20, n_features_to_keep=10)
    # X_train = fs.fit_transform(x_discretas.values, y)

    return return_df, selected_variables


# Função para normalizar as variáveis continuas e discretas
def normalize(df, variables, norm_type='MIN_MAX'):
    scaler = None

    if norm_type == 'MIN_MAX':
        print("\t-> Realizando normalização por Min-Max.")
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = MinMaxScaler()
    elif norm_type == 'Z-SCORE':
        print("\t-> Realizando normalização por Z-score.")
        scaler = StandardScaler()

    df[variables['num']] = scaler.fit_transform(df[variables['num']])

    return df


# Função principal do preprocessamento
def preprocessing(data, n_features):
    print(f"-> Realizando preprocessamentos...")

    # Procurando missing values
    missing_values(data)

    # Descartando linhas e colunas que não servem com análise exploratória
    data = exploratory_data_analysis(data)

    # Separando as variáveis
    separated = separate_vars(data)

    # Removendo variáveis que mais de 40% das amostras possuem o zero
    # Não remove das variáveis binárias, pq é natural que talvez 50% seja zero e 50%, seja um
    data, separated = remove_zero_columns(data, separated, 0.4)

    # Seleção de variáveis
    data, separated = variable_selection(data, separated, n_features=n_features)

    # Normalizando os dados
    data = normalize(data, separated)

    return data
