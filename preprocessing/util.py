import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Procura por missing values


def missing_values(df):
    if df.isna().any().any() or df.isnull().any().any():
        print("-> Missing value encontrado. Realize o tratamento")
        return True
    else:
        print("-> Não foram encontrados Missing Values")
        return False


# Realiza uma análise exploratória e retira linhas e colunas problemáticas
def exploratory_data_analysis(df):
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


# Função para normalizar as variáveis continuas e discretas
def normalize(df, variables, norm_type='MIN_MAX'):
    scaler = None

    if norm_type == 'MIN_MAX':
        scaler = MinMaxScaler()
    elif norm_type == 'Z-SCORE':
        scaler = StandardScaler()

    df[variables['num']] = scaler.fit_transform(df[variables['num']])

    return df


# Função principal do preprocessamento
def preprocessing(data):
    # Procurando missing values
    missing_values(data)

    # Descartando linhas e colunas que não servem com análise exploratória
    data = exploratory_data_analysis(data)

    # Separando as variáveis
    separated = separate_vars(data)

    # Removendo variáveis que mais de 40% das amostras possuem o zero
    # Não remove das variáveis binárias, pq é natural que talvez 50% seja zero e 50%, seja um
    data, separated = remove_zero_columns(data, separated, 0.4)

    # Normalizando os dados
    data = normalize(data, separated)

    return data
