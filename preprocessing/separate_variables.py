import numpy as np


# Separando as variáveis por tipo e retornando um dicionário
def separate_vars(df):
    print("\t-> Separando variáveis.")

    variables = {
        'output': ['classe'],
        'binary': [],
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
                        column not in variables['binary']]

    # Separando as variáveis discretas e contínuas
    for var in variables['num']:
        if isinstance(df[var][0], np.int64):
            variables['num_discrete'].append(var)
        else:
            variables['num_continuous'].append(var)

    return variables
