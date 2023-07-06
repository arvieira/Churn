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
    # Adicionando o safra_geracao e tmcode para remoção pq são um identificadores
    # Adicionando churn_score para remoção pq ela está altamente relacionada à saída
    remove_columns = ['safra_geracao', 'churn_score', 'tmcode']

    # Colunas com um único valor para todas as linhas e que não discrimina nada
    for column in df.columns[:-1]:
        unique = df[column].unique()
        if len(unique) == 1:
            remove_columns.append(column)

    # Removendo colunas
    df = df.drop(columns=remove_columns, axis='columns')

    return df
