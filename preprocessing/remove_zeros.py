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
