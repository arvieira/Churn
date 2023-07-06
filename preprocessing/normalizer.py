from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Função para normalizar as variáveis continuas e discretas
def normalize(df, variables, norm_type):
    scaler = None

    if norm_type == 'MIN_MAX':
        print("\t-> Realizando normalização por Min-Max.")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # scaler = MinMaxScaler()
    elif norm_type == 'Z-SCORE':
        print("\t-> Realizando normalização por Z-score.")
        scaler = StandardScaler()

    df[variables['num']] = scaler.fit_transform(df[variables['num']])

    return df
