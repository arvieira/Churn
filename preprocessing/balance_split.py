from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.model_selection import train_test_split
from models.constraints import SEED


# Balanceamento da base e separação em treino e teste
def base_balance(df, alg='random'):
    # Separando a base
    X, y = get_xy(df)

    # Verificando o balanceamento da base
    # Balanceando a base de dados com undersampling aleatório como passo inicial
    # Verificando o balanceamento da base após o undersampling
    # print(df['classe'].value_counts())
    X, y = balance(X, y, alg)
    # print(y.value_counts())

    # Juntando a base
    X['classe'] = y

    return X


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


def base_split(df):
    # Separando a base
    X, y = get_xy(df)

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


# Separa as variáveis da saída
def get_xy(df):
    # Separando as variáveis de entrada e saída em X e y
    X = df.drop(columns=['classe'])
    y = df['classe']
    # print(X)
    # print(y)

    return X, y
