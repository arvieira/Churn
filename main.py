import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, cv
import xgboost as xgb
from sklearn import svm


# Semente utilizada para reprodutibilidade de experimentos
from preprocessing.util import preprocessing

SEED = 7457854


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


# Balanceamento da base e separação em treino e teste
def balance_train_test(df, alg='random'):
    # Separando as variáveis de entrada e saída em X e y
    X = df.drop(columns=['classe'])
    y = df['classe']
    # print(X)
    # print(y)

    # Verificando o balanceamento da base
    # Balanceando a base de dados com undersampling aleatório como passo inicial
    # Verificando o balanceamento da base após o undersampling
    # print(df['classe'].value_counts())
    X, y = balance(X, y, alg)
    # print(y.value_counts())

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


# Balanceia a base com o algoritmo escolhido
def balance(df_X, df_y, alg):
    if alg == 'random':
        print(f"-> Baleanceando a base. Método escolhido: Random...")
        sampler = RandomUnderSampler(sampling_strategy='majority')
    elif alg == 'tomek':
        print(f"-> Baleanceando a base. Método escolhido: TomekLink...")
        sampler = TomekLinks()
    else:
        print('Algoritmo de sampling não conhecido')
        return df_X, df_y

    return sampler.fit_resample(df_X, df_y)


# Calcula e imprime a matriz de confusão para um modelo
def evaluate(alg, real, predicted):
    print(f"-> Avaliando o {alg} com matriz de confusão...")
    print(pd.DataFrame(confusion_matrix(real, predicted), columns=['No Churn Pred', 'Churn Pred'],
                       index=['No Churn Real', 'Churn Real']))
    print(metrics.classification_report(real, predicted, digits=5))


# Executa o treino e a avaliação dos modelos escolhidos
def train_models(df_X_train, df_X_test, df_y_train, df_y_test):
    # Treinando e avaliando o modelo XGBoost no modo raw
    create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test)

    # Treinando e avaliando o modelo SVM no modo raw
    create_svm(df_X_train, df_X_test, df_y_train, df_y_test)

    # Treinando e avaliando o modelo SVM no modo raw com AdaBoost
    # create_svm_adaboost(df_X_train, df_X_test, df_y_train, df_y_test)


# Treinando e avaliando o XGBoost
def create_xgboost(df_X_train, df_X_test, df_y_train, df_y_test):
    # Sem cross-validation
    print("-> Treinando o XGBoost...")
    xgboost = XGBClassifier()
    xgboost.fit(df_X_train, df_y_train)
    df_y_pred = xgboost.predict(df_X_test)
    evaluate('XGBoost', df_y_test, df_y_pred)

    # Com cross-validation
    # print("-> Treinando o XGBoost com cross-validation...")
    # dtrain = xgb.DMatrix(data=df_X_train, label=df_y_train)
    # params = {
    #     "objective": "binary:logistic",
    #     'colsample_bytree': 0.3,
    #     'learning_rate': 0.1,
    #     'max_depth': 5,
    #     'alpha': 10
    # }
    # xgb_cv = cv(params=params, dtrain=dtrain, nfold=5, metrics='auc', seed=SEED)
    # print(xgb_cv)


# Treinando e avaliando o SVM
def create_svm(df_X_train, df_X_test, df_y_train, df_y_test):
    # Sem cross-validation
    print("-> Treinando o SVM...")
    clf = svm.SVC(kernel='linear', random_state=SEED, max_iter=10000)
    clf.fit(df_X_train, df_y_train)
    df_y_pred = clf.predict(df_X_test)
    evaluate('SVM', df_y_test, df_y_pred)


# Treinando e avaliando o SVM com AdaBoost
def create_svm_adaboost(df_X_train, df_X_test, df_y_train, df_y_test):
    # Sem cross-validation
    print("-> Treinando o SVM com AdaBoost...")
    clf = AdaBoostClassifier(
        svm.SVC(probability=True, kernel='linear', random_state=SEED, max_iter=5000),
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME'
    )
    clf.fit(df_X_train, df_y_train)
    df_y_pred = clf.predict(df_X_test)
    evaluate('SVM com AdaBoost', df_y_test, df_y_pred)


if __name__ == '__main__':
    # Lendo a base de dados e agregando os valores da classe em churn e no-churn
    data = read_database('base.csv')
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
    data = preprocessing(data)

    # Balanceando a base e separando treino e teste
    X_train, X_test, y_train, y_test = balance_train_test(data, 'tomek')

    # Treinando os modelos com preprocessamento
    train_models(X_train, X_test, y_train, y_test)
