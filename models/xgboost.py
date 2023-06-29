import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

# TODO Verificar se precisa disso tudo na carga do iris ou se posso usar iris['data'] e iris['target']

# Carregando a base do Iris
iris = datasets.load_iris()

# Criando um DataFrame
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target

# Nomeando as colunas e removendo missing values
iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True)

# Separando variáveis de entrada e saída
iris_X = iris_df.iloc[:, [0, 1, 2, 3]]
iris_Y = iris_df['class']

# Dividindo a base em treino e teste
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=0.3, random_state=1234513)

# TODO colocar um GridSearchCV para os hyperparâmetros do XGBoost

# Criando o modelo com hyperparâmetros
model = XGBClassifier(n_estimators=50, subsample=0.5, max_depth=4, eta=0.1, gamma=0, reg_alpha=0, reg_lambda=1,
                      scale_pos_weight=0.2, booster='gbtree', eval_metric='logloss')
model.fit(X_train, y_train)
print(model)

# Imprimindo a acurácia
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia: %.2f%%" % (accuracy * 100.0))
