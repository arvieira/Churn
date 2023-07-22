from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import datasets


# Carregando a base do Iris
iris = datasets.load_iris()

# Dividindo a base em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=1234513, stratify=iris.target)

# Criando o modelo com hyperparâmetros
model = XGBClassifier()
params = {
    'n_estimators': [30, 50, 70],
    'subsample': [0.5, 1],
    'max_depth': [4, 6],
    'eta': [0.1, 0.5, 1],
    'booster': ['gbtree'],
    'eval_metric': ['logloss']
}
grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring='accuracy',
    cv=5
)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
print(grid.best_params_)

# Imprimindo a acurácia
model = grid.best_estimator_
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia: %.2f%%" % (accuracy * 100.0))
