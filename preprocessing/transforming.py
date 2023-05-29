from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def transform_data(X, y):
    # standardizing the input feature
    sc = StandardScaler()
    _X = sc.fit_transform(X)

    # Estratificando a base
    _X_train, _X_test, _y_train, _y_test = train_test_split(X, y, stratify=y, test_size=0.25)

    return _X_train, _X_test, _y_train, _y_test
