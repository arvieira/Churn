from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

from models.constraints import SEED
from results.evaluator import evaluate


# Treinando e avaliando o SVM com AdaBoost
def create_svm_adaboost(df_X_train, df_X_test, df_y_train, df_y_test, grid_search=False):
    # Sem cross-validation
    print("-> Treinando o SVM com AdaBoost...")
    clf = AdaBoostClassifier(
        svm.SVC(probability=True, kernel='rbf', random_state=SEED, max_iter=10000, C=1, gamma=1),
        n_estimators=55,
        learning_rate=1.0,
        algorithm='SAMME'
    )
    clf.fit(df_X_train, df_y_train)

    df_y_pred = clf.predict(df_X_test)
    pred_proba = clf.predict_proba(df_X_test)[::, 1]
    return evaluate('SVM com AdaBoost', df_y_test, df_y_pred, pred_proba)