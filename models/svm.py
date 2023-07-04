from sklearn import svm

from models.constraints import SEED
from resultados.evaluator import evaluate


# Treinando e avaliando o SVM
def create_svm(df_X_train, df_X_test, df_y_train, df_y_test):
    # Sem cross-validation
    print("-> Treinando o SVM...")
    clf = svm.SVC(kernel='linear', random_state=SEED, max_iter=10000)
    clf.fit(df_X_train, df_y_train)
    df_y_pred = clf.predict(df_X_test)
    evaluate('SVM', df_y_test, df_y_pred)