import pandas as pd

from imblearn.over_sampling import SMOTE


# Monta uma base sem balancear
def mount_unbalanced_base(dataset):
    df_class_1 = dataset.loc[dataset['detratores'] == 0]
    df_class_2 = dataset.loc[dataset['detratores'] == 1]
    return pd.concat([df_class_1, df_class_2], ignore_index=True)


# Monta uma base balanceada com undersampling
def mount_undersampling_base(dataset, sample=17000):
    df_class_1 = dataset.loc[dataset['detratores'] == 0].sample(sample)
    df_class_2 = dataset.loc[dataset['detratores'] == 1].sample(sample)
    return pd.concat([df_class_1, df_class_2], ignore_index=True)


# Equalizando a base criando novas amostras com o SMOTE
def smote_equalizer(_X_train, _y_train):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(_X_train, _y_train)
