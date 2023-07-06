import mrmr
from kydavra import ReliefFSelector
from sklearn.decomposition import PCA


# Seleção de variáveis
def variable_selection(df, variables, n_features=30):
    print("\t-> Selecionando variáveis...")
    # A ideia é que a dissertação utilizou 27 variáveis com essa base, então tentaremos usar
    # 30 para ver se fica melhor.
    x = df[variables['num_continuous'] + variables['num_discrete'] + variables['binary']]
    y = df[variables['output']]

    # mRMR
    # Instalar
    # pip install mrmr_selection
    # pip install polars
    selected_features = mrmr.mrmr_classif(x, y, K=n_features)
    print(f"\t-> {n_features} variáveis selecionadas: {selected_features}")

    # Separando as variáveis por tipo para o retorno
    selected_variables = {
        'output': ['classe'],
        'binary': [value for value in variables['binary'] if value in selected_features],
        'num': [],
        'num_discrete': [value for value in variables['num_discrete'] if value in selected_features],
        'num_continuous': [value for value in variables['num_continuous'] if value in selected_features]
    }
    selected_variables['num'] = selected_variables['num_discrete'] + selected_variables['num_continuous']

    # Removendo as variáveis da base
    return_df = df[selected_features]
    return_df[variables['output']] = y

    # Knowledge base
    # chi2 para entradas discretas positivas com saída discreta
    # Tem valores negativos nas discretas que eu não posso utilizar no chi2
    # print('chi2:')
    # test = SelectKBest(score_func=chi2, k=number_of_features)
    # fit = test.fit(x_discretas, y)
    # np.set_printoptions(precision=3)
    # print(fit.scores_)
    # features = fit.transform(x_discretas)
    # print(features.columns)

    # ReliefF para entradas DISCRETAS com saída discreta
    # Não funciona devido ao excesso de registros
    # fs = ReliefFSelector(n_neighbors=20, n_features=10)
    # x_discretas['classe'] = y
    # selected_discrete = fs.select(x_discretas, 'classe')
    #
    # fs = ReliefF(n_neighbors=20, n_features_to_keep=10)
    # X_train = fs.fit_transform(x_discretas.values, y)

    return return_df, selected_variables
