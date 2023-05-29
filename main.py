import pandas as pd

from models.mlp_model import mlp_model, report
from preprocessing.bases_variables import CARLOS_ALBERTO
from preprocessing.sampling import mount_unbalanced_base, smote_equalizer
from preprocessing.transforming import transform_data


if __name__ == '__main__':
    # Criando o DataFrame com import do CSV
    basecsv = pd.read_csv('base.csv', sep=',')
    basecsv['detratores'] = basecsv['classe'].map(lambda x: 1 if x == 0 else 0)

    # Selecionando as colunas referentes ao trabalho Carlos Alberto
    dataset = basecsv[CARLOS_ALBERTO]
    df = dataset.describe()
    # print(dataset.describe())
    # print(dataset.columns)

    # Montando a base para rodar
    base = mount_unbalanced_base(dataset)

    # Separando as variáveis de entrada e a de saída
    X, y = base.iloc[:, 1:38], base.iloc[:, 0]
    # print(f'y:{Y}')

    # Transformações dos dados
    X_train, X_test, y_train, y_test = transform_data(X, y)

    # Equalizando com SMOTE
    X_train, y_train = smote_equalizer(X_train, y_train)

    # Imprimindo a distribuição
    # print('y_train class distribution')
    # print(y_train.value_counts(normalize=True))
    # print('y_test class distribution')
    # print(y_test.value_counts(normalize=True))

    # Guardando as divisões na variável data_sets
    data_sets = [(X_train, y_train), (X_test, y_test)]

    # Usando classificador de churn com MLP (Multilayer Perceptron)
    name = ['Classificador de Churn - MLP']
    grid_search_model = mlp_model(data_sets, name, X, y)
    results = pd.DataFrame(grid_search_model.cv_results_)
    results = results.sort_values(by='rank_test_score', ascending=False)
    print('results mean_test_score', results['mean_test_score'].round(3).head())
    print('results std_test_score', results['std_test_score'].round(3).head())
    report(results.cv_results_)
