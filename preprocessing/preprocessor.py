from preprocessing.exploratory_analysis import exploratory_data_analysis
from preprocessing.missing_values import missing_values
from preprocessing.normalizer import normalize
from preprocessing.outlier_remover import dbscan_remove_outliers
from preprocessing.remove_zeros import remove_zero_columns
from preprocessing.separate_variables import separate_vars
from preprocessing.variables_selection import variable_selection


# Função principal do preprocessamento
def preprocessing(data, zero_percentage=0.4, n_features=30, epsilon=None, normalizer='MIN_MAX'):
    print(f"-> Realizando preprocessamentos...")

    # Procurando missing values
    missing_values(data)

    # Descartando linhas e colunas que não servem com análise exploratória
    data = exploratory_data_analysis(data)

    # Separando as variáveis
    separated = separate_vars(data)

    # Removendo variáveis que mais de 40% das amostras possuem o zero
    # Não remove das variáveis binárias, pq é natural que talvez 50% seja zero e 50%, seja um
    data, separated = remove_zero_columns(data, separated, zero_percentage)

    # Seleção de variáveis
    data, separated = variable_selection(data, separated, n_features=n_features)

    # Removendo outliers pelo DBSCAN
    data = dbscan_remove_outliers(data, separated, epsilon)

    # Normalizando os dados
    data = normalize(data, separated, normalizer)

    return data
