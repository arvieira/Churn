import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# Removendo outliers com o DBSCAN
def dbscan_remove_outliers(df, variables, epsilon):
    print("\t-> Iniciando DBSCAN para remoção de outliers.")
    selected = df.drop(columns=variables['output']).columns

    if not epsilon:
        print("\t-> Exibindo gráfico do DBSCAN para calculo do epsilon...")
        # Calculando o epsilon
        neigh = NearestNeighbors(n_neighbors=len(selected))
        nbrs = neigh.fit(df[selected])

        distances, indices = nbrs.kneighbors(df[selected])
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]

        # Procurar o cotovelo da função, aonde ela já está ficando vertical
        plt.figure(figsize=(20, 10))
        plt.plot(distances)
        plt.title('K-distance Graph\nAnote o cotovelo para o epsilon', fontsize=20)
        plt.xlabel('Data Points sorted by distance', fontsize=14)
        plt.ylabel('Epsilon', fontsize=14)
        plt.show()

        # Pegando o valor de epsilon do usuário
        try:
            epsilon = int(input('\tPor favor, insira o valor do eixo y no cotovelo para o epsilon: '))
        except ValueError:
            print('O número inserido não é um inteiro.')

    if isinstance(epsilon, int) and epsilon > 0:
        print("\t-> Removendo outliers com DBSCAN...")
        model = DBSCAN(eps=epsilon, min_samples=len(selected) + 1).fit(df[selected])
        outliers = df[model.labels_ == -1]

        df = df.drop(outliers.index)
        print(f"\t-> {len(outliers)} outliers removidos da base. Base: {df.shape}")
    else:
        print(f'\t-> O epsilon inserido não é um inteiro ou não é maior que zero. Não removendo outliers.')

    return df
