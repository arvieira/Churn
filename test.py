import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


data = pd.read_csv('bases/SelecaoVariaveis.csv')

print(data.describe())
print(data[data['Acurácia'] > 0.69].sort_values('Acurácia', ascending=False).to_string())

# split = data[['Número de Variáveis', 'AUC']]
# split.set_index('Número de Variáveis').plot(color='red')
# plt.ylabel('AUC')
# plt.title('AUC por Número de Variáveis Selecionadas')
# plt.show()
