from ReliefF import ReliefF
import numpy as np
from sklearn import datasets
import pandas as pd


#example of multi class problem
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target

print(iris_df)

# fs = ReliefF(n_neighbors=20, n_features_to_keep=2)
# X_train = fs.fit_transform(X, Y)
# print("(No. of tuples, No. of Columns before ReliefF) : "+str(iris.data.shape)+
#       "\n(No. of tuples, No. of Columns after ReliefF) : "+str(X_train.shape))