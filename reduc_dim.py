from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

import prep_data as p_pd

df = pd.read_csv("csv_cleaned.csv", sep=",")
df.info()

# print(df.columns)

corr_matrix = df.corr()
print(corr_matrix["descr_grav"])

# tot_corr = np.corrcoef(df)
# corr = np.corrcoef(df['descr_grav'], df['num_veh'])
# print(tot_corr)
# print(corr)






# pca = PCA(n_components=2)
# X_red = pca.fit_transform(X)

# print("X : ", X.shape)
# print("X reduced: ", X_red.shape)

# 
# plt.figure(figsize=(13,10))
# plt.scatter(X_red[:,0],X_red[:,1],c=y)
# plt.axis("off")
# plt.colorbar()
# plt.show()

