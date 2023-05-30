from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

import prep_data as p_pd

df = pd.read_csv("IA_csv.csv", sep=",")
df.info()
# Num_Acc,num_veh,id_usa,date,ville,id_code_insee,latitude,longitude,descr_cat_veh,descr_agglo,
# descr_athmo,descr_lum,descr_etat_surf,description_intersection,an_nais,age,place,
# descr_dispo_secu,descr_grav,descr_motif_traj,descr_type_col
# df.drop(columns=['',''])

print(df.columns)

# corr_matrix = df.corr()
# print(corr_matrix["descr_grav"])

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

