from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('C:/Users/Adrien/OneDrive/Bureau/Work/ISEN/Big_Data/Projet A3-20230522/ProjetCIR3_IA/csv_cleaned.csv', delimiter=',', encoding="ISO-8859-1")
# line_count = 0
# for row in data:
#     if line_count == 0:
#         print(f'Column names are {", ".join(row)}')
#     line_count += 1
       
#     print(f'Processed {line_count} lines.')


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

# # Calcul de la corrélation
# mX = sum(x)/len(x)
# mY = sum(y)/len(y)

# cov = sum((a - mX) * (b - mY) for (a,b) in zip(x,y)) / len(x)

# stdevX = (sum((a - mX)**2 for a in x)/len(x))**0.5
# stdevY = (sum((b - mY)**2 for b in y)/len(y))**0.5

# corr_pearson = round(cov/(stdevX*stdevY),3)