from matplotlib import pyplot as plt 
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px

df_corr = pd.read_csv("CSV_IA.csv", sep=",")

def correlation(df_corr):
    """
    Give the correlation matrix and print it

    """
    corr_matrix = df_corr.corr()
    # print(corr_matrix["descr_grav"])
    # print(corr_matrix)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(range(df_corr.select_dtypes(['number']).shape[1]), df_corr.select_dtypes(['number']).columns, fontsize=12, rotation=90)
    plt.yticks(range(df_corr.select_dtypes(['number']).shape[1]), df_corr.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()
