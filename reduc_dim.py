from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px


def correlation(df_corr):
    """
    Give the correlation matrix and print it

    """
    corr_matrix = df_corr.corr()
    # print(corr_matrix["descr_grav"])
    # print(corr_matrix)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(
        range(df_corr.select_dtypes(["number"]).shape[1]),
        df_corr.select_dtypes(["number"]).columns,
        fontsize=12,
        rotation=90,
    )
    plt.yticks(
        range(df_corr.select_dtypes(["number"]).shape[1]),
        df_corr.select_dtypes(["number"]).columns,
        fontsize=14,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()


def reduc_dim_grav(df_corr):
    """
    Drop non-interesting data in the matrix

    """
    corr_matrix = df_corr.corr()
    list_corr = abs(corr_matrix["descr_grav"])
    # print(corr_matrix["descr_grav"])
    corr_ordered = list_corr.sort_values()
    # print(corr_matrix)
    return df_corr.drop(
        columns=[
            "mois",
            "id_usa",
            "descr_athmo",
            "descr_etat_surf",
            "descr_agglo",
            "num_veh",
            "X",
            "descr_motif_traj",
            "place",
            "descr_dispo_secu",
            "heure",
            "id_code_insee",
        ]
    )


# reduc_dim_grav(df_corr)
# df_corr.info()
