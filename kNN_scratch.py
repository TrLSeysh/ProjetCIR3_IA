import pandas as pd
import numpy as np
import scipy.stats

df_knn = pd.read_csv("CSV_IA_red.csv", sep=",")

def kNN_scratch(df_knn):
    """
    KNN Classification from scratch

    """
    X = df_knn.drop("descr_grav", axis=1)
    X = X.values
    y = df_knn["descr_grav"]
    y = y.values

    new_data = np.array([
         47.8,
         -2.2,
         0,
         1,
         0,
         2002,
         0,
    ])

    distances = np.linalg.norm(X - new_data, axis=1)
    k = 3
    nearest_neighbor_ids = distances.argsort()[:k]
    # print(nearest_neighbor_ids)
    nearest_neighbor_grav = y[nearest_neighbor_ids]
    # print(nearest_neighbor_grav)
    prediction = nearest_neighbor_grav.mean()
    print('Prediction de gravité: ',prediction)

    mode_result = scipy.stats.mode(y, keepdims=True)
    modalite = mode_result.mode[0]

    print("Modalité la plus commune de descr_grav :", modalite)

