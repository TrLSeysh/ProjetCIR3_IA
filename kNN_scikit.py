import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

import repartition_données as rpd


def kNN_scikit(df_knn, X_train, X_test, y_train, y_test):
    """
    KNN Classification with sklearn

    """
    X = df_knn.drop("descr_grav", axis=1)
    X = X.values
    y = df_knn["descr_grav"]
    y = y.values

    knn_model = KNeighborsRegressor(n_neighbors=48)
    knn_model = knn_model.fit(X_train, y_train)
    test_preds = knn_model.predict(X_test).round()

    return test_preds, knn_model


# from sklearn.model_selection import GridSearchCV

# parameters = {"n_neighbors": range(1, 50)}
# gridsearch = GridSearchCV(knn_model, parameters)
# gridsearch.fit(X_train, y_train)
# print(gridsearch.best_params_)
