import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from matplotlib import pyplot as plt 

df_knn = pd.read_csv("CSV_IA_red.csv", sep=",")
df_knn = df_knn.dropna()

def kNN_scikit(df_knn):
    """
    KNN Classification with sklearn

    """
    X = df_knn.drop("descr_grav", axis=1)
    X = X.values
    y = df_knn["descr_grav"]
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12345
    )
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    # Evaluation RMSE sur base Training
    train_preds = knn_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print('RMSE sur base Training', rmse)

    # Evaluation RMSE sur base Test
    test_preds = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print('RMSE sur base Test', rmse)

    # Ploting the predict classes classification
    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(
       X_test[:, 1], X_test[:, 0], c=test_preds, s=50, cmap=cmap
    )
    f.colorbar(points)
    plt.show()

kNN_scikit(df_knn)