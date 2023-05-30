"""
Python file used to cluster data by using scikit-learn library

"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_cluster(X, y = None):
    """
    Create cluster to display
    """
    plt.scatter(X[0], X[1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=10)
    plt.ylabel("$x_2$", fontsize=10, rotation=0)

def k_mean(df_prep):
    """
    Function which calculate k_mean using sklearn library

    """
    X = [df_prep["longitude"], df_prep["latitude"]]

    plt.figure(figsize=(8, 4))
    plot_cluster(X)
    plt.show()

    kmeans = KMeans(n_clusters=10)
    X_predict = kmeans.fit_predict(X)
    print(X_predict)

