"""
Python file used to cluster data by using scikit-learn library

"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

def plot_cluster(X, y = None):
    """
    Create cluster to display
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=10)
    plt.ylabel("$x_2$", fontsize=10, rotation=0)

def k_mean(df_prep):
    """
    Function which calculate k_mean using sklearn library

    """
    nb_cluster = 21
    X = np.column_stack((df_prep["longitude"], df_prep["latitude"]))

    kmeans = KMeans(n_clusters=nb_cluster, n_init=10).fit(X)
    X_predict = kmeans.predict(X)

    fig = px.scatter_mapbox(X, lat=X[:, 1], lon=X[:, 0], color=X_predict, color_continuous_scale=px.colors.sequential.Viridis)
    fig2 = px.scatter_mapbox(kmeans.cluster_centers_, lat=kmeans.cluster_centers_[:, 1], lon=kmeans.cluster_centers_[:, 0]).update_traces(marker={"size": 10, "color":"red"})
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.add_trace(fig2.data[0])
    fig.show()

