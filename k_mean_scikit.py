"""
Python file used to cluster data by using scikit-learn library

"""
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np


def k_mean(X, nb_cluster):
    """
    Function which calculate k_mean using sklearn library

    """
    kmeans = KMeans(n_clusters=nb_cluster, n_init=10).fit(X)
    X_predict = kmeans.predict(X)

    return {"X_pred": X_predict, "cluster_center": kmeans.cluster_centers_}


def display_kmean(df_prep, nb_cluster):
    """
    Function which display calculated result by k_mean() function

    """
    X = np.column_stack((df_prep["longitude"], df_prep["latitude"]))
    pred = k_mean(X, nb_cluster)

    fig = px.scatter_mapbox(
        X,
        lat=X[:, 1],
        lon=X[:, 0],
        color=pred["X_pred"],
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig2 = px.scatter_mapbox(
        pred["cluster_center"],
        lat=pred["cluster_center"][:, 1],
        lon=pred["cluster_center"][:, 0],
    ).update_traces(marker={"size": 10, "color": "red"})

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.add_trace(fig2.data[0])
    fig.show()
