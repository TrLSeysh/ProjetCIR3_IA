"""

Function used to evaluate results obtained during IA learning

"""
import sklearn.metrics as sk_m
import numpy as np
import k_mean_scikit as p_kmsci
import K_mean_scratch as p_kmscr
import matplotlib.pyplot as plt


def evaluate_kmeans(df_prep):
    """
    Evaluate regression results using :

    • Silhouette Coefficient
    • Calinski Harabasz Index
    • Davies Bouldin Index

    """
    s_score = {"from_scratch": [], "scikit": []}
    ch_score = {"from_scratch": [], "scikit": []}
    bd_score = {"from_scratch": [], "scikit": []}
    nb_clusters = [3, 5, 8, 12]

    X = np.column_stack((df_prep["longitude"], df_prep["latitude"]))

    for i in enumerate(nb_clusters):
        calc_score(s_score, ch_score, bd_score, X, i[1])
    
    plt.plot(nb_clusters, s_score["scikit"], color="red", marker="o", label="scikit")
    plt.plot(nb_clusters, s_score["from_scratch"], color="black", marker="o", label="from scratch")
    
    plt.xticks(nb_clusters)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score obtenu (0 à 1)")
    plt.title("Coefficient de silhouette en fonction du nombre de cluster")
    plt.legend()
    plt.show()

    plt.plot(nb_clusters, ch_score["scikit"], color="red", marker="o", label="scikit")
    plt.plot(nb_clusters, ch_score["from_scratch"], color="black", marker="o", label="from scratch")

    plt.xticks(nb_clusters)
    plt.xticks(nb_clusters)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score obtenu (0 à ∞)")
    plt.title("Index de Calinski Harabasz en fonction du nombre de cluster")
    plt.legend()
    plt.show()

    plt.plot(nb_clusters, bd_score["scikit"], color="red", marker="o", label="scikit")
    plt.plot(nb_clusters, bd_score["from_scratch"], color="black", marker="o", label="from_scratch")

    plt.xticks(nb_clusters)
    plt.xticks(nb_clusters)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score obtenu (0 à 1)")
    plt.title("Index de Bouldin Davies en fonction du nombre de cluster")
    plt.legend()
    plt.show()



def calc_score(s_score, ch_score, bd_score, X, nb_clusters):
    """
    Calculate score for evaluate_kmeans()

    """
    print(f"\nCalcul des scores pour n = {nb_clusters}...\n")
    pred = p_kmsci.k_mean(X, nb_clusters)

    s_score["scikit"].append(sk_m.silhouette_score(X, pred["X_pred"]))
    ch_score["scikit"].append(sk_m.calinski_harabasz_score(X, pred["X_pred"]))
    bd_score["scikit"].append(sk_m.davies_bouldin_score(X, pred["X_pred"]))

    print(
        f"Résultat coefficient de silhouette avec n = {nb_clusters} et méthode scikit : ",
        s_score["scikit"][-1],
    )
    print(
        f"Résultat index de Calinski Harabasz avec n = {nb_clusters} et méthode scikit : ",
        ch_score["scikit"][-1],
    )
    print(
        f"Résultat index de Bouldin Davies avec n = {nb_clusters} et méthode scikit :",
        bd_score["scikit"][-1],
        "\n",
    )

    labels, centroids = p_kmscr.kmeans(X[:, 0], X[:, 1], nb_clusters)

    s_score["from_scratch"].append(sk_m.silhouette_score(X, labels))
    ch_score["from_scratch"].append(sk_m.calinski_harabasz_score(X, labels))
    bd_score["from_scratch"].append(sk_m.davies_bouldin_score(X, labels))

    print(
        f"Résultat coefficient de silhouette avec n = {nb_clusters} et méthode from scratch : ",
        s_score["from_scratch"][-1],
    )
    print(
        f"Résultat index de Calinski Harabasz avec n = {nb_clusters} et méthode from scratch : ",
        ch_score["from_scratch"][-1],
    )
    print(
        f"Résultat index de Bouldin Davies avec n = {nb_clusters} et méthode from scratch :",
        bd_score["from_scratch"][-1],
        "\n",
    )


def evaluate_classifaction(X, X_pred):
    """
    evaluate classification results using :

    • F1-Score
    • Confusion matrix
    • Precision-Recall
    • Compute Area Under the Receiver Operating Characteristic Curve

    """
    print("none")
