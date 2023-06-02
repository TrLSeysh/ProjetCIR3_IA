"""

Function used to evaluate results obtained during IA learning

"""
import sklearn.metrics as sk_m
import numpy as np
import k_mean_scikit as p_kmsci
import K_mean_scratch as p_kmscr
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_kmeans(df_prep):
    """
    Evaluate regression results using :

    • Silhouette Coefficient
    • Calinski Harabasz Index
    • Davies Bouldin Index

    """
    s_score = {"from_scratch": {"L1": [], "L2": [], "Haversine": []}, "scikit": []}
    ch_score = {"from_scratch": {"L1": [], "L2": [], "Haversine": []}, "scikit": []}
    bd_score = {"from_scratch": {"L1": [], "L2": [], "Haversine": []}, "scikit": []}
    nb_clusters = np.array([3, 5, 8, 12])

    X = np.column_stack((df_prep["longitude"], df_prep["latitude"]))

    for i in enumerate(nb_clusters):
        calc_score(s_score, ch_score, bd_score, X, i[1])

    print(s_score["scikit"])
    plt.bar(nb_clusters - 0.3, s_score["scikit"], 0.2, color="#E5C3D1", label="scikit")
    plt.bar(
        nb_clusters - 0.1,
        s_score["from_scratch"]["L1"],
        0.2,
        color="#A0DDFF",
        label="L1",
    )
    plt.bar(
        nb_clusters + 0.1,
        s_score["from_scratch"]["L2"],
        0.2,
        color="#DEF6CA",
        label="L2",
    )
    plt.bar(
        nb_clusters + 0.3,
        s_score["from_scratch"]["Haversine"],
        0.2,
        color="#FFB997",
        label="Haversine",
    )

    plt.xticks(nb_clusters)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score obtenu (0 à 1)")
    plt.title("Coefficient de silhouette en fonction du nombre de cluster")
    plt.legend()
    plt.show()

    plt.bar(nb_clusters - 0.3, ch_score["scikit"], 0.2, color="#E5C3D1", label="scikit")
    plt.bar(
        nb_clusters - 0.1,
        ch_score["from_scratch"]["L1"],
        0.2,
        color="#A0DDFF",
        label="L1",
    )
    plt.bar(
        nb_clusters + 0.1,
        ch_score["from_scratch"]["L2"],
        0.2,
        color="#DEF6CA",
        label="L2",
    )
    plt.bar(
        nb_clusters + 0.3,
        ch_score["from_scratch"]["Haversine"],
        0.2,
        color="#FFB997",
        label="Haversine",
    )

    plt.xticks(nb_clusters)
    plt.xticks(nb_clusters)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score obtenu (0 à ∞)")
    plt.title("Index de Calinski Harabasz en fonction du nombre de cluster")
    plt.legend()
    plt.show()

    plt.bar(nb_clusters - 0.3, bd_score["scikit"], 0.2, color="#E5C3D1", label="scikit")
    plt.bar(
        nb_clusters - 0.1,
        bd_score["from_scratch"]["L1"],
        0.2,
        color="#A0DDFF",
        label="L1",
    )
    plt.bar(
        nb_clusters + 0.1,
        bd_score["from_scratch"]["L2"],
        0.2,
        color="#DEF6CA",
        label="L2",
    )
    plt.bar(
        nb_clusters + 0.3,
        bd_score["from_scratch"]["Haversine"],
        0.2,
        color="#FFB997",
        label="Haversine",
    )

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

    methods = ["L1", "L2", "Haversine"]

    for name in methods:
        labels, centroids = p_kmscr.kmeans(
            X[:, 1], X[:, 0], 100, nb_clusters, func=name
        )

        s_score["from_scratch"][name].append(sk_m.silhouette_score(X, labels))
        ch_score["from_scratch"][name].append(sk_m.calinski_harabasz_score(X, labels))
        bd_score["from_scratch"][name].append(sk_m.davies_bouldin_score(X, labels))

        print(
            f"Résultat coefficient de silhouette avec n = {nb_clusters} et méthode {name} : ",
            s_score["from_scratch"][name][-1],
        )
        print(
            f"Résultat index de Calinski Harabasz avec n = {nb_clusters} et méthode {name} : ",
            ch_score["from_scratch"][name][-1],
        )
        print(
            f"Résultat index de Bouldin Davies avec n = {nb_clusters} et méthode {name} :",
            bd_score["from_scratch"][name][-1],
            "\n",
        )

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, RocCurveDisplay 

def evaluate_classifaction(knn_model, X_test, y_test, test_preds):
    """
    evaluate classification results using :

    • Accuracy
    • Confusion matrix
    • Precision-Recall
    • Compute Area Under the Receiver Operating Characteristic Curve

    """
    # Accuracy, Precision and recall
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, average='macro')
    recall = recall_score(y_test, test_preds, average='macro')
    print('Classification Accuracy: ', accuracy)
    print('Classification Precision: ', precision)
    print('Classification Recall: ', recall)

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # ROC
    y_scores = knn_model.predict(X_test)  

    # Calculer les taux de faux positifs et de vrais positifs
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Afficher la courbe ROC
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()
    plt.show()
