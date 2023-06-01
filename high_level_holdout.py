import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import repartition_données as rp_dn


def support_vector_machine():
    X_train, X_test, y_train, y_test = rp_dn.hold_out()

    smv = SVC()
    smv.fit(X_train, y_train)

    y_pred = smv.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Affichage des métriques
    print("SVM")
    print("Score de prédiction:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1)


# support_vector_machine()


def rand_forest():
    X_train, X_test, y_train, y_test = rp_dn.hold_out()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Random Forest")
    print("Score de prédiction:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1)


# rand_forest()
def Multilayer():
    X_train, X_test, y_train, y_test = rp_dn.hold_out()
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Multi-layer Perceptron")
    print("Score de prédiction:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1)


def trois_algo_holdout():
    support_vector_machine()
    rand_forest()
    Multilayer()


trois_algo_holdout()
