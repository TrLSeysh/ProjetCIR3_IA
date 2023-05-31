import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score
import repartition_données as rp_dn


def high_level_Loo():
    X_train, X_test, y_train , y_test = rp_dn.Leave_one_out()
    smv = SVC()
    smv.fit(X_train,y_train)

    y_pred = smv.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test,y_pred,average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Affichage des métriques
    print("SVM")
    print("Score de prédiction:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1)

    rf= RandomForestClassifier()
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test,y_pred,average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Random Forest")
    print("Score de prédiction:", accuracy)
    print("Précision:", precision,)
    print("Rappel:", recall)
    print("F1-score:", f1)

    mlp = MLPClassifier()
    mlp.fit(X_train,y_train)

    y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average="weighted",zero_division=1)
    recall = recall_score(y_test,y_pred,average="weighted",zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Multi-layer Perceptron")
    print("Score de prédiction:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1) 



high_level_Loo()    



