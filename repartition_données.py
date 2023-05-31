import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_score

def repart_donnees():
    df=pd.read_csv("CSV_IA_red.csv")

    df= df.dropna()


    X =  df.drop(["descr_grav"], axis=1)
    y = df["descr_grav"]

    X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.3, random_state=4)

    model= RandomForestClassifier(random_state= 42)
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test,y_pred, average="weighted")

    print("Hold out")
    print("Score de prédiction:", accuracy)
    print("Précision:", precision)
    print("Rappel:", recall)
    print("F1-score:", f1)

    X = X[:1000]
    y = y[:1000]

    X= X.values
    y= y.values

    performance_score = 0.0

#boucle Leave Out one
    for i in range(len(X)):
     X_train = np.delete(X, i, axis=0)
     Y_train = np.delete(y, i, axis=0)
    
        # Entraîner la forêt aléatoire
     model = RandomForestClassifier()
     model.fit(X_train, Y_train)
    
     X_test = X[i].reshape(1, -1)
     y_pred = model.predict(X_test)
    
    # Comparer la prédiction avec la vraie valeur de la variable cible
     y_true = np.array((y[i],))
    
    # Mettre à jour le score de performance
     performance_score += accuracy_score(y_true, y_pred)


# Calculer la moyenne du score de performance
    average_performance_score = performance_score / len(X)
    print("Leave One Out")
    print("Score de performance moyen(accuracy_score) :", average_performance_score)