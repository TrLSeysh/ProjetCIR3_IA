import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_score

def hold_out():
    df=pd.read_csv("CSV_IA_red.csv", sep=",")
    df= df.dropna()
    X =  df.drop(["descr_grav"], axis=1)
    y = df["descr_grav"]
    X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=50)
    return X_train, X_test, y_train ,y_test

def leave_one_out():
    df = pd.read_csv("CSV_IA_red.csv")

    df = df.dropna()
    df = df.head(2000) 
    X = df.drop(["latitude","longitude","an_nais","descr_grav"], axis=1).values
    y = df["descr_grav"].values

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    print("done")
    return X_train, X_test, y_train, y_test

leave_one_out()