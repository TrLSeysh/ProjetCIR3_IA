"""
functions used to divide data for training and test base

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut


def hold_out(csv="CSV_IA_red.csv"):
    df=pd.read_csv(csv)
    df= df.dropna()
    X = df.drop(["descr_grav"], axis=1)
    y = df["descr_grav"]
    X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=50)
    print("Hold out")
    return X_train, X_test, y_train ,y_test

def leave_one_out(csv ="CSV_IA_red.csv"):
    df = pd.read_csv(csv)
    df = df.dropna()
    df = df.sample(n=1000)
    X = df.drop(["latitude","longitude","an_nais","descr_grav"], axis=1).values
    y = df["descr_grav"].values

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    print("Leave One Out")
    return X_train, X_test, y_train, y_test