import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df=pd.read_csv("CSV_IA.csv")

print(df["descr_grav"])
"""
X =  df.drop("descr_grav", axis=1)
y = df["descr_grav"]



X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.3, random_state=4)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)
"""