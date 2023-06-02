import pandas as pd
import numpy as np
from collections import Counter

import repartition_données as rpd

def euclidean_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)):
        if isinstance(x1[i], (float, int)) and isinstance(x2[i], (float, int)):
            distance += (float(x1[i]) - float(x2[i])) ** 2
    return np.sqrt(distance)

class kNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Distances entre x et tous les exemples d'entraînement
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Trie des distances et obtention des indices k voisins les plus proches
        k_indices = np.argsort(distances)[:self.k]

        # Etiquettes des k voisins les plus proches
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Etiquette la plus fréquente
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def test_kNN_scr(X_train, X_test, y_train):
    knn_scratch = kNN(k=3)
    knn_scratch.fit(X_train, y_train)
    predictions = knn_scratch.predict(X_test)
    print("Predictions:", predictions)
    np.set_printoptions(suppress=True)
    print(X_train.values[0])    
    print(X_test.values[0])
    return predictions, knn_scratch

