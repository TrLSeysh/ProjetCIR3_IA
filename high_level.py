import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import repartition_données as rp_dn

def support_vector_machine():
    X_train, X_test, y_train, test = rp_dn.hold_out()
