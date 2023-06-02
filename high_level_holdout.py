import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import repartition_données as rp_dn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

def plot_roc_curve(true_y, y_prob):
  
    fpr, tpr = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def high_level_hold_out():


def support_vector_machine():
    X_train, X_test, y_train, y_test = rp_dn.hold_out()
   
    smv = SVC()
    smv.fit(X_train,y_train)

    y_pred = smv.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average="weighted")
    recall = recall_score(y_test,y_pred,average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("SVM")
    print("Score de prédiction:", accuracy_svm)
    print("Précision:", precision_svm)
    print("Rappel:", recall_svm)
    print("F1-score:", f1_svm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat_svm)
    disp.plot()
    plt.show()

#support_vector_machine()

def rand_forest():
    X_train, X_test, y_train, y_test = rp_dn.hold_out()
    rf= RandomForestClassifier()
    rf.fit(X_train,y_train)

    best_rf.fit(X_train, y_train)
    y_pred_rf = best_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf, average="weighted")
    recall_rf = recall_score(y_test, y_pred_rf, average="weighted")
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
    cf_mat_rf = confusion_matrix(y_test, y_pred_rf)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred, average="weighted")
    recall = recall_score(y_test,y_pred,average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Random Forest")
    print("Score de prédiction:", accuracy_rf)
    print("Précision:", precision_rf)
    print("Rappel:", recall_rf)
    print("F1-score:", f1_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat_rf)
    disp.plot()
    plt.show()

    y_prob_rf = best_rf.predict(X_test)
    plot_roc_curve(y_test, y_prob_rf)
    plt.title('Courbe de ROC - Random Forest')
    plt.show()

#rand_forest()
def Multilayer():
     X_train, X_test, y_train, y_test = rp_dn.hold_out()
     mlp = MLPClassifier()
     mlp.fit(X_train,y_train)

     y_pred = mlp.predict(X_test)

     accuracy = accuracy_score(y_test,y_pred)
     precision = precision_score(y_test,y_pred, average="weighted")
     recall = recall_score(y_test,y_pred,average="weighted")
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