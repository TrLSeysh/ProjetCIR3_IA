import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import repartition_données as rp_dn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
#Fonction qui permet de tracer la courbe dde roc
def plot_roc_curve(true_y, y_prob):
    fpr, tpr = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

#La fonction hold out et affiche les matrices de confusion, courbe de ROC et les  metriques
def high_level_hold_out():
    X_train, X_test, y_train, y_test = rp_dn.hold_out()

    #Exemple d'hyperparametres, les memes que sur les sites de scikit learn
    #SVM - Support Vector machine
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm = SVC()
    #Le gridsearch les meilleurs parametres pour la classification SVM meme procédès pour les autres Random forest, MLP et Vote par majorité
    svm_grid = GridSearchCV(svm, svm_params, scoring='accuracy')
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_

    print("Meilleurs paramètres pour SVM:")
    print(svm_grid.best_params_)

    best_svm.fit(X_train, y_train)
    #Calcul des metriques et c'est le meme procédèes pour les autres comme Random Forest , MLP et Vote par majorité
    y_pred_svm = best_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm, average="weighted")
    recall_svm = recall_score(y_test, y_pred_svm, average="weighted")
    f1_svm = f1_score(y_test, y_pred_svm, average="weighted")
    cf_mat_svm = confusion_matrix(y_test, y_pred_svm)

    print("SVM")
    print("Score de prédiction:", accuracy_svm)
    print("Précision:", precision_svm)
    print("Rappel:", recall_svm)
    print("F1-score:", f1_svm)
    #Construction de la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat_svm)
    disp.plot()
    plt.show()
    #Utilisation de la fonction ROC en fonction de predict et son affichage
    # Pour optimiser la courbe de ROC on peut utiliser prob_predict 
    y_prob_svm = best_svm.predict(X_test)
    plot_roc_curve(y_test, y_prob_svm)
    plt.title('Courbe de ROC - SVM')
    plt.show()

    #Exemple d'hyperparametres, les memes que sur les sites de scikit learn
    #RF - Random Forest
    rf_params = {'n_estimators': [100, 200, 500], 'max_depth': [None, 5, 10]}
    rf = RandomForestClassifier()
    rf_grid = GridSearchCV(rf, rf_params, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    #On affiche les meilleurs parametres
    print("Meilleurs paramètres pour Random Forest:")
    print(rf_grid.best_params_)

    best_rf.fit(X_train, y_train)
    y_pred_rf = best_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf, average="weighted")
    recall_rf = recall_score(y_test, y_pred_rf, average="weighted")
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
    cf_mat_rf = confusion_matrix(y_test, y_pred_rf)

    #Affichages des metriques
    print("Random Forest")
    print("Score de prédiction:", accuracy_rf)
    print("Précision:", precision_rf)
    print("Rappel:", recall_rf)
    print("F1-score:", f1_rf)
    print("Matrice de confusion:")

    #Affichages des matrices de confusions
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat_rf)
    disp.plot()
    plt.show()

    #Affichages des ROC
    y_prob_rf = best_rf.predict(X_test)
    plot_roc_curve(y_test, y_prob_rf)
    plt.title('Courbe de ROC - Random Forest')
    plt.show()

    #Cas des hyperparametres pour le MLP basé sur le site scikit learn
    #MLP - Multi layer perceptron
    mlp_params = {'hidden_layer_sizes': [(50,), (100,), (200,)], 'activation': ['relu', 'tanh']}
    mlp = MLPClassifier(max_iter=1000)
    mlp_grid = GridSearchCV(mlp, mlp_params, scoring='accuracy')
    mlp_grid.fit(X_train, y_train)
    best_mlp = mlp_grid.best_estimator_

    print("Meilleurs paramètres pour MLP:")
    print(mlp_grid.best_params_)

    best_mlp.fit(X_train, y_train)
    y_pred_mlp = best_mlp.predict(X_test)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    precision_mlp = precision_score(y_test, y_pred_mlp, average="weighted")
    recall_mlp = recall_score(y_test, y_pred_mlp, average="weighted")
    f1_mlp = f1_score(y_test, y_pred_mlp, average="weighted")
    cf_mat_mlp = confusion_matrix(y_test, y_pred_mlp)

    print("Multi-layer Perceptron")
    print("Score de prédiction:", accuracy_mlp)
    print("Précision:", precision_mlp)
    print("Rappel:", recall_mlp)
    print("F1-score:", f1_mlp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat_mlp)
    disp.plot()
    plt.show()

    y_prob_mlp = best_mlp.predict(X_test)
    plot_roc_curve(y_test, y_prob_mlp)
    plt.title('Courbe de ROC - MLP')
    plt.show()
    
    #Algorithme qui permet de prendre la majorité
    classifiers = [('SVM', best_svm), ('Random Forest', best_rf), ('MLP', best_mlp)]
    voting_clf = VotingClassifier(classifiers, voting='hard')
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    precision_voting = precision_score(y_test, y_pred_voting, average="weighted")
    recall_voting = recall_score(y_test, y_pred_voting, average="weighted")
    f1_voting = f1_score(y_test, y_pred_voting, average="weighted")
    cf_mat_voting = confusion_matrix(y_test, y_pred_voting)

    #Affichages metriques
    print("Vote du Classifier par majorité")
    print("Score de prédiction:", accuracy_voting)
    print("Précision:", precision_voting)
    print("Rappel:", recall_voting)
    print("F1-score:", f1_voting)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mat_voting)
    disp.plot()
    plt.show()

    #Affichage des courbes de roc
    y_prob_voting = voting_clf.predict(X_test)
    plot_roc_curve(y_test, y_prob_voting)
    plt.title('Courbe de ROC - Majorité')
    plt.show()

