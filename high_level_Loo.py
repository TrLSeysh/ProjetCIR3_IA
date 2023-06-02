import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,roc_curve, confusion_matrix, ConfusionMatrixDisplay

def train_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return accuracy, precision, recall, f1


def leave_one_out(csv="CSV_IA_red.csv"):
    df = pd.read_csv(csv)
    df = df.dropna()
    df = df.head(1000)
    df = df.drop(["latitude", "longitude", "an_nais"], axis=1)  # Drop latitude, longitude, et an_nais 
    X = df.drop(["descr_grav"], axis=1).values
    y = df["descr_grav"].values

    loo = LeaveOneOut()
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    print("Leave one out sans GridSearchCV")
    return X_train_list, X_test_list, y_train_list, y_test_list


def high_level_cross_validation():
    X_train_list, X_test_list, y_train_list, y_test_list = leave_one_out()

    # Random Forest
    rf = RandomForestClassifier()
    rf_accuracy, rf_precision, rf_recall, rf_f1 = [], [], [], []
    for i in range(len(X_train_list)):
        accuracy, precision, recall, f1 = train_evaluate_classifier(rf, X_train_list[i], y_train_list[i],
                                                                    X_test_list[i], y_test_list[i])
        rf_accuracy.append(accuracy)
        rf_precision.append(precision)
        rf_recall.append(recall)
        rf_f1.append(f1)

    # SVM
    svm = SVC()
    svm_accuracy, svm_precision, svm_recall, svm_f1 = [], [], [], []
    for i in range(len(X_train_list)):
        accuracy, precision, recall, f1 = train_evaluate_classifier(svm, X_train_list[i], y_train_list[i],
                                                                    X_test_list[i], y_test_list[i])
        svm_accuracy.append(accuracy)
        svm_precision.append(precision)
        svm_recall.append(recall)
        svm_f1.append(f1)

    # MLP
    mlp = MLPClassifier(max_iter=1000)
    mlp_accuracy, mlp_precision, mlp_recall, mlp_f1 = [], [], [], []
    for i in range(len(X_train_list)):
        accuracy, precision, recall, f1 = train_evaluate_classifier(mlp, X_train_list[i], y_train_list[i],
                                                                    X_test_list[i], y_test_list[i])
        mlp_accuracy.append(accuracy)
        mlp_precision.append(precision)
        mlp_recall.append(recall)
        mlp_f1.append(f1)

    # Print the results
    print("Random Forest")
    print("Score de prédiction:", sum(rf_accuracy) / len(rf_accuracy))
    print("Précision:", sum(rf_precision) / len(rf_precision))
    print("Rappel:", sum(rf_recall) / len(rf_recall))
    print("F1-score:", sum(rf_f1) / len(rf_f1))

    print("SVM")
    print("Score de prédiction:", sum(svm_accuracy) / len(svm_accuracy))
    print("Précision:", sum(svm_precision) / len(svm_precision))
    print("Rappel:", sum(svm_recall) / len(svm_recall))
    print("F1-score:", sum(svm_f1) / len(svm_f1))

    print("Multi-layer Perceptron")
    print("Score de prédiction:", sum(mlp_accuracy) / len(mlp_accuracy))
    print("Précision:", sum(mlp_precision) / len(mlp_precision))
    print("Rappel:", sum(mlp_recall) / len(mlp_recall))
    print("F1-score:", sum(mlp_f1) / len(mlp_f1))


high_level_cross_validation()
