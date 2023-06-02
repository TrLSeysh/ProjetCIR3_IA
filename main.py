"""
Module pandas & numpy used for data formatting
"""
import pandas as pd
import decouverte_donnees as p_dd
import kNN_scikit as k_sci
import kNN_scratch as k_scr
import k_mean_scikit as p_kmsci
import K_mean_scratch as p_kmscr
import metrics as mt
import reduc_dim as rd
import prep_data as p_pd
import repartition_données as rpd

#Découverte données
#p_dd.decouverte_donnees()

df = pd.read_csv("csv_cleaned.csv", sep=",")

df_prep = p_pd.clean_data(df)
df_prep.to_csv("CSV_IA.csv", index=False)


# Reduction de la dimension
# rd.correlation(df_prep)
df_reduc = rd.reduc_dim_grav(df_prep)
df_reduc.to_csv("CSV_IA_red.csv", index=False)
# rd.correlation(df_reduc)


# Calcul du kmeans avec 2 méthodes
p_kmscr.Kmeans_scratch(12)
p_kmsci.display_kmean(df_prep, 12)

# Evalue les méthodes kmeans
mt.evaluate_kmeans(df_reduc[:1000])


# Répartition des données
X_train, X_test, y_train, y_test = rpd.hold_out(df_reduc)


# Classification KNN
test_preds, knn_model = k_sci.kNN_scikit(df_reduc, X_train, X_test, y_train, y_test)
predictions, knn_scratch = k_scr.test_kNN_scr(X_train, X_test, y_train)

#Evaluation KNN
mt.evaluate_classifaction(knn_model, X_test, y_test, test_preds)
