"""
Module pandas & numpy used for data formatting
"""
import pandas as pd
import numpy as np  
import decouverte_donnees as p_dd

df = pd.read_csv("csv_cleaned.csv", sep=",")
df.info()

import prep_data as p_pd
df_prep = p_pd.clean_data(df)
df_prep.to_csv("CSV_IA.csv", index=False)

import reduc_dim as rd
# Reduction de la dimension
# rd.correlation(df_prep)
df_reduc = rd.reduc_dim_grav(df_prep)
df_reduc.to_csv("CSV_IA_red.csv", index=False)
# rd.correlation(df_reduc)

import k_mean_scikit as p_kmsci
import K_mean_scratch as p_kmscr
import metrics as mt

#p_kmscr.Kmeans_scratch()
#p_kmsci.display_kmean(df_prep, 21)
#p_kmscr.Kmeans_scratch()
# mt.evaluate_kmeans(df_reduc[:1000])

import repartition_données as rpd
# Répartition des données
X_train, X_test, y_train, y_test = rpd.hold_out()

import kNN_scikit as k_sci
import kNN_scratch as k_scr

# Classification KNN
test_preds, knn_model = k_sci.kNN_scikit(df_reduc, X_train, X_test, y_train, y_test)
predictions, knn_scratch = k_scr.test_kNN_scr(X_train, X_test, y_train)

mt.evaluate_classifaction(knn_model, X_test, y_test, test_preds)