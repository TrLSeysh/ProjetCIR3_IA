"""
Module pandas & numpy used for data formatting
"""
import pandas as pd
import numpy as np  

import decouverte_donnees as p_dd
import prep_data as p_pd
import reduc_dim as rd
import k_mean_scikit as p_kmsci
import K_mean_scratch as p_kmscr
import kNN_scikit as k_sci
import kNN_scratch as k_scr

df = pd.read_csv("csv_cleaned.csv", sep=",")
df.info()

df_prep = p_pd.clean_data(df)
df_prep.to_csv("CSV_IA.csv", index=False)

#p_kmscr.Kmeans_scratch()
p_kmsci.k_mean(df_prep)
p_kmscr.Kmeans_scratch()

# Reduction de la dimension
# rd.correlation(df_prep)
df_reduc = rd.reduc_dim_grav(df_prep)
df_reduc.to_csv("CSV_IA_red.csv", index=False)
# rd.correlation(df_reduc)

# Classification
k_sci.kNN_scikit(df_reduc)
k_scr.kNN_scratch(df_reduc)