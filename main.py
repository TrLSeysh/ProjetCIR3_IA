"""
Module pandas & numpy used for data formatting
"""
import pandas as pd
import numpy as np  

import decouverte_donnees as p_dd
import prep_data as p_pd
import k_mean_scikit as p_kms

df = pd.read_csv("csv_cleaned.csv", sep=",")
df.info()

df_prep = p_pd.clean_data(df)
df_prep.to_csv("CSV_IA.csv", index=False)

p_kms.k_mean(df_prep)
