"""
Module pandas & numpy used for data formatting
"""
import pandas as pd
import numpy as np

import decouverte_donnees as p_dd
import prep_data as p_pd
import reduc_dim as rd

df = pd.read_csv("csv_cleaned.csv", sep=",")

df_prep = p_pd.clean_data(df)
df_prep.to_csv("CSV_IA.csv", index=False)
rd.correlation(df_prep)
