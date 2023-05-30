"""
Module pandas & numpy used for data formatting
"""
import pandas as pd
import numpy as np

import decouverte_donnees as p_dd
import prep_data as p_pd

df = pd.read_csv("csv_cleaned.csv", sep=",")

p_pd.clean_data(df)
