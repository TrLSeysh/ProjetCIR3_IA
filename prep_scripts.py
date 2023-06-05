import pandas as pd
import reduc_dim as rd
import prep_data as p_pd
import high_level_Loo as p_hll

df = pd.read_csv("csv_cleaned.csv", sep=",")

df_prep = p_pd.clean_data(df)
df_prep.to_csv("CSV_IA.csv", index=False)

# Reduction de la dimension
df_reduc = rd.reduc_dim_grav(df_prep)
df_reduc.to_csv("CSV_IA_red.csv", index=False)

p_hll.high_level_cross_validation()
