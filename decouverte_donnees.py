import pandas as pd

df= pd.read_csv("csv_cleaned.csv", encoding="latin-1")

print("SIUUUU :",df['descr_grav'].unique())