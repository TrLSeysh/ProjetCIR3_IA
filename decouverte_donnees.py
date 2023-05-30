import pandas as pd
import plotly.express as px


df= pd.read_csv("stat_acc_V4.csv", encoding="latin-1")


#Remplacement de la gravité de l'accident par une valeur en chiffre
#c(Indemne = 1, Tué = 2, "Blessé hospitalisé" = 3, "Blessé léger" = 4)
print("SIUUUU :",df['descr_grav'].unique())

print("Nombre de colonnes:", len(df.columns))

print("Nombre de lignes(taille de la base de données):", len(df))

print("Nombre de valeurs dans la base de données(colonnes * lignes):", df.size)


fig_grav = px.histogram(df['descr_grav'])
fig_grav.show()

fig_lum = px.histogram(df['descr_lum'])
fig_lum.show()

fig_athmo= px.histogram(df['descr_athmo'])
fig_athmo.show()

fig_surface = px.histogram(df['descr_etat_surf'])
fig_surface.show()

fig_vehicule = px.histogram(df['num_veh'])
fig_vehicule.show()