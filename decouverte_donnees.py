import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def decouverte_donnees():
    df= pd.read_csv("stat_acc_V4.csv", encoding="latin-1")


    #Remplacement de la gravité de l'accident par une valeur en chiffre
    #c(Indemne = 1, Tué = 2, "Blessé hospitalisé" = 3, "Blessé léger" = 4)
    print("SIUUUU :",df['descr_grav'].unique())

    print("Nombre de colonnes:", len(df.columns))

    print("Nombre de lignes(taille de la base de données):", len(df))

    print("Nombre de valeurs dans la base de données(colonnes * lignes):", df.size)

    # Première page
    plt.figure(1)

    plt.hist(df['descr_grav'])
    plt.title("Gravité des accidents")

    #Deuxième
    plt.figure(2)

    plt.hist(df['descr_lum'])
    plt.title("Luminosité de l'accident")

    #Troisième
    plt.figure(3)

    plt.hist(df['descr_athmo'])
    plt.title("L'athmosphère de l'accident")

    #Quatrième
    plt.figure(4)

    plt.hist(df['descr_etat_surf'])
    plt.title("Etat de la route")

    # Affichage des figures
    plt.show()