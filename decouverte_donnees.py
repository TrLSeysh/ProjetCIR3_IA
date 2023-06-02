import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def decouverte_donnees(csv ="stat_acc_V4.csv"):
    df= pd.read_csv(csv, encoding="latin-1")


    print("Valeur cible :",df['descr_grav'].unique())


    print("Features:", len(df.columns)-1)

    print("Instance:", len(df.columns))

    print("Nombre de lignes(instace par classe):", len(df))

    print("Nombre de valeurs dans la base de données(colonnes * lignes):", df.size)

    # Première page
    plt.figure(1)

    plt.hist(df["descr_grav"])
    plt.title("Gravité des accidents")

    # Deuxième
    plt.figure(2)

    plt.hist(df["descr_lum"])
    plt.title("Luminosité de l'accident")

    # Troisième
    plt.figure(3)

    plt.hist(df["descr_athmo"])
    plt.title("L'athmosphère de l'accident")

    # Quatrième
    plt.figure(4)

    plt.hist(df["descr_etat_surf"])
    plt.title("Etat de la route")

    # Affichage des figures
    plt.show()
