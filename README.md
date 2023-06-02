# ProjetCIR3_IA

### Description -------------

Projet réalisé dans le cadre du Projet BIG DATA de l'ISEN Yncrea Ouest.
Tristan Saëz - Adrien LeBoucher - Vincent Le Brenn

### -------------------------

### Comment utiliser ce projet ?

Il y a 2 possibilité, soit en lançant tout simplement le fichier main.pyavec python ou en lançant train_predict.py qui ne donne que les fichier nécessaire à l'execution des scripts bash.

Cependant, avant toutes choses, il faut installer les packets avec la commande suivante :

```bash
pip install -r requirements.txt
```

### Lancer un script bash :

Pour lancer un script bash, il suffit d'executer une des commandes suivantes en remplaçant les arguments entre [] par ceux voulus :

```bash
.\scripts.sh -m kmean [latitude] [longitude] [centroïdes]
.\scripts.sh -m knn [info accident] [CSV]
.\scripts.sh -m classification [info accident] [méthode]
```

Exemple : 

```bash
.\scripts.sh -m kmeans 5 9 ((1.3, 1.5),(4.3, 2.2))
```

> ⚠ Il est nécessaire d'avoir git (ou linux) pour executer un script bash


### Description des fichier :

1. csv_cleaned.csv : fichier initial séparé par des ","
2. CSV_IA.csv, CSV_IA_red.csv : fichiers csv créés après préparation des données et réduction de dimension
3. conversion_data_to_num.xlsx : fichier excel contenant la conversion données non-numériques <=> données numériques
4. commune_2009.csv : liste des communes de france en 2009
5. requirements.txt : fichier contenant les lib nécessaires à l'execution des programmes
6. fichiers .sav : fichiers binaires contenant les algorithmes entrainés (construits dans le même programme que les algorithmes)
7. scripts.sh : fichier bash permettant 

