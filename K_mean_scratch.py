import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def haversine_distance(lat1, lon1, lat2, lon2):  
    #moyenne du rayon de la terre 
    R = 6371.0 

    #Convertir en radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    #Distance entre 2 longitude et 2 latitude
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    #Calcul entre deux points géographiques à partir de leur coordonnées de latitude et longitude
    #Source du calcul:  https://towardsdatascience.com/calculating-distance-between-two-geolocations-in-python-26ad3afe287b 
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def kmeans(latitude, longitude, n_clusters, max_iters=100):
    # Convertir les coordonnées en radians
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)

    data = np.column_stack((lat_rad, lon_rad))


    #Calcul du centroid

    #Cette ligne de code sélectionne de manière aléatoire n_clusters éléments à partir du tableau data 
    #et les assigne à la variable centroids pour être utilisés comme centres dans un algorithme de clustering.
    centroids = data[np.random.choice(range(len(data)), n_clusters, replace=False)]
    for _ in range(max_iters):

        #calcul des distances via la fonction haversine_distance où lat1 et lon1 sont des centroids
        #Ensuite on itère la paire lat et lon en fonction de data

        # np.array([1, 2, 3])
        # >>>array([1, 2, 3])
        distances = np.array([haversine_distance(lat, lon, centroids[:, 0], centroids[:, 1]) for lat, lon in data])

        # Trouver les indices des distances minimales dans chaque ligne du tableau "distances"
        labels = np.argmin(distances, axis=1)

        # Calcul les nouveaux centres avec la moyenne des points de données appartenant à chaque cluster
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Vérifier si les centres actuels sont égaux aux nouveaux centres
        if np.all(centroids == new_centroids):
            break

        # Mettre à jour les centres avec les nouveaux centres
        centroids = new_centroids

        # Retourner les étiquettes des clusters et les centres finaux
        return labels, centroids

def Kmeans_scratch(user_clusters):
    df= pd.read_csv("CSV_IA.csv", encoding="latin-1")

    #Avoir les valeurs de la dataframe sur une variable
    latitude = df['latitude'].values
    longitude = df['longitude'].values
    n_clusters = user_clusters
    
    labels, centroids = kmeans(latitude, longitude, n_clusters)

    #Faire apparaitre la carte
    plt.scatter(longitude, latitude, c=labels)

    #Faire apparaitre les centroides
    plt.scatter(centroids[:, 1], centroids[:, 0], c='red')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Clustering K-means')
    plt.legend()
    plt.show()

    data = pd.DataFrame({'latitude': latitude, 'longitude': longitude, 'labels': labels})
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", color="labels" )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
