import sys
import numpy as np
import pandas as pd

def get_cluster(lat, lon, centroids):
    """
    get nearest cluster from point

    """
    dist = 100000000000000
    cluster = 0

    for centroid in enumerate(centroids):
        dist_temp = np.sqrt((lat - lon) ** 2 + (centroid[1][0] - centroid[1][1]) ** 2)
        if dist_temp < dist:
            dist = dist_temp
            cluster = centroid[0]
    
    return cluster


MODE = sys.argv[1]

if MODE == "kmeans":
    try:
        LAT = float(sys.argv[2])
        LON = float(sys.argv[3])
        print(np.float_(sys.argv[4:]))
        centroids = np.array(np.float_(sys.argv[4:])).reshape(-1, 2)
    except IndexError:
        print("Arguments seem to be missing, try running command with args for LAT, LON & centroids\n")
        sys.exit(1)
    
    print(f"latitude : {LAT}\nlongitude : {LON}\ncentroids : {centroids}\n")
    cluster_id = get_cluster(LAT, LON, centroids)
    print("cluster : ", cluster_id)

    pd.DataFrame({"point":{"id":None, "lat":LAT, "lon":LON}, "cluster":{"id":cluster_id, "lat":centroids[cluster_id][0], "lon":centroids[cluster_id][1]}}).to_json("cluster_point.json")

elif MODE == "knn":
    try:
        info_acc = list(sys.argv[2:9])
        CSV = float(sys.argv[10])
    except IndexError:
        print("Arguments seem to be missing, try running command with args for info_acc & CSV\n")
        sys.exit(1)

    

elif MODE == "classification":
    try:
        info_acc = list(sys.argv[2:9])
        METHOD = float(sys.argv[10])
    except IndexError:
        print("Arguments seem to be missing, try running command with args for info_acc & METHOD\n")
        exit(1)
    
else:
    print(f"mode {MODE} non reconnu, les modes acceptés sont :\n1. kmeans\n2. knn\n3. classification\n")
    exit(1)

    