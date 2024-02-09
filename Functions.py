### Importing the required Libraries

import numpy as np
import random
import math

### defining custom distance

def custom_distance(a, b):
    epsilon = 1e-10  # Small constant to avoid zero or negative values
    A=np.array(a)
    B=np.array(b)
    return np.sum((A - B) * np.power(np.log(A + epsilon) - np.log(B + epsilon), 2))

### Making sure coodinates are positive for using log

def translation_non_negatives(data,minimum_allowed=1):
    # Trouver la valeur minimale pour chaque dimension
    min_values = np.min(data, axis=0)

    # Calculer la constante de translation pour chaque dimension
    translation_constants = abs(np.minimum(0,min_values))+minimum_allowed

    # Ajouter la constante de translation à chaque coordonnée
    translated_data = data + translation_constants

    return translated_data, translation_constants

### initializing the membership matrix with random values

def initializeMembershipMatrix(nb_data_points, nb_clusters):
    membership_mat = list()
    for dpidx in range(nb_data_points): #dpidx=data_point_index
        random_num_list = [random.random() for i in range(nb_clusters)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return np.array(membership_mat) #list of lists, can be converted using np.array(membership_mat)

### calculating the cluster center, is done in every iteration

def calculateClusterCenter(df, membership_mat, nb_data_points, nb_clusters, fuzziness=2):
    #cluster_mem_val = zip(*membership_mat)
    cluster_mem_val = np.array(membership_mat).T
    cluster_centers = list()
    for j in range(nb_clusters):
        x = list(cluster_mem_val[j])
        xraised = [e ** fuzziness for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(nb_data_points):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        #numerator = map(sum, zip(*temp_num))
        #center = [z/denominator for z in numerator]
        #cluster_centers.append(center)
        numerator = np.sum(temp_num, axis=0)
        center = numerator / denominator
        cluster_centers.append(center.tolist())
    return cluster_centers

### updating the membership values using the cluster centers

def updateMembershipValue(df, membership_mat, cluster_centers,nb_data_points, nb_clusters, fuzziness=2):
    p = float(2/(fuzziness-1))
    for i in range(nb_data_points):
        x = list(df.iloc[i])
        distances = [custom_distance(x, cluster_centers[j]) for j in range(nb_clusters)]
        for j in range(nb_clusters):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(nb_clusters)])
            membership_mat[i][j] = float(1/den)       
    return membership_mat

### Function defined which returns the Clusters from the Membership Matrix

def getClusters(membership_mat, cluster_centers, nb_clusters):
    cluster_labels = list()
    for i in range(nb_clusters):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels

### the final fcm function, 

def fuzzyCMeansClustering(df, nb_data_points, nb_clusters, MAX_ITER, tol=1e-4):
    # Membership Matrix
    membership_mat = initializeMembershipMatrix(nb_data_points, nb_clusters)
    curr = 0
    while curr <= MAX_ITER:
        cluster_centers = calculateClusterCenter(df, membership_mat, nb_data_points, nb_clusters, fuzziness=2)
        new_membership_mat = updateMembershipValue(df, membership_mat, cluster_centers,nb_data_points, nb_clusters)
        cluster_labels = getClusters(new_membership_mat, cluster_centers, nb_clusters)
        curr += 1

        # Check convergence
        if np.linalg.norm(new_membership_mat - membership_mat) < tol:
            break
        membership_mat = new_membership_mat
    
    return cluster_labels, cluster_centers

