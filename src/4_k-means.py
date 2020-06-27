#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools
from sklearn.cluster import KMeans
import os
import math




def load_data(model):
  data = "../dataset/features/data_features/"+model
  data_dirs  = os.listdir(data)
  data_list = []
  data_names =[]
  data_ids=[]

  for image_name in data_dirs:
    #print(image_name)
    #print(os.path.join(data, image_name))
    data_feature = np.load(os.path.join(data, image_name),"r")
    #print(data_feature.shape)
    data_id = int(int(image_name.replace(".npy",""))/100)
    data_list.append(data_feature.tolist())
    data_names.append(image_name)
    data_ids.append(data_id)
    
  return data_names,data_list,data_ids




def load_query(model):
  query ="../dataset/features/query_features/"+model
  query_dirs  = os.listdir(query)
  query_dirs.sort()
  query_list = []
  query_names = []
  query_ids=[]
  #print(query_dirs)

  for image_name in query_dirs:
    #print(image_name)
    query_feature = np.load(os.path.join(query, image_name))
    #print(query_feature.shape)
    query_id = int(int(image_name.replace(".npy",""))/100)
    query_list.append((query_feature.tolist()))
    query_names.append(image_name)
    query_ids.append(query_id)
                       
  return query_names, query_list,query_ids



import math
import time
def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)



def manhattan_distance(point1, point2):
    data_list.append(list(point1))
    query_list.append(list(point2))
    dist = distance.cdist(data_list,query_list ,'cityblock').tolist()[0][0]
    return dist

# function returns WSS score for k values from 1 to kmax
def train_Kmeans(data, kmax,data_names, query_names, query,  model):
   
    kmeans = KMeans( n_clusters=k).fit(data)
    centroids = kmeans.cluster_centers_
   
    pred_clusters_data = kmeans.predict(data)
    pred_clusters_query = kmeans.predict(query)
  
    #start_time = time.time()
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(pred_clusters_query)):
    
        cluster_data_items_indices = [ind for ind, e in enumerate(pred_clusters_data) if e == pred_clusters_query[i]]
        #print("len(cluster_data_items_indices):",len(cluster_data_items_indices))
        curr_sse = 0
        query_dist_list = []
        for index in cluster_data_items_indices:
            data_vector = data[index]
            data_name = data_names[index]
            query_vector = query[i]
            query_name=query_names[i]
            dist = euclidean_distance(data_vector, query_vector)
            
            query_dist_list.append((dist, index, data_name))
        sorted_query_dist_list = sorted(query_dist_list, key=lambda tup: tup[0])
        
        
        #Save the result file
        #num_top_item = [1, 5, 10]
        num_top_item = [10]
        for topk in num_top_item:
            #print("topk :", topk)
            result_file = open("../results/result_kmeans-"+model+"-topk_"+str(topk)+"-cluster_"+str(kmax)+".txt", "a+")
            k_nearest_distances_and_indices = []
            #print("len(sorted_query_dist_list):",len(sorted_query_dist_list))
            if len(sorted_query_dist_list)>=topk:
                k_nearest_distances_and_indices = sorted_query_dist_list[:topk]
            else: 
                k_nearest_distances_and_indices = sorted_query_dist_list
        
            line = query_name.replace(".npy",".jpg") +" "
            rank=0
            for retrieved_img in k_nearest_distances_and_indices:
            
              if retrieved_img[0]>0.0:
                line +=  str(rank) + " "+ str(retrieved_img[2]).replace(".npy",".jpg")+ " "
                rank+=1

            line += "\n"
            print("line :",line)
            result_file.write(line)
            


    #end_time = time.time()
    #print("images retrieved in seconds (euclidean) :", (end_time-start_time))


clusters = [5,10,50,75,100,150,200,250,300,350,400,450,500]
#models =["vgg19", "resnet50" , "inceptionv3"]
models =[ "resnet50" ]

for model in models:
    
    data_name,data, data_ids = load_data(model)
    query_name, query,query_ids = load_query(model)
    for k in clusters :
                print("cluster :",k)
                s =train_Kmeans(data,k,data_name,query_name, query, model)






