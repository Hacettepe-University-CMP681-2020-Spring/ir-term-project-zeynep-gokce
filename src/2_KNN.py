from collections import Counter
import math
from scipy.spatial import distance
import os
import numpy as np
import time



def load_data(model):
  data = "../dataset/features/data_features/"+model
  data_dirs  = os.listdir(data)
  data_list = []
  data_names =[]
  data_ids=[]
  data_=[]
  for image_name in data_dirs:
    #print(image_name)
    #print(os.path.join(data, image_name))
    data_feature = np.load(os.path.join(data, image_name),"r")
    #print(data_feature.shape)
    data_id = int(int(image_name.replace(".npy",""))/100)
    data_list.append(data_feature.tolist())
    data_names.append(image_name)
    data_ids.append(data_id)
    data_.append((data_feature.tolist(),data_id, image_name))
  return data_


def load_query(model):
  query = "../dataset/features/query_features/"+model
  query_dirs  = os.listdir(query)
  query_dirs.sort()
  query_list = []
  query_names = []
  query_ids=[]
  query_=[]

  for image_name in query_dirs:
    #print(image_name)
    query_feature = np.load(os.path.join(query, image_name))
    #print(query_feature.shape)
    query_id = int(int(image_name.replace(".npy",""))/100)
    query_list.append((query_feature.tolist()))
    query_names.append(image_name)
    query_ids.append(query_id)
    query_.append((query_feature.tolist(),query_id,image_name))
                       
  return query_




def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    #data = data[0:5]

    
    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[0], query[0])
        # 3.2 Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index, example[2]))
    
    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices, key=lambda tup: tup[0])
    
    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][1] for distance, i, name in k_nearest_distances_and_indices]
    #print("k_nearest_labels : " ,k_nearest_labels)
    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

def mode(labels):
  return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2, _):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def manhattan_distance(point1, point2):
    
    data_list=[]
    query_list=[]

    data_list.append(list(point1))
    query_list.append(list(point2))
    dist = distance.cdist(data_list,query_list ,'cityblock').tolist()[0][0]
    return dist

def euclidean_distance(point1, point2):

    data_list=[]
    query_list=[]

    data_list.append(list(point1))
    query_list.append(list(point2))
    dist = distance.cdist(data_list,query_list ,'euclidean').tolist()[0][0]
    #print(dist)
    return dist



def result_image_retrieval(data, query, distance_function,num_top_item,model):
  #print(distance_function.__name__)
  result_file = open("../results/result_knn-"+model+"-"+str(distance_function.__name__)+"-top"+str(num_top_item)+".txt", "w")
  for q in query:
    clf_k_nearest_neighbors, clf_prediction = knn(data, q, k=num_top_item, distance_fn=distance_function, choice_fn=mode)
    
    line = q[2].replace(".npy",".jpg") +" "
    rank=0
    for retrieved_img in clf_k_nearest_neighbors:
      if retrieved_img[0]>0.0:
        line +=  str(rank) + " "+ str(retrieved_img[2]).replace(".npy",".jpg")+ " "
        rank+=1
        
    line += "\n"

    result_file.write(line)
    

def main():
    
    models =["vgg19", "resnet50" , "inceptionv3"]
    num_top_item = [1, 5, 10]

    for model in models:
      data = load_data(model)
      query= load_query(model)

      for top_k in num_top_item: 
        #print("model : ", model)
        #print("top k:", top_k) 
        start_time = time.time()

        result_image_retrieval(data, query, euclidean_distance, top_k, model)
        end_time = time.time()
        #print("images retrieved in seconds (euclidean) :", (end_time-start_time))
        result_image_retrieval(data, query, manhattan_distance,top_k,model)



if __name__ == '__main__':
  main()
