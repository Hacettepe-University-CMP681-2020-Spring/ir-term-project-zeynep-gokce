import numpy as np
import cv2
from sklearn.metrics import label_ranking_average_precision_score
import time
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
scores = []



def load_data(model):
  data = "../dataset/features/data_features/"+model
  data_dirs  = os.listdir(data)
  data_feature = []
  data_labels=[]
  data_name = []

  for image_name in data_dirs:
    #print(image_name)
    #print(os.path.join(data, image_name))
    feature = np.load(os.path.join(data, image_name),"r")
    #print(feature.shape)
    data_id = int(int(image_name.replace(".npy",""))/100)
    data_feature.append(feature.tolist())
    data_labels.append(data_id)
    data_name.append(image_name.replace(".npy",""))

  data_feature = np.array(data_feature)
  data_labels = np.array(data_labels)
  data_name = np.array(data_name)
  return data_feature,data_labels,data_name

def load_query(model):
  query = "../dataset/features/query_features/"+model
  query_dirs  = os.listdir(query)
  query_feature = []
  query_labels=[]
  query_name = []

  for image_name in query_dirs:
    
    feature = np.load(os.path.join(query, image_name))
    query_id = int(int(image_name.replace(".npy",""))/100)
    query_feature.append(feature.tolist())
    query_labels.append(query_id)
    query_name.append(image_name.replace(".npy",""))

  query_feature = np.array(query_feature)
  query_labels = np.array(query_labels)
  query_name = np.array(query_name)
  return query_feature,query_labels,query_name



def retrieve_closest_elements(test_code, test_label, learned_codes,model):

    data_feature,data_labels,data_name = load_data(model)
    query_feature,query_labels,query_name = load_query(model)

    distances = []
    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(data_labels).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]


    sorted_distances = sorted_distance_with_labels[:,0][-1] - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    return sorted_distances, sorted_labels, sorted_indexes 


def compute_average_precision_score(test_codes, test_labels, learned_codes, n_samples,model):
    out_labels = []
    out_distances = []
    retrieved_elements_indexes = []
    for i in range(len(test_codes)):
        #print(i)
        sorted_distances, sorted_labels, sorted_indexes = retrieve_closest_elements(test_codes[i], test_labels[i], learned_codes, model)
        out_distances.append(sorted_distances[:n_samples])
        out_labels.append(sorted_labels[:n_samples])
        retrieved_elements_indexes.append(sorted_indexes[:n_samples])

    out_labels = np.array(out_labels)
    out_labels_file_name = '../result_images/out_labels_{}'.format(n_samples)
    np.save(out_labels_file_name, out_labels)

    out_distances_file_name = '../result_images/out_distances_{}'.format(n_samples)
    out_distances = np.array(out_distances)
    np.save(out_distances_file_name, out_distances)
    score = label_ranking_average_precision_score(out_labels, out_distances)
    scores.append(score)
    return score


def retrieve_closest_images(test_element, test_label,test_name, model,n_samples=10):
    data_feature,data_labels,data_name = load_data(model)
    query_feature,query_labels,query_name = load_query(model)

    distances = []

    for code in data_feature:
        distance = np.linalg.norm(code - test_element)
        distances.append(distance)
    nb_elements = data_feature.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(data_labels).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = sorted_distance_with_labels[:,0][-1] - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    kept_indexes = sorted_indexes[:n_samples]

    score = label_ranking_average_precision_score(np.array([sorted_labels[:n_samples]]), np.array([sorted_distances[:n_samples]]))

    print("Average precision ranking score for tested element is {}".format(score))
   
    original_image = cv2.imread("../dataset/images/query_images/"+test_name+".jpg")
    dsize = (112,112)
    original_image = cv2.resize(original_image, dsize)
    cv2.imshow('original_image', original_image)

    #cv2.waitKey(0)
    print(kept_indexes)
    print(int(kept_indexes[0]))
    retrieved_images = cv2.imread("../dataset/images/data_images/"+ data_name[int(kept_indexes[0])]+".jpg") 
    retrieved_images = cv2.resize(retrieved_images, dsize)
    print(retrieved_images)
    for i in range(1, n_samples):
        print(i)
        print(retrieved_images.shape)
        ret_image = cv2.imread("../dataset/images/data_images/"+data_name[int(kept_indexes[i])]+".jpg") 
        ret_image = cv2.resize(ret_image, dsize)
        retrieved_images = np.hstack((retrieved_images,ret_image))
    cv2.imshow('Results', retrieved_images)


    cv2.imwrite('../result_images/original_image.jpg',original_image )
    cv2.imwrite('../result_images/retrieved_results.jpg',retrieved_images )


def test_model(n_test_samples, n_train_samples,model):

    data_feature,data_labels,data_name = load_data(model)
    query_feature,query_labels,query_name = load_query(model)

    indexes = np.arange(len(query_labels))
    np.random.shuffle(indexes)
    indexes = indexes[:n_test_samples]

    #print('Start computing score for {} train samples'.format(n_train_samples))
    t1 = time.time()
    score = compute_average_precision_score(query_feature[indexes], query_labels[indexes], data_feature, n_train_samples,model)
    t2 = time.time()
    #print('Score computed in: ', t2-t1)
    #print('Model score:', score)
    return score


# To test the whole model
n_test_samples = 500

n_train_samples = [5,50,100,150,250,500,750,1000,1400]


models = ["resnet50"]
for model in models:
    
    total_score_list =[]
    for n_train_sample in n_train_samples:
        total_score_list.append(test_model(n_test_samples, n_train_sample,model))

    np.save('../result_images/scores', np.array(scores))
    

    #total_score_list =[0.948,0.832, 0.815,0.811,0.802,0.787,0.781,0.780,0.780]
    data_feature,data_labels,data_name = load_data(model)
    query_feature,query_labels,query_name = load_query(model)

    # To retrieve closest image
    retrieve_closest_images(query_feature[0], query_labels[0],query_name[0],model)






plt.figure(figsize=(15, 8))
plt.plot(n_train_samples, total_score_list, marker='o')
plt.title("Model scoring results")
plt.xlabel("Number of retrieved images assessed", fontsize=16)
plt.ylabel("Score", fontsize=16)
plt.xticks(n_train_samples, rotation='vertical')
plt.show();
