
# Content Based Image Retrieval System for Inria Holidays Dataset
This is explanation of the image retrieval system for all sample images on [Inria Holidays Dataset](http://lear.inrialpes.fr/~jegou/data.php).

## Dataset Preparation

The dataset has 500 different groups.The first images for each group are used as query image, others are image database.
For this reason, two different folders (query_images and data_images) are created under ./dataset/images/ folder.

## Feature Extraction

All images from the given in [Inria Holidays Dataset](http://lear.inrialpes.fr/~jegou/data.php)  are extracted using  1_feature_extraction.py script under ./src folder. This script is used for extraction three different deep learning models which are VGG19, Inception V3 and Resnet50 models. It reads the RGB images from the ./dataset/images folder, saves the features to the ./dataset/features/ folder.



## Nearest Neighbor Approach

Using features for each image, the different k numbers are used in the script ./2_KNN.py to obtaine top k nearest (related) images from the database. It saves the results under ./results folder.



## K-Means
To evaluate the k-means approach, the 4_k-means.py is used for the dataset. The result files are saved to the ./results folder.



## Evaluation
Two different evaluation metric is presented in this github folder.

### Ranking Average Precision Score for Nearest Neigbor Approach

In order to evalaute the nearest model with ranking average score, 3_ranking_average_score.py script is used under ./src folder.

### mAP Evaluation 

To obtain the mAP evalutaion metric for each results files which is presented in the dataset,  ./eval_holidays/run_map_metric.py. script is used. It utilizes holidays_map.py script to apply the mAP evaluation metric on each result files.
