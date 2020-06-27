from scipy.misc import imresize
from keras.applications import vgg19, inception_v3, resnet50
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os

DATASET_DIR ="../dataset"
image_dir = os.path.join(DATASET_DIR, "images")



def image_batch_generator(image_names, batch_size):
    num_batches = len(image_names) // batch_size
    
    for i in range(num_batches):
        batch = image_names[i * batch_size : (i + 1) * batch_size]
        yield batch
    batch = image_names[(i+1) * batch_size:]
    yield batch
    
def vectorize_images(image_dir, image_size, preprocessor, 
                     model, feature_dir, batch_size=10):
    image_names = os.listdir(image_dir)
    num_vecs = 0
    vecs = []
    for image_batch in image_batch_generator(image_names, batch_size):
        batched_images = []
        for image_name in image_batch:
            image = plt.imread(os.path.join(image_dir, image_name))
            image = imresize(image, (image_size, image_size))
            batched_images.append(image)
        X = preprocessor(np.array(batched_images, dtype="float32"))
        vectors = model.predict(X)
       
        
        for i in range(vectors.shape[0]):
            if len(batched_images) == vectors.shape[0]:
                file_name = feature_dir+image_batch[i].replace(".jpg","")
                np.save(file_name,vectors[i])
            num_vecs += 1
    
    print("number of vectors : ", num_vecs)




data = ["data","query"]

### VGG19 Feture Extraction
IMAGE_SIZE = 224
vgg19_model = vgg19.VGG19(weights="imagenet", include_top=True)
# vgg19_model.summary()
model = Model(input=vgg19_model.input,
             output=vgg19_model.get_layer("fc2").output)
preprocessor = vgg19.preprocess_input

for d in data:
    feature_dir = "../dataset/features/"+d +"_"+"features/vgg19/"
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    vectorize_images(image_dir+"/"+d+"_images", IMAGE_SIZE, preprocessor, model, feature_dir)


### ResNet50 Feture Extraction
IMAGE_SIZE = 224
resnet_model = resnet50.ResNet50(weights="imagenet", include_top=True)
print(resnet_model.summary())

model = Model(input=resnet_model.input,
             output=resnet_model.get_layer("avg_pool").output)
preprocessor = resnet50.preprocess_input


for d in data:
    feature_dir = "../dataset/features/"+d +"_"+"features/resnet50/"
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    vectorize_images(image_dir+"/"+d+"_images", IMAGE_SIZE, preprocessor, model, feature_dir)


#### Inception V3 Feature Extraction
IMAGE_SIZE = 299
inception_model = inception_v3.InceptionV3(weights="imagenet", include_top=True)
print(inception_model.summary())
model = Model(input=inception_model.input,
             output=inception_model.get_layer("avg_pool").output)
preprocessor = inception_v3.preprocess_input


for d in data:
    feature_dir = "../dataset/features/"+d +"_"+"features/inceptionv3/"
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)
    vectorize_images(image_dir+"/"+d+"_images", IMAGE_SIZE, preprocessor, model, feature_dir)


