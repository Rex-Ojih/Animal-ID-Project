import tensorflow as tf
from tensorflow import keras
import numpy as np #used for array opperations. 
import matplotlib.pyplot as plt #used to show image.
import os #used to itterate through directories and join paths.
import cv2 #used for image operations (Preprocessing).
import random #used for randomizing order of data.
import pickle as pk #used to write processed images to a binary file.
from tqdm import tqdm

Img_Size = 100 #size of immage.
DataDir = "D:/Pictures/Animals_10" #string value containing path to dataset directory on machiine. Example: "C:/Datasets/Images".
Classifications = ["Butterfly", "Cat", "Chicken", "Cow", "Dog", "Elephant",
                   "Horse", "Sheep", "Spider", "Squirrel"] #array of strings carrying image classifications, must be names of classification directories. 
training_data = [] #array for training data. 

def create_training_data():
    for classification in tqdm(Classifications): #'classification' is itterator for 'Classifications'.
        path = os.path.join(DataDir, classification) #gets path to classification dir. 
        class_num = Classifications.index(classification) #converts classification into a number for model.
        for img in os.listdir(path): #itterates through each image in classification directory.
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED) #converts images to pixel array. Second param of 'imread'specifies how to take in the image (COLOR or GRAYSCALE). Uses BGR
                new_img_array = cv2.resize(img_array, (Img_Size, Img_Size)) #resized images to 'Img_Size' by pixels, does not crop image (less detail or more detail).
                new_img_array = cv2.cvtColor(new_img_array, cv2.COLOR_BGR2RGB) #converts image from BGR to RGB.
#                plt.imshow(new_img_array, cmap="gray")  #if you want to show each image, uncomment this line and next line. 
#                plt.show()
                training_data.append([new_img_array, class_num]) #append images (pixel arrays) and their corresponding lables to 'training_data' array.
                
            except Exception as e: #if an image is broken, instead of ending program it moves on
                pass

create_training_data() #calling function defined above.
random.shuffle(training_data) #randomizes order of 'training_data' array.
            
#packing images (pixel arrays) and their corresponding labels into variables to be passed into the model.
X = [] #features set (images).
y = [] #labels. 

for features, label in tqdm(training_data): #'features' itterates through images, 'label' itterates through labels
    X.append(features)
    y.append(label)

#convertr image (pixel array) into numpy array.
#first param defines the number of features (images) to reshape into numpy array. -1 means all. 
#the next two params are the expected image (pixel array) sizes for hight and width.
#the last param is for the number of color channels.(1 - greyscale, 3 - RGB or BGR).
X = np.array(X).reshape(-1, Img_Size, Img_Size, 3) 

#now we want to binary write processed images (pixel arrays) stored in 'X' to a file for training the model multiple times after 
#tweaks are made to the model. We will no longer need to process the images (pixel arrays) each time we want to train the tweaked model.
pickle_out = open("X-features.pk", "wb")
pk.dump(X, pickle_out) #binary write to filename.pix, if file does not exist, it will be created.
pickle_out.close()

pickle_out = open("y-labels.pk", "wb")
pk.dump(y, pickle_out) #binary write to filename.pix, if file does not exist, it will be created.
pickle_out.close() 

#use varname = pk.load(open('filename.pix, 'rb')) to open file when you want to access processed images.
#the images (pixel arrays) will be loaded into 'varname' the exact way they were stored. Array 'varname' will be an exact copy of array 'X'.
