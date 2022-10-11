# Aerial image semantic segmentation using drone image dataset

# making all the imports

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle


# Importing custom defined functions
from plotting_utilities import *
from preprocess import *
from model import unet
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Variables
EPOCHS = 500

# import the label file

classes = pd.read_csv("D:/Machine_learning/projects/Aerial image segmentation/data/class_dict_seg.csv", index_col = 'name')
classes['color']= classes.values.tolist()
classes = classes.drop(classes.iloc[:,0:3], axis=1)

#print(classes.loc['pool',:])

# Collecting all the images
# Collect all the file names to be imported
ipath = "D:/Machine_learning/projects/Aerial image segmentation/data/dataset/semantic_drone_dataset/original_images"
all_file_names = file_name_collector(ipath)
# Split the file names to train and split
train_list, test_list = file_names_to_import(all_file_names)
# Import the train and test images
train_image = dataset_import_image(train_list, path=ipath)
print('\n completed importing train images')
test_image = dataset_import_image(test_list, path=ipath)
print('\n completed importing test images')



# Collecting all the masks
# Collect all the file names to be imported
mpath = 'D:/Machine_learning/projects/Aerial image segmentation/data/dataset/semantic_drone_dataset/label_images_semantic'
all_file_names = file_name_collector(mpath)
# Split the file names to train and split
mask_train_list, mask_test_list = file_names_to_import(all_file_names)
# Import the train and test images
train_mask = dataset_import_mask(mask_train_list, path=mpath)
print('\n completed importing train masks')
test_mask = dataset_import_mask(mask_test_list, path=mpath)
print('\n completed importing test masks')

train_mask = to_categorical(train_mask, num_classes = 23)
test_mask = to_categorical(test_mask, num_classes = 23)
# Now convert the colors to label


# Currently, we have :
# Train image, Train mask
# Test image, Test mask

# preprocess the images and masks
training_dataset = create_dataset(train_image, train_mask)
# test_image, test_mask = process_image_and_mask(test_image, test_mask)

testing_dataset = create_dataset(test_image, test_mask)


# Verifying all shapes

print(f'Image train set size is: ', training_dataset)
print("done")
# Print a sample image

for element in training_dataset:
    image = element[0][1]
    plot_single_image(image)
    print(image.shape)
    break

# Plotting an image and it's corresponding mask

# for item in training_dataset:
#     img = item[0][0]
#     mask = item[1][0]
#     #print(img)
#     plot_sample_image_mask(img, mask)
#     break


# Get the unet model

# Clearing some ram
#import gc 
#gc.collect()

unet.summary()

unet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = unet.fit(training_dataset, validation_data = testing_dataset, epochs =EPOCHS, steps_per_epoch = 10, validation_steps = 10)
#history = unet.fit(training_dataset, validation_data = testing_dataset, EPOCHS =200)

# Plotting the learning curves

plot_learning_curves(history.history["accuracy"], history.history["loss"], history.history["val_accuracy"], history.history["val_loss"], range(EPOCHS))


# Save model

#unet.save('/unet.h5')
saved_model = pickle.dumps(unet)
