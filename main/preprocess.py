# Contains functions that is used to import data, create input pipeline and preprocess the images


import os
import cv2
import tensorflow as tf
import sys
import random
import numpy as np

BATCH_SIZE = 8


# def dataset_import(path):
#    '''
#    This function can be used to import dataset images when a filepath to the folder containing images is given. 
#    The output will be a list.
#    '''
#    images = []
#    image_count = 0

#    for filenames in os.listdir(path):
#        image_path = os.path.join(path, filenames)
#        img = cv2.imread(image_path)
#        image_count = image_count + 1
#        img = tf.image.resize(img, [300,300])
#        images.append(img)
#        sys.stdout.write('\rFile read: %d' %image_count)
#        sys.stdout.flush()        
#    return images

# Default file path

def file_name_collector(path):
    """
    First the names of files are collected using this function.
    """
    full_file_names = []
    for filenames in os.listdir(path):
        full_file_names.append(filenames)
    return full_file_names


def file_names_to_import(full_file_names):
    """
    The names collected are split into train and test using this function.
    Later these names can be connected with path to retrieve the files.
    """
    full_file_names.sort()
    random.seed(230)
    random.shuffle(full_file_names)
    split = int(0.8 * len(full_file_names))
    train_file_names = full_file_names[:split]
    test_file_names = full_file_names[split:]
    return train_file_names, test_file_names


def decode_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [128, 128], method='nearest')
    return img

def decode_mask(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1)
    #img = tf.math.reduce_max(img,axis=-1,keepdims=True)
    img = tf.image.resize(img, [128, 128], method='nearest')
    return img


def dataset_import_image(required_filenames, path):
    """
    This function can be used to import dataset images when a filepath to the folder containing images is given. 
    The output will be a list.
    """
    images = []
    image_count = 0

    for filenames in required_filenames:
        image_path = os.path.join(path, filenames)
        img = decode_image(image_path)
        image_count = image_count + 1
        images.append(img)
        sys.stdout.write('\rFile read: %d' % image_count)
        sys.stdout.flush()
    return images

def dataset_import_mask(required_filenames, path):
    """
    This function can be used to import dataset images when a filepath to the folder containing images is given. 
    The output will be a list.
    """
    images = []
    image_count = 0

    for filenames in required_filenames:
        image_path = os.path.join(path, filenames)
        img = decode_mask(image_path)
        image_count = image_count + 1
        images.append(img)
        sys.stdout.write('\rFile read: %d' % image_count)
        sys.stdout.flush()
    return images


def create_dataset(image, mask):
    """
    Creates a tensorflow dataset object which can be used in models
    """
    BUFFER_SIZE = 100
    training_dataset = tf.data.Dataset.from_tensor_slices((image, mask))

    training_dataset = training_dataset.cache()
    #training_dataset = training_dataset.shuffle(BUFFER_SIZE)
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    return training_dataset
