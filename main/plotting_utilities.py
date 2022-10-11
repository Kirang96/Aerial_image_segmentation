
# Plotting utilities contain functions that can be used to plot images, masks and metrics.


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plot_single_image(image_file):
    """
    This function can be used to plot a sample image from the dataset
    """

    plt.figure(figsize=(10,10))
    plt.imshow(image_file)
    plt.title('Sample image')


def process_image(image):
    """
    This function can process the image by casting it to float 32 and converting it to numpy array.
    """
    image = tf.cast(image, tf.float32)
    image = np.array(image/255.0)
    return image


def plot_sample_image_mask(image, mask):
    """
    utility function to plot the image and mask
    """
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image)
    ax2.imshow(mask)
    ax1.set_title('image mask')
    ax2.set_title('sample mask')


def process_and_plot(image_dataset, mask_dataset, num):
    """
    This can be used to easily plot the image and mask by just giving a number
    """
    sample_image = process_image(image_dataset[num])
    sample_mask = process_image(mask_dataset[num])
    plot_sample_image_mask(sample_image, sample_mask)


def plot_learning_curves(acc,loss,val_acc,val_loss,epochs_range):
  '''
  function to plot learning curves
  '''
  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()


