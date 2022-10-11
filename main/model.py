# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 17:33:44 2022

@author: Kiran George
"""

# U-net model

# Making the imports
import tensorflow as tf

# Defining a conv block which can be called for all two conv layers

def conv_block(input_tensor, n_filters, kernel):
    for i in range(1):
        x = tf.keras.layers.Conv2D(n_filters, kernel_size = (kernel,kernel), activation = 'relu', padding="same")(input_tensor)
    return x
    
    
def encoder_block(num_filters, data):
    conv = conv_block(data, num_filters, 3)
    x = tf.keras.layers.MaxPool2D(pool_size = (2,2), strides=(2,2))(conv)
    out = tf.keras.layers.Dropout(0.2)(x)
    return conv, out
    
# Creating the encoder

def encoder(data):
    conv1,out1 = encoder_block(64, data)
    conv2,out2 = encoder_block(128, out1)
    conv3,out3 = encoder_block(256, out2)
    conv4,out4 = encoder_block(512, out3)
    return conv1, conv2, conv3, conv4, out4


def decoder_block(data, input_layer, n_filters):
    deconv_out = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size = (3,3), strides= (2,2), padding = 'same')(data)
    concat = tf.keras.layers.concatenate([ deconv_out, input_layer])
    conv_out = conv_block(concat, n_filters, 2)
    conv_out = tf.keras.layers.Dropout(0.3)(conv_out)
    
    return conv_out


def decoder(data, conv1, conv2, conv3, conv4):
    deconv_1 = decoder_block(data, conv4, 512)
    deconv_2 = decoder_block(deconv_1, conv3, 256)
    deconv_3 = decoder_block(deconv_2, conv2, 128)
    deconv_4 = decoder_block(deconv_3, conv1, 64)
    return deconv_4

def arrange_model(data):
    conv1, conv2, conv3, conv4, out4 = encoder(data)
    bottle_neck_out = conv_block(out4, 1024, 2)
    decoder_out = decoder(bottle_neck_out,conv1, conv2, conv3, conv4)
    output = tf.keras.layers.Conv2D(23, (1,1), activation = 'softmax')(decoder_out)
    return output
    
def unet():
    inputs = tf.keras.Input(shape = (128, 128, 3))
    outputs = arrange_model(inputs)
    unet = tf.keras.Model(inputs = inputs, outputs = outputs, name = "U-Net")
    return unet    


unet = unet()
unet.summary()
    
#dot_img_file = '/model.png'
#tf.keras.utils.plot_model(unet, to_file=dot_img_file, show_shapes=True)
    


    