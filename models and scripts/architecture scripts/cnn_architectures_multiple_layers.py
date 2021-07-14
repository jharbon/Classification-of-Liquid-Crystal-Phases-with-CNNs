# -*- coding: utf-8 -*-

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


def cnn_1_layer(classes):
    model = Sequential([
    
        Conv2D(filters = 32,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same',
               input_shape = (256, 256, 1)
               ),
    
        BatchNormalization(),
        GlobalAveragePooling2D(),
    
        Flatten(),
    
        Dense(units = 256, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = 128, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = classes, activation = 'softmax')
    
        ])
    
    return model


def cnn_2_layers(classes):
    model = Sequential([
    
        Conv2D(filters = 32,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same',
               input_shape = (256, 256, 1)
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 64,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        GlobalAveragePooling2D(),
    
        Flatten(),
    
        Dense(units = 256, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = 128, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = classes, activation = 'softmax')
    
        ])
    
    return model


def cnn_3_layers(classes, start_filters = 32):
    model = Sequential([
    
        Conv2D(filters = start_filters,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same',
               input_shape = (256, 256, 1)
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = start_filters * 2,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = start_filters * 4,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        GlobalAveragePooling2D(),
    
        Flatten(),
    
        Dense(units = 256, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = 128, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = classes, activation = 'softmax')
    
        ])
    
    return model


def cnn_4_layers(classes):
    model = Sequential([
    
        Conv2D(filters = 32,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same',
               input_shape = (256, 256, 1)
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 64,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 128,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 256,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        GlobalAveragePooling2D(),
    
        Flatten(),
    
        Dense(units = 256, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = 128, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = classes, activation = 'softmax')
    
        ])
    
    return model

def cnn_5_layers(classes):
    model = Sequential([
    
        Conv2D(filters = 32,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same',
               input_shape = (256, 256, 1)
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 64,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 128,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 256,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 512,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        GlobalAveragePooling2D(),
    
        Flatten(),
    
        Dense(units = 256, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = 128, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = classes, activation = 'softmax')
    
        ])
    
    return model


def cnn_6_layers(classes):
    model = Sequential([
    
        Conv2D(filters = 32,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same',
               input_shape = (256, 256, 1)
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 64,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 128,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 256,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 512,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        MaxPooling2D(pool_size = 2, strides = 2),
    
        Conv2D(filters = 1024,
               kernel_size = (3, 3),
               activation = 'relu',
               padding = 'same'
               ),
    
        BatchNormalization(),
        GlobalAveragePooling2D(),
    
        Flatten(),
    
        Dense(units = 256, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = 128, activation = 'relu'),
        Dropout(0.5),
    
        Dense(units = classes, activation = 'softmax')
    
        ])
    
    return model
