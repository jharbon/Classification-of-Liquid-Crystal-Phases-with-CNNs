# -*- coding: utf-8 -*-

# example of creating a CNN with an efficient inception module
from keras.models import Model, Input
from keras.layers import Conv2D, Dropout, Dense
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out


# downscaled inception architecture which only uses the first three modules
def inception_3mods(input_shape, classes):
    visible = Input(input_shape)
    
    X = Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, activation = 'relu', padding = 'same')(visible)
    
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    X = Conv2D(filters = 192, kernel_size = (3, 3), strides = 1, activation = 'relu', padding = 'same')(X)
    X = Conv2D(filters = 192, kernel_size = (3, 3), strides = 1, activation = 'relu', padding = 'same')(X)
    
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception (3a)
    X = inception_module(X, 64, 96, 128, 16, 32, 32)
    X = inception_module(X, 64, 96, 128, 16, 32, 32)
    
    # Inception (3b)
    X = inception_module(X, 128, 128, 192, 32, 96, 64)
    X = inception_module(X, 128, 128, 192, 32, 96, 64)
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception (4a)
    X = inception_module(X, 192, 96, 208, 16, 48, 64)
    X = inception_module(X, 192, 96, 208, 16, 48, 64)
    
    X = GlobalAveragePooling2D()(X)
    
    X = Dense(units = 512, activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    
    X = Dense(units = classes, activation = 'softmax')(X)
    
    return Model(inputs = visible, outputs = X)


# Full Inception architecture with a slightly higher dropout rate
def inception(input_shape, classes):
    visible = Input(input_shape)
    
    X = Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, activation = 'relu', padding = 'same')(visible)
    
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    X = Conv2D(filters = 192, kernel_size = (3, 3), strides = 1, activation = 'relu', padding = 'same')(X)
    X = Conv2D(filters = 192, kernel_size = (3, 3), strides = 1, activation = 'relu', padding = 'same')(X)
    
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception (3a)
    X = inception_module(X, 64, 96, 128, 16, 32, 32)
    X = inception_module(X, 64, 96, 128, 16, 32, 32)
    
    # Inception (3b)
    X = inception_module(X, 128, 128, 192, 32, 96, 64)
    X = inception_module(X, 128, 128, 192, 32, 96, 64)
    
    # Pooling
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception (4a)
    X = inception_module(X, 192, 96, 208, 16, 48, 64)
    X = inception_module(X, 192, 96, 208, 16, 48, 64)
    
    # Inception (4b)
    X = inception_module(X, 160, 112, 224, 24, 64, 64)
    X = inception_module(X, 160, 112, 224, 24, 64, 64)
    
    # Inception (4c)
    X = inception_module(X, 128, 128, 256, 24, 64, 64)
    X = inception_module(X, 128, 128, 256, 24, 64, 64)
    
    # Inception (4d)
    X = inception_module(X, 112, 144, 288, 32, 64, 64)
    X = inception_module(X, 112, 144, 288, 32, 64, 64)
    
    # Inception (4e)
    X = inception_module(X, 256, 160, 320, 32, 128, 128)
    X = inception_module(X, 256, 160, 320, 32, 128, 128)
    
    # Pooling
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception (5a)
    X = inception_module(X, 256, 160, 320, 32, 128, 128)
    X = inception_module(X, 256, 160, 320, 32, 128, 128)
    
    # Inception (5b)
    X = inception_module(X, 384, 192, 384, 48, 128, 128)
    X = inception_module(X, 384, 192, 384, 48, 128, 128)
    
    X = GlobalAveragePooling2D()(X)
    
    X = Dense(units = 1000, activation = 'relu')(X)
    X = Dropout(0.5)(X)
    
    
    X = Dense(units = classes, activation = 'softmax')(X)
    
    return Model(inputs = visible, outputs = X)










