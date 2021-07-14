# -*- coding: utf-8 -*-

#import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

import numpy as np

from os import listdir
from os.path import join


def generate_test_set(test_path):
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    files_list = [file for folder in listdir(test_path) for file in listdir(join(test_path, folder))]
    num_images = len(files_list)

    test_set = test_datagen.flow_from_directory(
        directory = test_path,
        target_size = (256, 256),
        batch_size = num_images,
        class_mode = "categorical",
        color_mode = "grayscale")
    
    return test_set

'''Defined a function for evaluating a model on test set data.

 Can enter a path for test data and create test set inside function,
 or can create a test set outside the function if the function will 
 be used a lot in the same script/notebook and you don't want to wait
 for a new test set to be created every time you call the function.'''
 
def test_accuracy(model_path, test_path = None, test_set = None, test_loss = False):
    if test_path is not None:
        test_set = generate_test_set(test_path)
        
    elif test_set is not None:
        test_set = test_set
    
    test_batch = test_set.next()    
    x = test_batch[0]
    y = test_batch[1]
    
    model = load_model(model_path)
    
    loss_acc = model.evaluate(x, y, batch_size = 1)
    
    if test_loss == True:
        return loss_acc
    
    else:
        return loss_acc[1]
    
# define a function which can take a test set and compute predicted class labels for images
# also access the true class labels for the images 
# return a tuple containing predicted and true labels    
def pred_true_labels(model_path, test_path):
    test_set = generate_test_set(test_path)
    
    model = load_model(model_path)
    
    # use model to make predictions for the probability of each class for each image
    preds = model.predict_generator(test_set)
    # take the highest probability class for each image as the predicted class label for that image
    pred_labels = np.argmax(preds, axis = 1)
    
    
    true_labels = test_set.classes
    # define a list containing the unique class names
    class_labels = list(test_set.class_indices.keys())
    
    return (true_labels, pred_labels, class_labels)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    