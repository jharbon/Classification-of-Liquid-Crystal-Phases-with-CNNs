# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

from PIL import Image

from os import walk

def generate_labels_preds(test_path, model_path, image_size = 256, test_accuracy = False):
    
    # Count number of images from every class folder and return total
    num_images = sum(len(files) for _, _, files in walk(test_path))
    
    # Generate batches of test images
    test_datagen = ImageDataGenerator(rescale=1.0/255) 
    test_gen = test_datagen.flow_from_directory(
        directory = test_path,
        target_size = (image_size, image_size),
        color_mode = 'grayscale',
        class_mode = 'categorical',
        batch_size = num_images,
        shuffle = False)
    
    # Create a batch which includes all images
    test_batch = test_gen.next()
    
    # Save numpy arrays of shape (image_size, image_size)
    x = test_batch[0]
    
    # Vector of vectors with the class confidence values for each image
    # There will be a definite class with a confidence of 1 for each image
    y = test_batch[1]
    
    labels = np.argmax(y, axis = 1)
    
    model = tf.keras.models.load_model(model_path)
    
    # Evaluate the overall test accuracy
    if test_accuracy:
        model.evaluate(
            x,
            y,
            batch_size = 1,
            steps=num_images,
            verbose=2)
    
        
    preds = np.argmax(model.predict(x), axis = 1) 
    
    return labels, preds

def swap_smFI_2class(labels):   
    # can be used to swap the places of SmF and SmI in the binary con mat
     for index, value in enumerate(labels):
            if value == 0:
                labels[index] = 1
                
            else:
                labels[index] = 0
                
     return labels           
                     
                
def swap_smFI_5class(labels):   
    # can be used to swap the places of SmF and SmI in the 5 class con mat
     for index, value in enumerate(labels):
            if value == 3:
                labels[index] = 4
                
            elif value == 4:
                labels[index] = 3
     
     return labels           


def plot_con_matrix(test_path, title, model_path, save_path, class_names):
    
    labels_preds = generate_labels_preds(test_path = test_path, model_path = model_path)
    
    labels = labels_preds[0]
    preds = labels_preds[1]
    
    con_mat = confusion_matrix(labels, preds) 
    con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    
    x = range(len(class_names))
    
    con_mat_df = pd.DataFrame(con_mat,
                              index = class_names,
                              columns = class_names)
    
    con_mat_df.round(decimals = 2)
    
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams["font.weight"] = 'bold'
    plt.rcParams["font.size"] = 30
    plt.rcParams['axes.titlesize'] = 40
    plt.rcParams['axes.titleweight'] = 'bold'
  
    plt.figure(figsize=(22, 16))
    plt.xticks(x, labels)
    plt.yticks(x, labels, rotation = 90, va = "center")
    if title is not None:
        plt.title(title)
    
    sns.heatmap(con_mat_df, annot=True, cmap = "Blues")
    plt.ylabel('True Class', labelpad = 20)
    plt.xlabel('Predicted Class', labelpad = 20)
    
    plt.savefig(save_path)
    plt.show()
    
    
def generate_multi_labels_preds(test_path, model_paths):
    labels_preds = []
    
    for model_path in model_paths:
        labels_preds.append((generate_labels_preds(test_path, model_path)))
    
    return labels_preds      

    
def plot_mean_confusion_matrix(labels_preds, class_names, save_path_mean, save_path_std,
                               reorder_smFI_2class = False, reorder_smFI_5class = False,
                               font_scale = 1.0, fig_size = None):
    matrix_dim = len(class_names)
    con_mats = np.empty((len(labels_preds), matrix_dim, matrix_dim))
    for index, label_pred in enumerate(labels_preds):
        labels = label_pred[0]
        preds = label_pred[1]
        
        if reorder_smFI_2class:
             labels = swap_smFI_2class(labels)
             preds = swap_smFI_2class(preds)
             
        if reorder_smFI_5class:
             labels = swap_smFI_5class(labels)
             preds = swap_smFI_5class(preds)     
             
        con_mat = confusion_matrix(labels, preds)
        con_mats[index] = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        
    con_mats_mean = np.mean(con_mats, axis=0)
    con_mats_mean_df = pd.DataFrame(con_mats_mean, index=class_names, columns=class_names)
    con_mats_mean_df = con_mats_mean_df.round(decimals = 2)
    
    con_mats_err = np.std(con_mats, axis=0)
    con_mats_err_df = pd.DataFrame(con_mats_err, index=class_names, columns=class_names)
    con_mats_err_df = con_mats_err_df.round(decimals = 2)
    
    if fig_size == None:
        figsize=(matrix_dim*2, matrix_dim)
    else:
        figsize = fig_size
    fig, ax = plt.subplots(figsize = (fig_size))
    #fig.suptitle(title, fontsize=16)
    
    sns.set(font_scale=font_scale)
    sns.heatmap(con_mats_mean_df, annot=True, cmap=plt.cm.Blues, cbar=False, square=True)
    
    ax.set_ylabel('True Phase')
    ax.set_xlabel('Predicted Phase')
    
    plt.tight_layout(w_pad=0.0, h_pad=1.5)
    
    plt.savefig(save_path_mean)
    plt.show()
    
    if fig_size == None:
        figsize=(matrix_dim*2, matrix_dim)
    else:
        figsize = fig_size
    fig, ax = plt.subplots(figsize = (fig_size))
    #fig.suptitle(title, fontsize=16)
    
    sns.set(font_scale=font_scale)
    sns.heatmap(con_mats_err_df, annot=True, cmap=plt.cm.BuPu, cbar=False, square=True)
    
    ax.set_ylabel('True Phase')
    ax.set_xlabel('Predicted Phase')
    
    plt.tight_layout(w_pad=0.0, h_pad=1.5)
    
    plt.savefig(save_path_std)
    plt.show()


    
    
    
    
    
    
    
    