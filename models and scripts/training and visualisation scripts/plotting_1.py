# -*- coding: utf-8 -*-

from os.path import isfile, join
from os import listdir

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

import sys
sys.path.append('D:\MPhys DL Liquid Crystals\models')

from model_evaluation import generate_test_set, test_accuracy, pred_true_labels

def plot_val_hist(history_path, save_path, show = True):
    history_df = pd.read_csv(history_path)
    
    fig, (ax1, ax2) = plt.subplots(nrows = 2, figsize = (14, 9))
    fig.suptitle('Plot of Training and Validation Metrics Against Number of Completed Epochs')
    
    ax1.set_title('Training and Validation Accuracies vs Epoch Number')
    ax1.plot(history_df['epoch'], history_df['accuracy'], color = 'r')
    ax1.plot(history_df['epoch'], history_df['val_accuracy'], color = 'g')
    ax1.legend(labels = ['Training Accuracy', 'Validation Accuracy'])
    
    ax2.set_title('Training and Validation Loss vs Epoch Number')
    ax2.plot(history_df['epoch'], history_df['loss'], color = 'r')
    ax2.plot(history_df['epoch'], history_df['val_loss'], color = 'g')
    ax2.legend(labels = ['Training Loss', 'Validation Loss'])
    
    if not isfile(save_path):
        plt.savefig(save_path)
    
    if show == True:
        plt.show()


def plot_layers_acc(checkpoints_folder, test_path, save_path, title, show = True):
    if isfile(save_path) and show == False:
        return
    
    elif isfile(save_path) and show == True:
        image = mpimg.imread(save_path)
        plt.imshow(image)
        plt.show()
        return
    
    test_set = generate_test_set(test_path)
    
    accs = []
    for folder in listdir(checkpoints_folder):
        model_path = join(checkpoints_folder, folder)
        test_acc = test_accuracy(model_path, test_set = test_set)
        accs.append(test_acc)
        
    max_layers = len(accs)
    num_layers = []
    
    for i in range(1, max_layers + 1):
        num_layers.append(i)
        i += 1
        
    fig, ax = plt.subplots(figsize = (12, 8))
    fig.suptitle(title) 
    
    ax.set_xlabel('Number of Convolutional Layers')
    ax.set_ylabel('Test Accuracy')
    
    ax.plot(num_layers, accs, color = 'r')
    
    if not isfile(save_path):
        plt.savefig(save_path)
    
    if show == True:
        plt.show()
    
        
        
    
def plot_augs_layers_acc(checkpoints_folder, test_path, save_path, show = True):
    if isfile(save_path) and show == False:
        return
    
    elif isfile(save_path) and show == True:
        image = mpimg.imread(save_path)
        plt.imshow(image)
        plt.show()
        return
    
    no_augs_acc = []
    flip_augs_acc = []
    all_augs_acc = []
    num_layers = []
    
    test_set = generate_test_set(test_path)
    
    no_augs_folder = join(checkpoints_folder, 'no augs')
    flip_augs_folder = join(checkpoints_folder, 'flip augs')
    all_augs_folder = join(checkpoints_folder, 'all augs')
    
    for folder in listdir(no_augs_folder):
        model_path = join(no_augs_folder, folder)
        test_acc = test_accuracy(model_path, test_set = test_set)
        no_augs_acc.append(test_acc)
        
    for folder in listdir(flip_augs_folder):
        model_path = join(flip_augs_folder, folder)
        test_acc = test_accuracy(model_path, test_set = test_set)
        flip_augs_acc.append(test_acc)

    for folder in listdir(all_augs_folder):
        model_path = join(all_augs_folder, folder)
        test_acc = test_accuracy(model_path, test_set = test_set)
        all_augs_acc.append(test_acc)
        
    max_layers = max(len(no_augs_acc), len(flip_augs_acc), len(all_augs_acc))
    
    for i in range(1, max_layers + 1):
        num_layers.append(i)
        i += 1
        
    fig, ax = plt.subplots(figsize = (12, 8))
    fig.suptitle('Test Accuracies for Different Numbers of Layers and Amounts of Image Augmentations')
    
    ax.set_xlabel('Number of Convolutional Layers')
    ax.set_ylabel('Test Accuracy')
    
    ax.plot(num_layers, no_augs_acc, color = 'r')
    ax.plot(num_layers, flip_augs_acc, color = 'g')
    ax.plot(num_layers, all_augs_acc, color = 'b')
    
    ax.legend(labels = ['No Augmentations', 'Flip Augmentations', 'All Augmentations'])
    
    if not isfile(save_path):
        plt.savefig(save_path)
    
    if show == True:
        plt.show()

    
def plot_confusion_matrix(model_path, test_path, save_path, title, show = True):
    if isfile(save_path) and show == False:
        return
    
    elif isfile(save_path) and show == True:
        image = mpimg.imread(save_path)
        plt.imshow(image)
        plt.show()
        return
     
    true_labels, pred_labels, class_labels = pred_true_labels(model_path, test_path)    
    
    con_mat = confusion_matrix(true_labels, pred_labels, class_labels, None, 'all')
    
    num_classes = len(class_labels)
    
    fig, ax = plt.subplots(figsize = (num_classes, num_classes))
    fig.suptitle(title)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    ax.sns.heatmap(con_mat, annot = True, cmap = 'Blues')
    
    plt.savefig(save_path)
    
    if show == True:
        plt.show()
    
    

if __name__ == '__main__':
    confusion_matrix(model_path = 'D:/MPhys DL Liquid Crystals/models/checkpoints/4 phases v williams/low_cap',
                     test_path = 'D:/MPhys DL Liquid Crystals/data/4 phases/v williams data/preprocessed and split/test',
                     save_path = 'D:/MPhys DL Liquid Crystals/models/graphs/4 phases v williams/low_cap_confusion_matrix',
                     title = 'Low_Cap Confusion Matrix')

    
        
        
    
    
    