# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from os.path import join

from model_evaluation import generate_test_set, test_accuracy

# Compute mean and sample standard deviation for validation accuracy values
# files_path specifies the directory containing the csv files
# file_names is used to specify which csv files to use and should contain strings
def valacc_mean_std(files_path, file_names):
    
    # List for storing the accuracy values
    val_accs = []
    
    for file in file_names:
        train_log = pd.read_csv(join(files_path, file))
        
        # Find row which corresponds to lowest validation loss
        best_row = train_log[train_log.val_loss == train_log.val_loss.min()]
        
        val_accs.append(best_row["val_accuracy"])
        
    val_accs = np.array(val_accs)
    
    # Compute mean and std for val_accs 
    val_mean = np.mean(val_accs)
    val_std = np.std(val_accs, ddof = 1)
    
    exit_message = ("Validation Accuracy Mean: {val_mean} \n"
    + "Validation Accuracy STD: {val_std}").format(val_mean = val_mean, val_std = val_std)
    
    print(exit_message)
    
def testacc_mean_std(models_path, model_names, test_path):
    
    test_accs = []
    
    t_set = generate_test_set(test_path)
    
    for model_name in model_names:
        test_acc = test_accuracy(model_path = join(models_path, model_name), test_set = t_set)
        
        test_accs.append(test_acc)
        
    test_accs = np.array(test_accs)
    
    # Compute mean and std for test_accs 
    test_mean = np.mean(test_accs)
    test_std = np.std(test_accs, ddof = 1)
    
    exit_message = ("Test Accuracy Mean: {test_mean} \n"
    + "Test Accuracy STD: {test_std}").format(test_mean = test_mean, test_std = test_std)
    
    print(exit_message)    
        
        
        
    
    
    
    
    
    
    

    