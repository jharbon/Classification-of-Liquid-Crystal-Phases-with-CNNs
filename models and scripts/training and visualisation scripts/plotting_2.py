# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

import numpy as np

from os.path import isfile

def test_accs_models(test_accs, test_error_bars, number_of_models, model_names, save_path, y_scale = (0, 100)):
    x = range(1, number_of_models + 1)
    
    plt.rcParams["font.weight"] = 'bold'
    plt.rcParams["font.size"] = 45
    plt.rcParams['axes.titlesize'] = 60
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(figsize = (45, 45))
    
    plt.title('Mean Test Accuracy Values for Models (tuned) on Smectic 3 Phases')
    ax.set_xlabel("Model", fontsize = 50)
    ax.set_ylabel("Mean Test Accuracy", fontsize = 50)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation = 90)
    ax.set_ylim(y_scale)
    
    ax.errorbar(x, y = test_accs, yerr = test_error_bars, ecolor = "b", capsize = 5,
                color = "r", marker = ".", markersize = 14, linewidth = 3)
    
    if not isfile(save_path):
        plt.savefig(save_path)
    
    
#test_accs = [74, 58, 73]  
#test_error_bars = [7.8, 5.8, 4.7]  
#save_path = 'D:/MPhys DL Liquid Crystals/models/graphs/semester 2/smectic 3 phases/mean test accuracies (tuned)'

#test_accs_models(test_accs, test_error_bars, number_of_models = 3,
 #                model_names = ['3 Layers', 'Inception_3M', 'ResNet50'], save_path = save_path)


def testaccs_vs_lr(test_accs_1, test_error_bars_1, test_accs_2, test_error_bars_2, 
                   lr_values, save_path, y_scale = (50, 100)):
    
    plt.rcParams["font.weight"] = 'bold'
    plt.rcParams["font.size"] = 45
    plt.rcParams['axes.titlesize'] = 60
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(figsize = (45, 45))
    
    plt.title('3 Layers, Start Filters = 64, Batch Size = 16, Mean Test vs Validation Accuracy')
    ax.set_xlabel("Learning Rate", fontsize = 50)
    ax.set_ylabel("Mean Accuracy", fontsize = 50)
    ax.set_ylim(y_scale)
    
    ax.errorbar(lr_values, y = test_accs_1, yerr = test_error_bars_1, ecolor = "r", capsize = 20,
                color = "r", marker = ".", markersize = 40, linewidth = 4)
    ax.errorbar(lr_values, y = test_accs_2, yerr = test_error_bars_2, ecolor = "b", capsize = 20,
                color = "b", marker = ".", markersize = 40, linewidth = 4)
    
    ax.legend(labels = ["Validation", "Test"])
    
    if not isfile(save_path):
        plt.savefig(save_path)
        

#test_accs_1 = [64, 66, 65, 65]
#test_error_bars_1 = [5.7, 6.2, 5.3, 3.9]
#test_accs_2 = [75, 82, 85, 81]
#test_error_bars_2 = [9.0, 4.6, 3.8, 9.4]
#lr_values = [0.0005, 0.0001, 0.00005, 0.00001]        
#testaccs_vs_lr(test_accs_1, test_error_bars_1, test_accs_2, test_error_bars_2,
 #              lr_values,
  #              save_path = 'D:/MPhys DL Liquid Crystals/models/graphs/semester 2/cholesteric and 4 smectics/3 layers, sf=64, bs=16')        

def bar_plot_times(no_aug_times, flip_aug_times, all_aug_times, save_path):
    models = ["1 Layer", "2 Layers", "3 Layers", "4 Layers", "5 Layers", "6 Layers", "ResNet50", "Inception"]
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    width = 0.25
    width_array = np.array([width, width, width, width, width, width, width, width])
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(figsize = (18, 16))
    
    ax.set_xlabel("Model", fontsize = 22)
    ax.set_ylabel("Training Time (minutes)", fontsize = 22)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation = 90)
    
    ax.bar(x - width_array, height = no_aug_times, color = "b", width = width)
    ax.bar(x, height = flip_aug_times, color = "r", width = width)
    ax.bar(x + width_array, height = all_aug_times, color = "g", width = width)
    ax.legend(labels = ['No Augmentations', 'Flip Augmentations', 'All Augmentations'])
    
    if not isfile(save_path):
        plt.savefig(save_path)
    
    

def bar_plot_dataset(ch_num, fluid_nums, hex_nums, save_path):
    phases = ["Ch", "SmA", "SmC", "SmI", "SmF"]
    x = np.array([1, 2, 3, 4, 5])
    
    #plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 15
    #plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.linewidth'] = 1
    fig, ax = plt.subplots(figsize = (8, 8))
    
    ax.set_xlabel("Phase", fontsize = 15)
    ax.set_ylabel("Number of Images", fontsize = 15)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    
    ax.bar(x, height = ch_num, color = "r")
    ax.bar(x, height = fluid_nums, color = "b")
    ax.bar(x, height = hex_nums, color = "g")
    ax.legend(labels = ["Cholesteric", "Fluid Smectic", "Hexatic Smectic"], fontsize = 12)
    
    if not isfile(save_path):
        plt.savefig(save_path)
    
    plt.show()

bar_plot_dataset(ch_num = [1646, 0, 0, 0, 0],
                 fluid_nums = [0, 1106, 2000, 0, 0],
                 hex_nums = [0, 0, 0, 762, 630],
                 save_path = 'D:/MPhys DL Liquid Crystals/models/graphs/semester 2/overall dataset')        
        
def class_size_comparison(class_sizes_1, class_sizes_2, class_names, dataset_versions, save_path):
    x = np.array(range(1, len(class_names) + 1))
    width = 0.3
    width_array = np.array([width] * len(class_names))
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 40
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(figsize = (30, 30))
    
    ax.set_ylabel("Number of Images", fontsize = 45)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation = 90)
    
    ax.bar(x - width_array/2, height = class_sizes_1, color = "r", width = width)
    ax.bar(x + width_array/2, height = class_sizes_2, color = "b", width = width)
    ax.legend(labels = dataset_versions)
    
    if not isfile(save_path):
        plt.savefig(save_path)        
        

#class_size_comparison(class_sizes_1 = [2520, 666, 840], class_sizes_2 = [3204, 2226, 840],
 #                     class_names = ['fluid', 'hexatic', 'SC'],
  #                    dataset_versions = ['Semester 1', 'Semester 2'],
   #                   save_path = 'D://MPhys DL Liquid Crystals/models/graphs/semester 2/smectic 3 phases/class size comparison')


def test_accs_augs_models(no_augs, no_augs_error, flip_augs, flip_augs_error,
                          all_augs, all_augs_error, save_path, y_scale = (0, 100)):
    models = ["1 Layer", "2 Layers", "3 Layers", "4 Layers", "5 Layers", "6 Layers", "ResNet50", "Inception"]
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.linewidth'] = 2
    fig, ax = plt.subplots(figsize = (18, 16))
    
    ax.set_xlabel("Model", fontsize = 22)
    ax.set_ylabel("Mean Test Accuracy", fontsize = 22)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation = 90)
    ax.set_ylim(y_scale)
    ax.legend(labels = ["No Augmentations", "Flip Augmentations", "All Augmentations"])
    
    ax.errorbar(x, y = no_augs, yerr = no_augs_error, ecolor = "b", capsize = 8,
                color = "b", marker = ".", markersize = 15, linewidth = 3)
    ax.errorbar(x, y = flip_augs, yerr = flip_augs_error, ecolor = "r", capsize = 8,
                color = "r", marker = ".", markersize = 15, linewidth = 3)
    ax.errorbar(x, y = all_augs, yerr = all_augs_error, ecolor = "g", capsize = 8,
                color = "g", marker = ".", markersize = 15, linewidth = 3)
    ax.legend(labels = ["No Augmentations", "Flip Augmentations", "All Augmentations"], loc = "lower right")
    
    if not isfile(save_path):
        plt.savefig(save_path)
        
        
"""test_accs_augs_models(no_augs = [80, 77, 85, 85, 83, 81, 80, 79], no_augs_error = [0, 0, 0, 0, 0, 0, 0, 0],
                      flip_augs = [84, 83, 89, 84, 88, 83, 81, 81], flip_augs_error = [0, 0, 0, 0, 0, 0, 0, 0],
                      all_augs = [81, 80, 80, 73, 83, 83, 80, 71], all_augs_error = [0, 0, 0, 0, 0, 0, 0, 0],
                      save_path = "D:/MPhys DL Liquid Crystals/models/graphs/4 phases/test accuracy and augmentations")"""



def plot_best_models(val_accs, val_errs, test_accs, test_errs, save_path,
                     lower_bound=50, upper_bound=100, linestyle='none'):
    plt.rcParams['axes.titley'] = 1.05
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    
    labels = ['Seq', 'Inc', 'RN50']
    
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(lower_bound, upper_bound)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Highest Mean Accuracy/%", labelpad = -6)
    
    trans1 = ax.transData + ScaledTranslation(-4/72, 0, fig.dpi_scale_trans)
    trans2 = ax.transData + ScaledTranslation(+4/72, 0, fig.dpi_scale_trans)
    ax.errorbar(labels, 
                 val_accs, 
                 yerr=val_errs, 
                 marker='o', 
                 capsize = 3,
                 linestyle=linestyle, 
                 transform=trans1)
    ax.errorbar(labels, 
                 test_accs, 
                 yerr=test_errs, 
                 marker='s', 
                 capsize = 3,
                 linestyle=linestyle, 
                 transform=trans2)
    ax.legend(['Validation', 'Test'], loc='lower left')
    
    plt.savefig(save_path)
    plt.show()     
    
#val_accs = [93, 95, 91]
#val_errs = [1, 2, 2]
#test_accs = [96, 98, 93]
#test_errs = [3, 2, 3]
    
#plot_best_models(val_accs, val_errs, test_accs, test_errs,
 #                save_path = "D:/MPhys DL Liquid Crystals/models/graphs/semester 2/cholesteric and smectic/ChSm best models")