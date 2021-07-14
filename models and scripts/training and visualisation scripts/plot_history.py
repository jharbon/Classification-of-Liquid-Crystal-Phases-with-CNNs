# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

def plot_hist(open_history_path, save_path):
    
    history_df = pd.read_csv(open_history_path)
    
    fig, (ax1, ax2) = plt.subplots(nrows = 2, figsize = (14, 9))
    fig.suptitle('Plot of Training and Test Metrics Against Number of Completed Epochs')
    
    ax1.set_title('Training and Validation Accuracies vs Epoch Number')
    ax1.plot(history_df['epoch'], history_df['accuracy'], color = 'r')
    ax1.plot(history_df['epoch'], history_df['val_accuracy'], color = 'g')
    ax1.legend(labels = ['Training Accuracy', 'Validation Accuracy'])
    
    ax2.set_title('Training and Validation Loss vs Epoch Number')
    ax2.plot(history_df['epoch'], history_df['loss'], color = 'r')
    ax2.plot(history_df['epoch'], history_df['val_loss'], color = 'g')
    ax2.legend(labels = ['Training Loss', 'Validation Loss'])
    
    plt.savefig(save_path)
    
    plt.show()
    
    
    