# -*- coding: utf-8 -*-

from os.path import isfile, join
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# function for preparing the training and validation sets
def train_val_sets(train_path, valid_path, batch_size, input_size = (256, 256)):
  train_datagen = ImageDataGenerator(rescale = 1./255,
                                     horizontal_flip = True,
                                     vertical_flip = True)
  
  train_set = train_datagen.flow_from_directory(
      directory = train_path,
      target_size = input_size,
      batch_size = batch_size,
      class_mode = "categorical",
      shuffle = True,
      color_mode = "grayscale")

  val_datagen = ImageDataGenerator(rescale = 1./255)

  val_set = val_datagen.flow_from_directory(
      directory = valid_path,
      target_size = input_size,
      batch_size = batch_size,
      class_mode = "categorical",
      color_mode = "grayscale")

  return train_set, val_set


# function for training models
def train(train_set, val_set, model, save_name, save_folder, history_path, checkpoints_path, times_path = None,
          save_times = False, lr = 0.0001, lr_decay = False, max_epochs = 100):
    
    # check if training history exists for this model and don't train if it does 
    if isfile(join(history_path, save_folder, save_name)):
        return
    
    #create callbacks
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
    import time

    class EpochTime(Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    epoch_times = EpochTime()        


    best_checkpoint = ModelCheckpoint(filepath = join(checkpoints_path, save_folder, save_name),
                                     monitor = "val_loss",
                                     save_best_only = True)

    early_stop = EarlyStopping(monitor = "val_loss",
                              patience = 25,
                              restore_best_weights = True)

    reduce_lr = ReduceLROnPlateau(monitor = "val_loss",
                                 patience = 10,
                                 factor = 0.5,
                                 min_lr = 0.000001)

    csv_logger = CSVLogger(join(history_path, save_folder, save_name))
    
    callback_list = [epoch_times, csv_logger, best_checkpoint, early_stop]
    
    if lr_decay:
        callback_list.append(reduce_lr)
    
    #Training the CNN

    #compile the CNN
    model.compile(optimizer = Adam(learning_rate = lr), 
               loss = CategoricalCrossentropy(),
               metrics = ["accuracy"])

    #fit the CNN
    model.fit(x = train_set,
           validation_data = val_set,
           epochs = max_epochs,
           callbacks = callback_list)
    
    if save_times:
         #list of times taken for full epoch computations
         times_list = epoch_times.times

         #save list as json to be used later
         with open(join(times_path, save_folder, save_name + ' epoch times'), 'w') as f:
              json.dump(times_list, f)
        
    
    
