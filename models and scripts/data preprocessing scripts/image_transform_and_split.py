# -*- coding: utf-8 -*-

from os.path import join
from os import listdir, mkdir

from PIL import Image

from numpy.random import randint



def load_images(path, grayscale):
    images = []
    
    #the join function from os.path allows us to specify an image to be opened
    #the convert method with the parameter 'L' opens the image as grayscale
    for file in listdir(path):
        image_path = join(path, file)
        if grayscale:
            images.append(Image.open(image_path).convert('L'))
        else:
            images.append(Image.open(image_path))
                          
    return images

def save_images(image_list, output_path, custom_prefix = None):
    num = 0
    for image in image_list:
        if custom_prefix == None:
            new_path = output_path + '/' + str(num) + '.png'
            
        else:
            new_path = output_path + '/' + custom_prefix + str(num) + '.png'
            
        image.save(new_path)
        num += 1


#accepts a path to class folders
#creates empty folders for those classes in a new directory
def make_classes(classes_path, output):
    for class_name in listdir(classes_path):
        new_dir_path = join(output, class_name)
        mkdir(new_dir_path)
        
        
#crop an arbitrary rectangle focused on the centre        
def crop_centre(image, crop_width, crop_height):
    width, height = image.size
    return image.crop(((width - crop_width) // 2,
                         (height - crop_height) // 2,
                         (width + crop_width) // 2,
                         (height + crop_height) // 2))        
            
def crop_square_resize(image, image_size):
    width, height = image.size
    
    #check if the image is square and crop if not square
    #crop_centre function is used to crop the largest square focused on centre
    if not width == height:
        image = crop_centre(image,
                          crop_width = min(width, height),
                          crop_height = min(width, height))
    
    return image.resize((image_size, image_size))
      

def crop_sub_images(image, sub_split, sub_size):
    width, height = image.size
    
    #how the image will be split up along the x and y axes
    x_split = sub_split[0]
    y_split = sub_split[1]
    
    all_subs = []
    
    i = 0
    j = 0
    
    while i < x_split:
        x_sub = image.crop((width * i / x_split, 
                          0,
                          width * (i+1) / x_split,
                          height))
        
        while j < y_split:
            xy_sub = x_sub.crop((0,
                                 height * j / y_split,
                                 width / x_split,
                                 height * (j+1) / y_split))
            
            xy_sub = crop_square_resize(xy_sub, sub_size)
            all_subs.append(xy_sub)
            j += 1
        
        j = 0    
        i += 1    
        
    return all_subs    
    

def transform(input_path, output_path, image_size, sub_split = None, grayscale = False):
    images = load_images(path = input_path, grayscale = grayscale) 
    new_images = []
    
    if sub_split is not None:
        new_images = [crop_sub_images(image, sub_split, image_size) for image in images]
        
        # flatten new_images into a list of images
        new_images = [image for sub_images in new_images for image in sub_images]
        
    else:
        new_images = [crop_square_resize(image, image_size) for image in images]
        
    save_images(new_images, output_path)    
            

    
    


def transform_split(input_path, output_path, valid_frac, test_frac, image_size, grayscale = False):
    train_path = join(output_path, 'train')
    mkdir(train_path)
    make_classes(classes_path = input_path, output = train_path)
    
    valid_path = join(output_path, 'valid')
    mkdir(valid_path)
    make_classes(classes_path = input_path, output = valid_path)
    
    test_path = join(output_path, 'test')
    mkdir(test_path)
    make_classes(classes_path = input_path, output = test_path)
    
    for class_folder in listdir(input_path):
        class_path = join(input_path, class_folder)
        
        
        #train_set list used to temporarily store images from a class
        #loop through the list and apply crop and resize to each element
        train_set = load_images(path = class_path, grayscale = grayscale)
        
        #set predetermined sizes for valid and test sets
        train_size = len(train_set)
        valid_size = int(valid_frac * (train_size))
        test_size = int(test_frac * (train_size))
        
        for i in range(train_size):
            train_set[i] = crop_square_resize(train_set[i], image_size)
            
        
        valid_set = []
        
        #randomly add elements from train set to valid set
        #delete an element from train set if it is added to valid set
        #train_size now needs to be updated after each loop due to the del 
        #stop adding when valid set reaches predetermined size 
        for i in range(valid_size):
            train_size = len(train_set)
            rand_i = randint(0, train_size)
            valid_set.append(train_set[rand_i])
            del train_set[rand_i]
                
        test_set = []
        
        #same procedure as valid set 
        for i in range(test_size):
            train_size = len(train_set)
            rand_i = randint(0, train_size)
            test_set.append(train_set[rand_i])
            del train_set[rand_i]
            
        
        
        save_path = join(train_path, class_folder)
        save_images(image_list = train_set, output_path = save_path)
        
        save_path = join(valid_path, class_folder)
        save_images(image_list = valid_set, output_path = save_path)
        
        save_path = join(test_path, class_folder)
        save_images(image_list = test_set, output_path = save_path)
        
        

input_p = 'D:/MPhys Liquid Crystals Datasets/Ingo December videos and images/cholesteric/temp'
output_p = 'D:/MPhys Liquid Crystals Datasets/Ingo December videos and images/cholesteric/cholesteric'

subs_list = []

imgs_list = load_images(input_p, grayscale = False)

for image in imgs_list:
    for sub in crop_sub_images(image, (2,3), 256):
        subs_list.append(sub)
    
    
save_images(subs_list, output_path = output_p, custom_prefix = 'Ch_f')    

        
        
        
        

  
  
  
  
  
            
    