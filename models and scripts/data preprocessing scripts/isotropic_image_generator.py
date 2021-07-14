# -*- coding: utf-8 -*-

from os.path import join

from numpy.random import randint
import numpy as np

from PIL import Image

def generate_noisy_array(max_brightness, image_size):
    noisy_array = np.zeros((image_size, image_size))
    
    for i in range(image_size):
        for j in range(image_size):
            noisy_array[i][j] = randint(0, max_brightness, dtype = np.uint8)
            
    return noisy_array        

def generate_isotropic_images(save_path, num_images, max_brightness = 50, image_size = 256):
    
    for num in range(num_images):
        noisy_image = Image.fromarray(generate_noisy_array(max_brightness,
                                                           image_size))
        noisy_image = noisy_image.convert('L')
        
        image_path = join(save_path, str(num) + '.png')
        noisy_image.save(image_path)
        
    
    
if __name__ == '__main__':
    generate_isotropic_images('D:\MPhys Liquid Crystals Datasets/dataset used for training 4 phases models/split/test/isotropic',
                              num_images = 400)