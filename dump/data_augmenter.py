
import tensorflow as tf
import keras
import cv2 as cv
import os
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
datagen = ImageDataGenerator(
                        # rotation_range =15,
                        #  width_shift_range = 0.2,
                        #  height_shift_range = 0.2,
                        #  rescale=1./255,
                        #  shear_range=0.2,
                        #  zoom_range=0.2,
                         horizontal_flip = True,
                        #  fill_mode = 'nearest',
                         data_format='channels_last',
                        #  brightness_range=[0.5, 1.5]
                         )


img_dir = "himalayan_cat" # Enter Directory of all images
data = []

i = 0
# path, dirs, _ = next(os.walk(img_dir))

file_count = 1
big_data = np.array([],dtype="uint8",ndmin=4)

img_paths = os.listdir(img_dir)
# img_count = len(img_paths) # to find number of images in folder

for img_path in img_paths:
    # print(img_path)
    # print("1")
    img = cv.imread(os.path.join(img_dir, img_path))
    cv.imshow("IMG", img)
    cv.waitKey(0)

    x = img_to_array(img)
    # x = x.reshape((1,) + x.shape)
    data.append(x)
    print(data)
    # data.append(x)
    # print(len(data))
    # print(data[0].shape)
    break

    # big_data.append(x)

    for batch in datagen.flow (x, batch_size=1, save_to_dir =r'himalayan_cat',save_prefix="a",save_format='jpg'):
        print(f"hehe{i}")
        i+=1
        if i==file_count:
            break

# print(len(big_data))
