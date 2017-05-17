import csv
import cv2
import numpy as np
import sklearn
import re
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, Cropping2D
from keras.optimizers import Adam
from random import shuffle
from sklearn.model_selection import train_test_split
		
# <markdowncell>
# ### Help functions:
# <codecell>
def return_true_for_given_percentage(percent):
    random = np.random.randint(100)
    if random < percent:
        return True
    else:
        return False
    
def to_float_func(a):
    return float(a)

# Deletes given percentage of samples with steering angle < angle_thresh
def remove_low_steering(data, percentage, angle_thresh):
    array = np.asarray(train_samples)
    vfunc = np.vectorize(to_float_func)
        
    steer_floats_arr = vfunc(array[:,3])
    indexes_small_steer_floats = np.where(np.abs(steer_floats_arr) < angle_thresh)[0]
    rows = []
    for i in list(indexes_small_steer_floats):
        if (return_true_for_given_percentage(percentage)):
            rows.append(i)
    print("{} samples with low steering angle were removed".format(len(rows)))
    data2 = np.delete(array, rows, 0)
    return data2

# returns indexes for center, left and right camera - each with 33% probability
def get_camera_index():
    random = np.random.randint(3)
    if (random == 0):
        return 0    # center camera index
    elif (random == 0):
        return 1    # left camera index
    else:
        return 2    # right camera index

def resize(image):
    resized = cv2.resize(image, (IMAGE_COLS, IMAGE_ROWS_BEFORE_CROP))
    return resized

def get_image_from_sample_line(sample, camera_index, resizing):
    image_name = re.split("[\\\/]+", sample[camera_index])[-1].lstrip()
    image_path = './data/IMG/'+ image_name
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   
    if (resizing==True):
        # resizing must be also performed in drive.py - before feeding the image to the model
        image = resize(image)
    return image

# flip image and steering for half of samples
def flip_image_and_steering(image, steering):
    random = np.random.randint(2)
    if (random == 0):
        image = np.fliplr(image)
        steering = -steering
    return image, steering

# <markdowncell>
# ### Samples generator definition
# <codecell>
def generator(samples, batch_size, angle_correction):
    num_samples = len(samples)
    angle_corrections = [0, angle_correction, -angle_correction]
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                camera_index = get_camera_index()
                image = get_image_from_sample_line(batch_sample, camera_index, resizing=True)                
                steering = float(batch_sample[3]) + angle_corrections[camera_index]
                
                image, steering = flip_image_and_steering(image, steering)
                
                images.append(image)
                angles.append(steering)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# <markdowncell>
# ### DNN model definition (Nvidia pipeline)
# <codecell>
def build_nvidia_model(dropout=.4):
    model = Sequential()
    input_shape_before_crop=(IMAGE_ROWS_BEFORE_CROP,IMAGE_COLS, CHANNELS)
    input_shape_after_crop=(IMAGE_ROWS, IMAGE_COLS, CHANNELS)
    # trim image to only see section with the road
    model.add(Cropping2D(cropping=((IMAGE_CROP_TOP,IMAGE_CROP_BOTTOM), (0,0)), input_shape=input_shape_before_crop))
    # pixels normalization using Lambda method
    model.add(Lambda(lambda x: x/127.5-1, input_shape=input_shape_after_crop))
    
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Flatten())    
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')
    return model

# <markdowncell>
# ### Help functions for data visualization
# <codecell>
def get_steering_angles(samples):
    angles = []
    dummy_image = np.zeros((10,10,3), np.uint8)
    cnt =0
    for sample in samples:
        steering = float(sample[3]) 
        steering_temp = steering
        if flip_active==True:
            image, steering = flip_image_and_steering(dummy_image, steering)
            if (steering_temp!=steering):
                cnt = cnt+1
        angles.append(steering)
    print("{} flips made".format(cnt))    
    return angles

def draw_histogram_of_steering_angles(samples, title):
    plt.figure(figsize=(12,4))
    n, bins, patches = plt.hist(get_steering_angles(samples), 100)
    plt.xlabel('Steering angles')
    plt.ylabel('No. of samples')
    plt.title(title)
    
def get_random_image_indexes(samples, no_indexes):
    indexes = []
    for i in range(1,no_indexes+1):
        index = np.random.randint(0, len(samples))
        indexes.append(index)
    return indexes    
    
def draw_images_examples(samples, indexes, images_x, images_y, title, transform_image):
    fig = plt.figure(figsize=(16,6))
    fig.suptitle(title, fontsize=15)
    
    cnt = 0
    for index in indexes:
        sample = samples[index]
        steering = float(sample[3])  
        image = get_image_from_sample_line(sample, 0, resizing=transform_image)
        if (transform_image==True):
            image = image[IMAGE_CROP_TOP:-IMAGE_CROP_BOTTOM, :]
        image = image.squeeze()
        cnt = cnt + 1
        ax = fig.add_subplot(images_y,images_x, cnt)
        ax.title.set_text("Steering: {}".format(steering))
        plt.imshow(image)
       
# <markdowncell>
# ### Main code starts here, reading samples, splitting data
# <codecell>

# Image dimesions after resizing from 160x320 and after cropping
IMAGE_ROWS = 100
IMAGE_COLS = 100
CHANNELS = 3

IMAGE_CROP_TOP = 40
IMAGE_CROP_BOTTOM = 20
flip_active = False

IMAGE_ROWS_BEFORE_CROP = IMAGE_ROWS + IMAGE_CROP_TOP + IMAGE_CROP_BOTTOM

samples = []
with open ('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# <markdowncell>
# ### Data visualization
# <codecell>
cols_images_to_present = 4;
rows_images_to_present = 2;
img_random_indexes = get_random_image_indexes(train_samples, cols_images_to_present*rows_images_to_present)
draw_images_examples(train_samples, img_random_indexes, cols_images_to_present, rows_images_to_present, 'Examples of original images from training set', transform_image=False)
draw_images_examples(train_samples, img_random_indexes, cols_images_to_present, rows_images_to_present, 'Examples of transformed images from training set', transform_image=True)

print("Number of training samples: {}".format(len(train_samples)))
draw_histogram_of_steering_angles(train_samples, 'Histogram of training samples')
train_samples = remove_low_steering(train_samples, 85, 0.03)  
print("Number of training samples after removing low steering: {}".format(len(train_samples)))
draw_histogram_of_steering_angles(train_samples, 'Histogram of training samples after equalization')    
flip_active = True 
draw_histogram_of_steering_angles(train_samples, 'Histogram of training samples after equalization and steer flipping')  

# <markdowncell>
# ### Run and save the model
# <codecell>
ANGLE_CORRECTION = 0.25
BATCH_SIZE = 128
NO_EPOCHS = 3

train_generator = generator(train_samples, BATCH_SIZE, ANGLE_CORRECTION)
validation_generator = generator(validation_samples, BATCH_SIZE, ANGLE_CORRECTION)

model = build_nvidia_model()

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, \
                    epochs=NO_EPOCHS, validation_data=validation_generator, \
                    validation_steps=len(validation_samples)/BATCH_SIZE)

model.save('model.h5')

