import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

#-----------------
# Data processing
#-----------------

# Load data
data_path = './training_data/' # AWS instance
# data_path = './data/' # default sim data from Udacity
# data_path = '../Simulator/' # my local sim data
img_path = data_path+'IMG/'
csv_file = data_path+'driving_log.csv'

# read the CSV file into a list
samples = []
with open(csv_file) as f:
    reader = csv.reader(f)
    next(reader) # skip 1st line, in case it's a header line
    for line in reader:
        samples.append(line)

# split out the validation dataset
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# offset camera steering function
# Attempt to provide more correction when steering AWAY from camera,
# and less correction when towards camera
def steering_corr(steer):
    k = 1.0
    b = 0.25
#     return k*(np.exp(steer)-1) + b
    return b # Aw crap, doesn't work. Just stick with bias then.

#set image color space
colorspace = cv2.COLOR_BGR2RGB

# generator function that acturally reads the image(s), on demand / as needed
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
#         shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # center camera
                img_name = img_path+batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(img_name), colorspace)
                center_angle = float(batch_sample[3])

                # left camera
                img_name = img_path+batch_sample[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(img_name), colorspace)
                left_angle = center_angle + steering_corr(center_angle)
                left_angle = min(max(left_angle, -1.0), 1.0)

                # right camera
                img_name = img_path+batch_sample[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(img_name), colorspace)
                right_angle = center_angle - steering_corr(-center_angle)
                right_angle = min(max(right_angle, -1.0), 1.0)

                # append data from cameras
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            X_train_orig = np.array(images)
            y_train_orig = np.array(angles)

            # create flipped image data as well
            X_train_flip = np.array(np.fliplr(images))
            y_train_flip =-np.array(angles)

            X_train = np.concatenate((X_train_orig, X_train_flip), axis=0)
            y_train = np.concatenate((y_train_orig, y_train_flip), axis=0)

            yield sklearn.utils.shuffle(X_train, y_train)

# instantiate generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# determine the input image shape
# by looking at 1st frame's center image
img_name = img_path+train_samples[0][0].split('/')[-1]
image0 = cv2.cvtColor(cv2.imread(img_name), colorspace)
imshape = image0.shape

#------------------
# Model definition
#------------------

# based upon Nvidia paper:
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Lambda(lambda x: ((x/255.0)-0.5),
                 input_shape=imshape,
                 name='lambda'))
model.add(Cropping2D(cropping=((50,20),(0,0)),
                     name='crop'))
model.add(Conv2D(nb_filter=24,
                 nb_row=5,
                 nb_col=5,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='elu',
                 name='conv1'))
model.add(Conv2D(nb_filter=36,
                 nb_row=5,
                 nb_col=5,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='elu',
                 name='conv2'))
model.add(Conv2D(nb_filter=48,
                 nb_row=5,
                 nb_col=5,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='elu',
                 name='conv3'))
model.add(Conv2D(nb_filter=64,
                 nb_row=3,
                 nb_col=3,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='elu',
                 name='conv4'))
model.add(Conv2D(nb_filter=64,
                 nb_row=3,
                 nb_col=3,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='elu',
                 name='conv5'))
model.add(Flatten(name='flat'))
model.add(Dropout(0.5, name='dropout'))
model.add(Dense(100, activation='elu', name='dense1'))
model.add(Dense(50, activation='elu', name='dense2'))
model.add(Dense(10, activation='elu', name='dense3'))
model.add(Dense(1, name='output'))

# model compilation
model.compile(loss='mse', optimizer='adam')

# set up Model checkpoint saving callback
from keras.callbacks import ModelCheckpoint

save_filename = 'model.h5'
checkpoint = ModelCheckpoint(save_filename,
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=False)
callbacks_list = [checkpoint]

# train the model
history_obj =  model.fit_generator(train_generator,
                                   samples_per_epoch=len(train_samples)*6,
                                   validation_data=validation_generator,
                                   nb_val_samples=len(validation_samples)*6,
                                   nb_epoch=6,
                                   callbacks=callbacks_list)

# plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'],'x-')
plt.plot(history_obj.history['val_loss'],'o-')
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.grid()
plt.show()

# print out model summary
print(model.summary())

exit()
