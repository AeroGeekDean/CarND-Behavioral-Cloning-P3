import csv
import cv2
import numpy as np

# Load data
# data_path = './data/' # default sim data from Udacity
# data_path = '../simulator/training data/' # my sim data
data_path = './training_data/'

csv_file = data_path + 'driving_log.csv'
lines = []
with open(csv_file) as f:
  reader = csv.reader(f)
  next(reader) # skip 1st line
  for line in reader:
    lines.append(line)

crop_hgt = 40
images=[]
measurements = []
for line in lines:
 
  source_path = line[0] #center channel video
  filename = source_path.split('/')[-1] #remove intermediate path
  image_file = data_path + 'IMG/' + filename
  image = cv2.imread(image_file)[crop_hgt:]
  images.append(image)

  measurement = float(line[3]) #steering angle
  measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# generate flipped image data
X_train_flip = np.array(np.fliplr(images))
y_train_flip = -np.array(measurements)

X_train = np.concatenate((X_train, X_train_flip), axis=0)
y_train = np.concatenate((y_train, y_train_flip), axis=0)

# train model  
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D
from keras.layers.convolutional import Conv2D

model = Sequential()
imshape = images[0].shape
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=imshape))
model.add(Conv2D(input_shape=(160,320,3),
                 nb_filter=24,
                 nb_row=8,
                 nb_col=8,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(nb_filter=36,
                 nb_row=5,
                 nb_col=5,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='relu'))
model.add(Conv2D(nb_filter=48,
                 nb_row=5,
                 nb_col=5,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='relu'))
model.add(Conv2D(nb_filter=64,
                 nb_row=5,
                 nb_col=5,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='relu'))
model.add(Conv2D(nb_filter=64,
                 nb_row=5,
                 nb_col=5,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam')
model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=8,
          batch_size=64)

model.save('model.h5')

exit()
