import csv
import cv2
import numpy as np

# Load data
# data_path = './data/' # default sim data from Udacity
data_path = '../simulator/training data/' # my sim data

csv_file = data_path + 'driving_log.csv'
lines = []
with open(csv_file) as f:
  reader = csv.reader(f)
  next(reader) # skip 1st line
  for line in reader:
    lines.append(line)

images = []
measurements = []
# print('lines[0:5] = ', lines[0:5])
for line in lines:
  # print('line = ',line)
  source_path = line[0] #center channel video
  # print('source_path = ', source_path)
  filename = source_path.split('/')[-1] #remove intermediate path
  image_file = data_path + 'IMG/' + filename
  image = cv2.imread(image_file)
  images.append(image)

  measurement = float(line[3]) #steering angle
  measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# generate flipped image data
X_train_flip = np.array(np.fliplr(images))
y_train_flip = -np.array(measurements)

exit()

# train model  
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D

model = Sequential()

model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))

model.add(Conv2D(input_shape=(160,320,3),
                 nb_filter=6,
                 nb_row=5,
                 nb_col=5,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(nb_filter=6,
                 nb_row=5,
                 nb_col=5,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam')
model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=8)

model.save('model.h5')

exit()