import csv
import cv2
import numpy as np

# Load data
# data_path = './data/' # default sim data from Udacity
# data_path = '../Simulator/' # my sim data
data_path = './training_data/' # AWS sim data
csv_file = data_path + 'driving_log.csv'

crop_hgt = 40
steer_corr = 0.02 # [normalized +/-1.0] (+) right

images=[]
measurements = []

with open(csv_file) as f:
  reader = csv.reader(f)
  next(reader) # skip 1st line
  for line in reader:
    # center channel
    steering_angle = float(line[3]) #steering angle
#    if(np.abs(steering_angle)>0.1): 
    if(1): 
      source_path = line[0]
      filename = source_path.split('/')[-1] #remove intermediate path
      image_file = data_path + 'IMG/' + filename
      image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)[crop_hgt:]
      images.append(image)

      measurements.append(steering_angle)

    # left channel
    if(np.random.random()<.1):
#    if(1):
      source_path = line[1]
      filename = source_path.split('/')[-1] #remove intermediate path
      image_file = data_path + 'IMG/' + filename
      image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)[crop_hgt:]
      images.append(image)

      measurement = steering_angle + steer_corr #steering angle
      measurement = min(measurement, 1.0)
      measurements.append(measurement)

    # right channel
    if(np.random.random()<.1):
#    if(1):
      source_path = line[2]
      filename = source_path.split('/')[-1] #remove intermediate path
      image_file = data_path + 'IMG/' + filename
      image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)[crop_hgt:]
      images.append(image)

      measurement = steering_angle - steer_corr #steering angle
      measurement = max(measurement, -1.0)
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
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D

model = Sequential()
imshape = images[0].shape
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=imshape))
model.add(Conv2D(nb_filter=24,
                 nb_row=8,
                 nb_col=8,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='elu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(nb_filter=36,
                 nb_row=5,
                 nb_col=5,
                 subsample=(2,2),
                 border_mode='valid',
                 activation='elu'))
model.add(Conv2D(nb_filter=48,
                 nb_row=5,
                 nb_col=5,
		 subsample=(2,2),
                 border_mode='valid',
                 activation='elu'))

model.add(Conv2D(nb_filter=64,
                 nb_row=5,
                 nb_col=5,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='elu'))
model.add(Conv2D(nb_filter=64,
                 nb_row=5,
                 nb_col=5,
                 subsample=(1,1),
                 border_mode='valid',
                 activation='elu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam')
model.fit(X_train, y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=6,
          batch_size=64)

model.save('model.h5')

exit()
