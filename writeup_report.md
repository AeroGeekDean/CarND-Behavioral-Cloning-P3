# **Udacity Self-Driving Car Project 3: Behavioral Cloning**

## Project Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of **good** driving behavior
* Build, a convolution neural network (CNN) in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_04_07_23_14_48_860.jpg "Center camera image"
[image2]: ./examples/left_2017_04_07_23_14_48_860.jpg "Left camera image"
[image3]: ./examples/right_2017_04_07_23_14_48_860.jpg "Right camera image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` - containing the script to create and train the model
* `drive.py` - for driving the car in autonomous mode
* `model.h5` - containing a trained convolution neural network
* `writeup_report.md` - (this page) summarizing the results
* `video.mp4` - showing the trained model successfully drive around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a slightly modified variation of the NVIDIA model, as described in their published [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). It consists mainly of 5 convolutional layers followed by 4 dense layers. A feature normalization lambda layer in used in the front end, and a dropout layer is used between the convolution and dense layers. ELU activation is used throughout.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to help reduce overfitting.

The dataset was randomly shuffled and split out for training and validation (80/20). The model was tested by running it through the simulator to verify the vehicle stay on the track. Two test laps were recorded to make sure the first lap wasn't just a lucky fluke!

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Judicial selection of training data was crucial for my success in training a model that would stay on the road. **Only two laps of training data were used**, one lap clockwise and one lap counter-clockwise around the track.

Training data was strategically created such that it would allow the model the greatest chance to clone the desired behaviors.

In additional to the center camera, both the left and right camera images were also used. Additive bias adjustments to the original label (steering angle) were made, to simulate correction back towards the center camera image. This triples (3x) the original lap data.

>  4,475 image frames x 3 cameras = 13,425 images

Additionally, duplicate horizontal flipped images were created along with the associated labels. This further doubles (2x) the data above. Thus bringing the total amount of data to be 6x the original laps.

> 13,425 frames x 2 for flipping = 26,850 total labeled images

**Note** that even with these data augmentation techniques employed, the ratio of 'number of data samples' to 'number of parameters' of the final model was still ~3%.

> 26,850 n_samples / 981,819 n_parameters = ~2.7%

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was 2 folds:
- **Follow the recommendation from the lessons.** This allows me to experience the issues that was discussed and tinker with adjustments to see their effects.
- **Start with / mimic a known working architecture and then iterate.** I started with the Nvidia model.

I iterated through the process of creating training data and evaluating the model a few times. Came up with a training data creation strategy and ended up adding an **additional dense layer** and a **dropout layer** to the model. I also changed the all activations from `RELU` to `ELU`, as ELU seems to 'penalize' negative neuron inputs while RELU just zeros out ignores it.

The data was automatically split 80/20 into training and validation.

#### 2. Final Model Architecture

My final model structure is as follow:

|Model layers|
|:---|
||
|Input image<br>(160,320,3)|
||
|Lambda layer<br>(feature normalization, 160x320x3)|
||
|Cropping2D layer<br>(remove sky and hood, 90x320x3)|
||
|Convolutional layer 1<br>(24 filter depth, 5x5 kernel, 2x2 stride, 43x158x24)|
||
|Convolutional layer 2<br>(36 filters depth, 5x5 kernel, 2x2 stride, 20x77x36)|
||
|Convolutional layer 3<br>(48 filters depth, 5x5 kernel, 2x2 stride, 8x37x48)|
||
|Convolutional layer 4<br>(64 filters depth, 3x3 kernel, 1x1 stride, 6x35x64)|
||
|Convolutional layer 5<br>(64 filters depth, 3x3 kernel, 1x1 stride, 4x33x64)|
||
|Flattening layer<br>(4x33x65 -> 8448)|
||
|Dropout layer<br>(50%)|
||
|Dense layer 1<br>(100 output units)|
||
|Dense layer 2<br>(50 output units)|
||
|Dense layer 3<br>(10 output units)|
||
|Dense layer 4<br>(1 output unit)|
||
|Steering output|

#### 3. Creation of the Training Set & Training Process

I collected several laps of simulator data using a mouse as input control for greater precision and smoothness. At first I drove these laps following typical 'racing lines', trying to smoothly **straightening out the corners by using the full width of the track.**

The resulting model was able to smoothly follow most of the gentle portions of the track. However, it had issues recognizing the sharper corners, and also **the model had a tendency to stay on right side of the track** (sort of like the 'racing lines' of my training data, but at wrong places on the track).

I then realized that evaluating and generating a 'racing line' path planning strategy would be a very difficult challenge for a model.

After tinkering with my model some, including playing with various steering correction functions for L/R image data augmentation. I finally realized that the amount of data -vs- model complexity is way out of balance. My 2 laps of data contains ~26k number of samples, while the model has just under 1 million parameters! That's only about 2~3%. Thus **I don't even have barely enough data to properly train and generalize the model!**

>*Mentally relating to simultaneous solving equations from high school algebra, where to analytically solve X parameters exactly, you need X number of different equations.* (Granted, training a NN model is different than analytically solving exact algebraic solutions...)

Thus I decided that... **if there's going to be any chance for my model to pickup on the desired behavior of successfully navigating around the track, it needs to be trained to focus on staying in between the lane lines. And in order to do that, the training data MUST strongly reinforce this behavior. Thus the training data MUST be driven at the center of the track as perfectly as possible, to provide a chance for the model to pickup on this behavior pattern.**

Here is an example image of center lane driving:
![alt text][image1]

And the associated images from the Left camera:
![alt text][image2]

And the Right cameras
![alt text][image3]

With this training data approach, I ended up NOT needing to record vehicle recovery data from the sides of the track.

There's [link to a video](./video.mp4) showing the trained model successfully navigating the track autonomously, for 2 laps.
