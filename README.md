# **Traffic Sign Recognition** 

## Writeup 

* Roman Stanchak (rstanchak <at> gmail <dot> com)
* March 27, 2017

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./data/web/cropped/web03.png "Traffic Sign 1"
[image5]: ./data/web/cropped/web04.png "Traffic Sign 2"
[image6]: ./data/web/cropped/web07.png "Traffic Sign 3"
[image7]: ./data/web/cropped/web11.png "Traffic Sign 4"
[image8]: ./data/web/cropped/web12.png "Traffic Sign 5"
[image9]: ./data/web/cropped/web13.png "Traffic Sign 6"
[image10]: ./data/web/cropped/web15.png "Traffic Sign 7"
[image11]: ./examples/training_examples_by_class.png "Number of training images by class"
[image12]: ./examples/training_samples.png
[image13]: ./examples/average_training_rgb.png
[image14]: ./examples/grayscale.jpg
[image15]: ./examples/preproccessed_training.png
[image16]: ./examples/average_training.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### 1. Writeup / README

#### 1.1 Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This document contains the project writeup. Other project artifacts:
* [Project notebook](https://github.com/rstanchak/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) 
* [Project notebook (html)](https://github.com/rstanchak/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

### 2. Data Set Summary & Exploration

#### 2.1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook. 

I used numpy and basic python functions to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43
* Number of validation examples = 4410

The table below breaks down the number training examples by class.  
| Class | Count | Class Name |
|:-----:|:-----:|:-----------|
| 0 | 	 180 	|Speed limit (20km/h) |
| 1 | 	 1980 	|Speed limit (30km/h) |
| 2 | 	 2010 	|Speed limit (50km/h) |
| 3 | 	1260 	|Speed limit (60km/h) |
| 4 | 	1770 	|Speed limit (70km/h) |
| 5 | 	1650 	|Speed limit (80km/h) |
| 6 | 	360 	|End of speed limit (80km/h) |
| 7 | 	1290 	|Speed limit (100km/h) |
| 8 | 	1260 	|Speed limit (120km/h) |
| 9     | 1320 	|No passing |
| 10    | 1800 	|No passing for vehicles over 3.5 metric tons |
| 11    | 1170 	|Right-of-way at the next intersection |
| 12    | 1890 	|Priority road |
| 13    | 1920 	|Yield |
| 14    | 690 	|Stop |
| 15 	| 540 	|No vehicles |
| 16   	| 360 	|Vehicles over 3.5 metric tons prohibited |
| 17   	| 990 	|No entry |
| 18   	| 1080 	|General caution |
| 19   	| 180 	|Dangerous curve to the left |
| 20   	| 300 	|Dangerous curve to the right |
| 21    | 270 	|Double curve |
| 22    | 330 	|Bumpy road |
| 23    | 450 	|Slippery road |
| 24    | 240 	|Road narrows on the right |
| 25    | 1350 	|Road work |
| 26    | 540 	|Traffic signals |
| 27    | 210 	|Pedestrians |
| 28    | 480 	|Children crossing |
| 29    | 240 	|Bicycles crossing |
| 30    | 390 	|Beware of ice/snow |
| 31    | 690 	|Wild animals crossing |
| 32    | 210 	|End of all speed and passing limits |
| 33    | 599 	|Turn right ahead |
| 34 	| 360 	|	Turn left ahead |
| 35    | 1080 	|Ahead only |
| 36    | 330 	|Go straight or right |
| 37    | 180 	|Go straight or left |
| 38    | 1860 	|Keep right |
| 39    | 270 	|Keep left |
| 40    | 300 	|Roundabout mandatory |
| 41    | 210 	|End of no passing |
| 42 	| 210 |	End of no passing by vehicles over 3.5 metric tons |


#### 2.2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cells of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the training data among the classes.

![alt text][image11]

Here is another exploratory visualization of the data set.  This shows 10 random training examples for each class.

![alt text][image12]

### 3. Design and Test a Model Architecture

#### 3.1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

To pre-process my image data, I centered and normalized the data, first by shifting the data by the average of each color channel of the training data, then by scaling from a [-127,127] range to [-1,1].  I chose to do this because it was discussed in the LeNet paper, the lecture videos and some Stanford lecture slides another student referenced (http://cs231n.stanford.edu/slides/2016/winter1516_lecture5.pdf). 

Here is the mean image in RGB.  It doesn't look like much.

![alt text][image16]

Viewing the color channels separately looks like an average of the triangular and circular signs.

![alt text][image13]

Here is the preprocessed training data.

![alt text][image15]

#### 3.2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The train, test, and validation sets were used as provided.  No data augmentation or train/validation mixing was done.

I chose not to augment my dataset because of time constraints.

#### 3.3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 10th code cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x32   |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x64     |
| Fully connected		| Outputs 120  									|
| RELU					|												|
| Fully connected		| Outputs 84                                    |
| RELU					|												|
| Dropout               |                                               |
| Fully connected		| Outputs 43                                    |
| Softmax				|            									|
|						|												|
 
For weight initialization, I utilized an approach from "Delving deep into rectifiers: Surpassing human-level performance on ImageNet 
classification
 by He et al., 2015", again referenced from http://cs231n.stanford.edu/slides/2016/winter1516_lecture5.pdf .

#### 3.4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 11th code cell of the ipython notebook. 

To train the model, I retained the same approach as the LeNet lab.
* Optimizer: AdamOptimizer
* Batch size: 64 (reduced due to local computing resource constraints)
* Epochs: 30
* Learning rate: 0.001
* Dropout layer keep probability: 0.50
 

#### 3.5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 12th cell of the Ipython notebook.

My approach for finding a solution was to port the LeNet lab example code and tutorial video and determine a baseline solution, and then apply some of the techniques discussed in lectures and in referenced literature to see what improved the accuracy.

The initial architecture took the raw RGB data as input and used the same network structure as the LeNet lab (adjusted for 3-channels instead of grayscale).

This resulted training and validation accuracy under the 93% threshold (Exact result not recorded).

I implemented the following adjustments:
* Reduced batch size to 64 to more quickly get feedback on training progress on an old laptop.
* Centering+normalization in the preprocessing step.  This was done based on discussion in the lecture videos and labs.  It led to a small improvement (not recorded).
* Weight initialization optimization.  This was done in an attempt to address an issue where train and validation accuracy got stuck at 5%.  After determining the issue was unrelated (just a stupid bug), I found that the optimization led to a small improvement (not recorded).
* Increased number of features in conv layers.  With training and validation accuracy still stuck at less than 93%, I referred to the architecture from the LeNet paper and noted that the minimum conv features used were 38 in the first layer and 64 in the second layer.  So to start, I bumped up my first and second layers to 36 and 64 features, respectively.  This resulted in a marked jump in the training accuracy, which was now well above the 93% benchmark.  The validation accuracy was around 91%.
* Increased Epochs to 30.   This was done because training and validation accuracy were continuing to improve after the first 10 epochs.  This resulted in the training accuracy continuing to improve, but the validation accuracy plateaued around 91%.
* Introduced dropout layer.  This was done because the training accuracy after 30 epochs was above 99%, but the validation accuracy was stuck around 91%, which I took as evidence that the network was overfitting.  The result after 30 epochs, is that the training set accuracy was 99.6% and the validation set accuracy was 95.3%.  The benchmark of 93% validation accuracy had been attained.

Unfortunately, due to external time constaints, I was not able to further investigate other optimizations so this was the final model.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 95.3%
* test set accuracy of 92.5%

### 4. Test a Model on New Images

#### 4.1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

| # | Image	| Label	| Notes |
|:--:|:------:|:----|:---------------------------------------------| 
| 1 | ![alt text][image4]  | Speed limit (70km/h) | |
| 2 | ![alt text][image5]  | Priority Road        | this image might be difficult to classify because it is skewed and low contrast. |
| 3 | ![alt text][image6]  | Stop | |
| 4 | ![alt text][image7]  | Turn Right Ahead     | |
| 5 | ![alt text][image8]  | Pedestrians          | |
| 6 | ![alt text][image9]  | Go Straight or Left  | |
| 7 | ![alt text][image10] | Go Straight or Right | this image might be difficult to classify because the image capture has an artifact (human error!). |


#### 4.2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

Here are the results of the prediction:


| # | Image	| Label	| Prediction |
|:--:|:------:|:----|:-----------|
| 1 | ![alt text][image4] | Speed limit (70km/h) | Speed limit 30 km/h |
| 2 | ![alt text][image5] | Priority Road | Priority Road |
| 3 | ![alt text][image6] | Stop | Stop |
| 4 | ![alt text][image7] | Turn Right Ahead | Turn Right Ahead |
| 5 | ![alt text][image8] | Pedestrians | Speed limit 70km/h  |
| 6 | ![alt text][image9] | Go Straight or Left | Go Straight or Left |
| 7 | ![alt text][image10] | Go Straight or Right | Yield |


The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 57%. This compares unfavorably to the accuracy on the test set of 92.5%.

#### 4.3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

1 ![alt text][image4] Speed limit (70km/h)

| Probability | Prediction |
|:-----------:|:----------:|
| 0.59 | Speed limit (30km/h) |
| 0.41 | Speed limit (70km/h) |
| 0.00 | Speed limit (80km/h) |
| 0.00 | Speed limit (120km/h) |
| 0.00 | Speed limit (100km/h) |

For the 70km/h image, the classifier predicts that the image contains a 30km/h with highest probability (59%) and 70km/h sign with slightly lower probability (41%).

![alt text][image5] Priority road

| Probability | Prediction |
|:-----------:|:----------:|
| 1.00 | Priority road |
| 0.00 | No passing for vehicles over 3.5 metric tons |
| 0.00 | Traffic signals |
| 0.00 | Road work |
| 0.00 | Right-of-way at the next intersection |

For the priority road image, the classifier predicts that the image contains a Priority road with high probability (100%). 

![alt text][image6] Stop

| Probability | Prediction |
|:-----------:|:----------:|
| 1.00 | Stop |
| 0.00 | Speed limit (30km/h) |
| 0.00 | Speed limit (80km/h) |
| 0.00 | No entry |
| 0.00 | Road work |

For the Stop sign image, the classifier predicts that the image contains a Stop sign with high probability (100%)

![alt text][image7] Turn right ahead

| Probability | Prediction |
|:-----------:|:----------:|
| 1.00 | Turn right ahead |
| 0.00 | Speed limit (20km/h) |
| 0.00 | Speed limit (30km/h) |
| 0.00 | Speed limit (50km/h) |
| 0.00 | Speed limit (60km/h) |

For the Turn right ahead sign image, the classifier predicts that the image contains a Turn right ahead sign with high probability (100%)

![alt text][image8] Pedestrians

| Probability | Prediction |
|:-----------:|:----------:|
| 0.65 | Speed limit (70km/h) |
| 0.35 | General caution |
| 0.00 | Speed limit (80km/h) |
| 0.00 | No vehicles |
| 0.00 | Speed limit (30km/h) |

For the Pedestrians image, the classifier predicts that the image contains a Speed limit (70km/h) with highest probability (65%), and General caution with lower probability (35%).  The correct class 'Pedestrians' was not in the top 5 predictions.

![alt text][image9] Go straight or left

| Probability | Prediction |
|:-----------:|:----------:|
| 1.00 | Go straight or left |
| 0.00 | Roundabout mandatory |
| 0.00 | Keep left |
| 0.00 | Ahead only |
| 0.00 | Speed limit (20km/h) |

For the Go straight or left sign image, the classifier predicts that the image contains a Turn right ahead sign with high probability (100%)

![alt text][image10] Go straight or right

| Probability | Prediction |
|:-----------:|:----------:|
| 0.59 | Yield |
| 0.41 | Traffic signals |
| 0.00 | Priority road |
| 0.00 | Go straight or right |
| 0.00 | No vehicles |

For the Go straight or right image, the classifier predicts that the the image contains a Yield sign with highest probability (59%) and a Traffic signals sign with slightly lower probability (41%).  The correct class was in the top 5, though the probability was less than 1%

