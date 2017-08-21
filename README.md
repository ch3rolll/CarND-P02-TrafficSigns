**Traffic Sign Recognition** 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to read the csv file and calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

A bar chart is used for visualizing the sample count for every signle class.

###Design and Test a Model Architecture

#### First step: grayscale. 
The reason behind it is that grayscale uses less infomation and big signal-to-noise ratio.

But weird thing happend to me and I will keep debugging it in the following days.

I tried some many methods: like just average the rgb, tf.image.rgb_to_grayscale, cv2.cvtColor. 
And also np.dot(rgb[...,:3], [0.299, 0.587, 0.114]). I am wondering why the images are not grayscaled. Maybe something wrong with my original dataset?


##### Second step: normalize.
The simplest one has been used. (img - 128)/128.


#### 3. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Normalized+grascaled image  | 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flattern      |  outputs 400              |
| Fully connected		| 400 --> 120									|
| Fully connected		| 120 --> 84       									|
| Fully connected		| 84 --> 49       									|

 

#### 4. Train my model

To train the model, I used an cross_entropy as loss function. Optimize the loss function with AdamOptimizer by reducing the mean value.

My model parameters:
* EPOCHS = 30
* BATCH_SIZE = 128
* rate = 0.0008

#### 5. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.940
* test set accuracy of 0.926

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Not really very clear about this two questions, I will spend more time on the real meaning behind the network. Like the fully-connected network, not really know the meaning of numbers.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The image size is not the same as what we have in the dataset.
So I used img = cv2.resize(img, (32,32)) to get those pictures in shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

I chose a blurry speed limit(30) to see the result.

Here are the results of the prediction:
* Speed Limit(30) --> Speed Limit(30)
* Speed Limit(50) --> Speed Limit(50)
* End of Speed Limit(80) --> End of Speed Limit(80)
* Stop --> Stop
* Turn right ahead --> Turn right ahead
* Ahead only --> Ahead only

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 100%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Lets take the first image as an example. The rest can be seen in the html file.

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Speed limit(30 m/s)									| 
| .0     				| Speed limit(20 m/s) 								|
| .0					| Speed limit(50 m/s)											|
| .0	      			| Speed limit(60 m/s)					 				|
| .0				    | Speed limit(70 m/s)      							|



