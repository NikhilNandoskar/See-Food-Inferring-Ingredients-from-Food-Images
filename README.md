# See-Food-Inferring-Ingredients-from-Food-Images

1) train.py: Is the Training model of VGG-16 built in Tensorflow
2) trasfer-learning-model.py: Is the transfer learning model of VGG-16 built using Tensornets
3) predict.py: Script for testing the trained model

Download link: 
dataset download: http://vireo.cs.cityu.edu.hk/VireoFood172/VireoFood172.zip
Original images (higher resolution): http://vireo.cs.cityu.edu.hk/VireoFood172/VireoFood172_ori.tar.gz.
label and split download: http://vireo.cs.cityu.edu.hk/VireoFood172/SplitAndIngreLabel.zip

Problem Statement: Many attentions have been given to food classification from images, but little is given to predicting the ingredients in a food image. Inferring food ingredients is important for the elders, the sick and many other groups. The last several years have seen the great advancement in food classification using neural network. We decided to build a neural network to infer the ingredients in a food image with TensorFlow.

We have leveraged Viero-Food 172 dataset which had 1,10,241 food images from 172 categories and had 353 ingredients classes. Wrote a python script which takes in food images and resizes them into 240*240 size and splits the dataset into two parts as training and testing. We take the median of all the RGB values of our images and subtract them from every pixel value. We then normalise the pixel values before splitting into training and testing.

We designed a VGG-16 architecture from scratch using TesnsorFlow. In the final layer i.e. the softmax layer we have 353 classes of food ingredients. Our model outputs top 5 probabilities of prediction. However, the training accuracy obtained was just 40%

We then implemented the Transfer Learning model of VGG-16 using Tensornets. In Tensorflow while building a model from scratch we used tf.nn.softmax_cross_entropy_with_logits_v2 function which performs softmax along with cross-entropy at the same place. However, in a pre-trained model, we already have softmax applied so we just need a function only performs cross entropy loss function. Such a function is available in Tensorflow called tf.losses.softmax_cross_entropy, which is used in our transfer learning model implementation. The accuracy obtained was 19.1%

The reasons for such low accuracy could be lack of resources available for training.

