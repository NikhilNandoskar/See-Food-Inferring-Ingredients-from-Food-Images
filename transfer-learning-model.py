# -*- coding: utf-8 -*-
"""Transfer_Learning_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nkFvgSDJsqkpfdillLUCfcl3kk3w1vKA
"""

# !pip install h5py
# !pip install scipy

# !pip install scikit-image
# !pip install tqdm

# !pip install Cython

# !pip install git+https://github.com/taehoonlee/tensornets.git

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm 
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
from PIL import Image
from random import sample
import shutil

import skimage
import skimage.io
import skimage.transform

import tensorflow as tf
import tensornets as nets

# If you have a very high-end GPU running on your machine, you can skip this section. We need this configuration, and I have run the training with NVIDIA GTX 1080Ti
# If this section throws an error comment it out
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.80

path="D:\\"
height=224
width=224
shutil.rmtree('saved_model',ignore_errors=True)
tf.set_random_seed(0)
n_classes=353

# Feature Engineering goes here. Create X_train, X_test, y_train, y_test
def prepare_json(label_lines):
    data_pair={}
    for line in label_lines:
        elements=line.strip('\n').split()
        data_pair[elements[0]]=elements[1:]
    return data_pair

def get_data(lines1,data_pair):
    cases=[]
    labels=[]
    for line in lines1:
        line=line.strip()
        elements=line.split('/')
        picture=''
        for ele in elements:
            picture=os.path.join(picture,ele)
        picture=os.path.join(path,'ready_chinese_food',picture)
        cases.append(picture)
        arr=np.array(data_pair[line],dtype=np.int32)
        for i in range(len(arr)):
            if arr[i]==-1:
                arr[i]=0
        labels.append(arr)
    labels=np.array(labels)
    print('Label reading done.')
    RGBvalues=np.zeros((len(cases),height,width,3),dtype=int)
    for i in range(len(cases)):
        im=Image.open(cases[i])
        dd=im.resize((height,width))
        dd=dd.convert('RGB')
        im.close()
        for j in range(height):
            for k in range(width):
                values=np.array(dd.getpixel((j,k)))
                RGBvalues[i][j][k]=values/255.
        if i%200==0:
            print('Data preparation: {}/{}'.format(str(i),str(len(cases))))
    return RGBvalues,labels

test_file=open(os.path.join(path,"SplitAndIngreLabel","TE.txt"),'r')
test_lines=test_file.readlines()
test_file.close()
train_file=open(os.path.join(path,"SplitAndIngreLabel","TR.txt"),'r')
train_lines=train_file.readlines()
train_file.close()
label_file=open(os.path.join(path,"SplitAndIngreLabel","IngreLabel.txt"),'r')
label_lines=label_file.readlines()
label_file.close()
data_pair=prepare_json(label_lines)
train_X,train_Y=get_data(train_lines,data_pair)
test_X,test_Y=get_data(test_lines,data_pair)

X = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3], name="input_X" )
y_truth = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="input_y")
#hold_prob = tf.placeholder(dtype = tf.float32, name="hold_probability")

# Hyper-parameters

learning_rate = 0.0002
#epochs = 100
#batch_size = 32

# Loading the VGG16 model
# Since we want to train on our own image dataset we'll set is_trainig = True and classes = 353
# nets.VGG16 returns the final layer of the VGG16 which is softmax. As VGG16 returns a layer which already applied softmax, we'll use tf.losses.softmax_cross_entropy function in our case
logits = nets.VGG16(X, is_training = True, classes = n_classes)
model = tf.identity(logits, name = 'logits')  # required to keep logits operation or else it would never run

cross_entropy = tf.losses.softmax_cross_entropy(y_truth, logits)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

matches = tf.equal(tf.argmax(model,1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(matches, tf.float32), name = 'accuracy')

model

# print outputs of each layer
logits.print_outputs()

logits.print_summary()

def pick_train_batch(n):
    list=sample(range(len(train_X)),n)
    batch_X=[]
    batch_Y=[]
    for i in list:
        batch_X.append(train_X[i])
        batch_Y.append(train_Y[i])
    return np.array(batch_X),np.array(batch_Y)
    
def pick_test_batch(n):
    list=sample(range(len(test_X)),n)
    batch_X=[]
    batch_Y=[]
    for i in list:
        batch_X.append(test_X[i])
        batch_Y.append(test_Y[i])
    return np.array(batch_X),np.array(batch_Y)

init = tf.global_variables_initializer()
test_acc = []
cost = []
cost_test = []
predict = []
n_e = 100  # As we are using a pre trained model we reaquire low number odf epochs
train_batch_size=1000
test_batch_size=200

# Training the pre trained model
save_model_path = 'saved model'
f1=open('log02.txt','w')

print('Training....')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6,allocator_type='BFC')
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    sess.run(logits.pretrained())
    print('model.pretrained done.....')
    
    print('Start Training')
    for i in range(n_e):
        train_batch_X,train_batch_Y=pick_train_batch(train_batch_size)
        test_batch_X,test_batch_Y=pick_test_batch(test_batch_size)
            
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_truth))
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cross_entropy)
            
        sess.run([optimizer], feed_dict={X:train_batch_X, y_truth:train_batch_Y})
        loss = sess.run(cross_entropy, feed_dict = {X:train_batch_X, y_truth:train_batch_Y}) 
        cost.append(loss)
            
        loss_test = sess.run(cross_entropy, feed_dict={X:test_batch_X,y_truth:test_batch_Y})
        cost_test.append(loss_test)
            
        #loss_test = sess.run(acc, feed_dict={X:batch[0],y_truth:batch[1], hold_prob:1.0})
        #train_acc += sess.run(acc, feed_dict={X:batches[0], y_truth:batches[1], hold_prob:1.0})
        if i % 5 == 0:
            f1.write("ith step Loss and Test Loss:\t"+str(i)+'\t'+str(loss)+'\t'+str(loss_test)+'\n')
            print("ith step Loss and Test Loss: ",(i,loss, loss_test))
            
        train_acc = sess.run(accuracy, feed_dict={X:train_batch_X, y_truth:train_batch_Y})
        predict.append(train_acc)
            
        test_temp = sess.run(accuracy, feed_dict={X:test_batch_X,y_truth:test_batch_Y})
        test_acc.append(test_temp)
        
    # # Save Model
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, save_model_path)

    plt.plot(predict,label='train accuracy')
    plt.plot(test_acc,label='test accuracy')
    plt.legend(loc='upper right')
    plt.title('accuracy')
    plt.savefig('acc.png')
    plt.close()
    f2=open('train-acc.txt','w')
    f2.writelines((str(b)+'\n') for b in predict)
    f2.close()
    f3=open('test-acc.txt','w')
    f3.writelines((str(b)+'\n') for b in test_acc)
    f3.close()
    plt.plot(cost,label='train loss')
    plt.plot(cost_test,label='test loss')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.savefig('loss.png')
    plt.close()
    f4=open('train-loss.txt','w')
    f4.writelines((str(b)+'\n') for b in cost)
    f4.close()
    f5=open('test-acc.txt','w')
    f5.writelines((str(b)+'\n') for b in cost_test)
    f5.close()
    inputs = {
        "features_placeholder": X,
        "labels_placeholder": y_truth,
        }
    outputs = {
        "logits": logits,
        "accuracy": accuracy,
        }
    tf.saved_model.simple_save(
        sess, 'saved_model', inputs, outputs
    )
    f1.write("train_acc\t"+str(predict[-1])+'%\n')
    f1.write("Testing Accuracy:\t"+str(test_acc[-1])+'%\n')
    print("train_acc", predict[-1])
    print("Testing Accuracy :\n",test_acc[-1])
    f1.close()

## Just trying
#with tf.Session() as sess:
 # sess.run(init)
 # sess.run(logits.pretrained())
  #print('model.pretrained')
