import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import argparse
from PIL import Image
from random import sample
import shutil

path="D:\\"
height=128
width=128

shutil.rmtree('saved_model')
tf.set_random_seed(0)

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
                RGBvalues[i][j][k]=values
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

n_labels=353 # number of labels
X = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3], name="input_X" )
y_truth = tf.placeholder(dtype=tf.float32, shape=[None, n_labels], name="input_y")
hold_prob = tf.placeholder(dtype = tf.float32, name="hold_probability")

# Function for Max Pooling
def max_pool(n):
    return tf.nn.max_pool(n, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# Convolution Layers
def conv_layers(X, hold_prob):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    #conv1_1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.1))
    #conv1_2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.1))
    conv1_1_filter = tf.Variable(initializer(shape=[3, 3, 3, 64]))
    conv1_2_filter = tf.Variable(initializer(shape=[3, 3, 64, 64]))

    
    #conv2_1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.1))
    #conv2_2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], mean=0, stddev=0.1))
    conv2_1_filter = tf.Variable(initializer(shape=[3, 3, 64, 128]))
    conv2_2_filter = tf.Variable(initializer(shape=[3, 3, 128, 128]))

    #conv3_1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], mean=0, stddev=0.1))
    #conv3_2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], mean=0, stddev=0.1))
    #conv3_3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], mean=0, stddev=0.1))
    conv3_1_filter = tf.Variable(initializer(shape=[3, 3, 128, 256]))
    conv3_2_filter = tf.Variable(initializer(shape=[3, 3, 256, 256]))
    conv3_3_filter = tf.Variable(initializer(shape=[3, 3, 256, 256]))
    
    #conv4_1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], mean=0, stddev=0.1))
    #conv4_2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=0.1))
    #conv4_3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=0.1))
    conv4_1_filter = tf.Variable(initializer(shape=[3, 3, 256, 512]))
    conv4_2_filter = tf.Variable(initializer(shape=[3, 3, 512, 512]))
    conv4_3_filter = tf.Variable(initializer(shape=[3, 3, 512, 512]))
    
    #conv5_1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=0.1))
    #conv5_2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=0.1))
    #conv5_3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], mean=0, stddev=0.1))
    conv5_1_filter = tf.Variable(initializer(shape=[3, 3, 512, 512]))
    conv5_2_filter = tf.Variable(initializer(shape=[3, 3, 512, 512]))
    conv5_3_filter = tf.Variable(initializer(shape=[3, 3, 512, 512]))
    
    # First Layer
    conv1_1 = tf.nn.conv2d(X,conv1_1_filter,strides=[1,1,1,1],padding="SAME")
    conv1_1 = tf.nn.relu(conv1_1)
    conv1_1_pooling = max_pool(conv1_1)
    conv1_1_bn = tf.layers.batch_normalization(conv1_1_pooling)
    
    conv1_2 = tf.nn.conv2d(conv1_1_bn,conv1_2_filter,strides=[1,1,1,1],padding="SAME")
    conv1_2 = tf.nn.relu(conv1_2)
    conv1_2_pooling = max_pool(conv1_2)
    conv1_2_bn = tf.layers.batch_normalization(conv1_2_pooling)
    
    # Second Layer
    conv2_1 = tf.nn.conv2d(conv1_2_bn,conv2_1_filter,strides=[1,1,1,1],padding="SAME")
    conv2_1 = tf.nn.relu(conv2_1)
    conv2_1_pooling = max_pool(conv2_1)
    conv2_1_bn = tf.layers.batch_normalization(conv2_1_pooling)
    
    conv2_2 = tf.nn.conv2d(conv2_1_bn,conv2_2_filter,strides=[1,1,1,1],padding="SAME")
    conv2_2 = tf.nn.relu(conv2_2)
    conv2_2_pooling = max_pool(conv2_2)
    conv2_2_bn = tf.layers.batch_normalization(conv2_2_pooling)
    
    # Third Layer
    conv3_1 = tf.nn.conv2d(conv2_2_bn,conv3_1_filter,strides=[1,1,1,1],padding="SAME")
    conv3_1 = tf.nn.relu(conv3_1)
    conv3_1_pooling = max_pool(conv3_1)
    conv3_1_bn = tf.layers.batch_normalization(conv3_1_pooling)
    
    conv3_2 = tf.nn.conv2d(conv3_1_bn,conv3_2_filter,strides=[1,1,1,1],padding="SAME")
    conv3_2 = tf.nn.relu(conv3_2)
    conv3_2_pooling = max_pool(conv3_2)
    conv3_2_bn = tf.layers.batch_normalization(conv3_2_pooling)
    
    conv3_3 = tf.nn.conv2d(conv3_2_bn,conv3_3_filter,strides=[1,1,1,1],padding="SAME")
    conv3_3 = tf.nn.relu(conv3_3)
    conv3_3_pooling = max_pool(conv3_3)
    conv3_3_bn = tf.layers.batch_normalization(conv3_3_pooling)
    
    # Fourth Layer
    conv4_1 = tf.nn.conv2d(conv3_3_bn,conv4_1_filter,strides=[1,1,1,1],padding="SAME")
    conv4_1 = tf.nn.relu(conv4_1)
    conv4_1_pooling = max_pool(conv4_1)
    conv4_1_bn = tf.layers.batch_normalization(conv4_1_pooling)
    
    conv4_2 = tf.nn.conv2d(conv4_1_bn,conv4_2_filter,strides=[1,1,1,1],padding="SAME")
    conv4_2 = tf.nn.relu(conv4_2)
    conv4_2_pooling = max_pool(conv4_2)
    conv4_2_bn = tf.layers.batch_normalization(conv4_2_pooling)
    
    conv4_3 = tf.nn.conv2d(conv4_2_bn,conv4_3_filter,strides=[1,1,1,1],padding="SAME")
    conv4_3 = tf.nn.relu(conv4_3)
    conv4_3_pooling = max_pool(conv4_3)
    conv4_3_bn = tf.layers.batch_normalization(conv4_3_pooling)
  
    # Fifth Layer
    conv5_1 = tf.nn.conv2d(conv4_3_bn,conv5_1_filter,strides=[1,1,1,1],padding="SAME")
    conv5_1 = tf.nn.relu(conv5_1)
    conv5_1_pooling = max_pool(conv5_1)
    conv5_1_bn = tf.layers.batch_normalization(conv5_1_pooling)
    
    conv5_2 = tf.nn.conv2d(conv5_1_bn,conv5_2_filter,strides=[1,1,1,1],padding="SAME")
    conv5_2 = tf.nn.relu(conv5_2)
    conv5_2_pooling = max_pool(conv5_2)
    conv5_2_bn = tf.layers.batch_normalization(conv5_2_pooling)
    
    conv5_3 = tf.nn.conv2d(conv5_2_bn,conv5_3_filter,strides=[1,1,1,1],padding="SAME")
    conv5_3 = tf.nn.relu(conv5_3)
    conv5_3_pooling = max_pool(conv5_3)
    conv5_3_bn = tf.layers.batch_normalization(conv5_3_pooling)
    
    # Flattening the Neurons
    flat = tf.layers.flatten(conv5_3_bn)
    
    # Fully Connected Layers
    full_layer_1 = tf.contrib.layers.fully_connected(inputs=flat,num_outputs=4096,activation_fn=tf.nn.relu)
    full_layer_1 = tf.nn.dropout(full_layer_1, keep_prob=hold_prob)
    #full_layer_1 = tf.layers.batch_normalization(full_layer_1)
    
    full_layer_2 = tf.contrib.layers.fully_connected(inputs=flat,num_outputs=4096,activation_fn=tf.nn.relu)
    full_layer_2 = tf.nn.dropout(full_layer_2, keep_prob=hold_prob)
    #full_layer_2 = tf.layers.batch_normalization(full_layer_2)
    
    out = tf.contrib.layers.fully_connected(inputs=full_layer_2,num_outputs=n_labels,activation_fn=None)
    return out

y_pred = conv_layers(X,hold_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_truth))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(cross_entropy)

matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_truth,1))
acc = tf.multiply(tf.reduce_mean(tf.cast(matches, dtype=tf.float32)), 100)

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
n_e = 50000
train_batch_size=100
test_batch_size=20
f1=open('log02.txt','w')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for i in range(n_e):
        train_batch_X,train_batch_Y=pick_train_batch(train_batch_size)
        test_batch_X,test_batch_Y=pick_test_batch(test_batch_size)
        
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_truth))
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cross_entropy)
        
        sess.run([optimizer], feed_dict={X:train_batch_X, y_truth:train_batch_Y, hold_prob:0.6})
        loss = sess.run(cross_entropy, feed_dict = {X:train_batch_X, y_truth:train_batch_Y, hold_prob:1.0}) 
        cost.append(loss)
        
        loss_test = sess.run(cross_entropy, feed_dict={X:test_batch_X,y_truth:test_batch_Y, hold_prob:1.0})
        cost_test.append(loss_test)
        
        #loss_test = sess.run(acc, feed_dict={X:batch[0],y_truth:batch[1], hold_prob:1.0})
        #train_acc += sess.run(acc, feed_dict={X:batches[0], y_truth:batches[1], hold_prob:1.0})
        if i % 10 == 0:
            f1.write("ith step Loss and Test Loss:\t"+str(i)+'\t'+str(loss)+'\t'+str(loss_test)+'\n')
            print("ith step Loss and Test Loss: ",(i,loss, loss_test))
        
        train_acc = sess.run(acc, feed_dict={X:train_batch_X, y_truth:train_batch_Y, hold_prob:1.0})
        predict.append(train_acc)
        
        test_temp = sess.run(acc, feed_dict={X:test_batch_X,y_truth:test_batch_Y, hold_prob:1.0})
        test_acc.append(test_temp)

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
        "prediction": y_pred,
        "accuracy": acc,
        }
    tf.saved_model.simple_save(
        sess, 'saved_model', inputs, outputs
    )
    f1.write("train_acc\t"+str(predict[-1])+'%\n')
    f1.write("Testing Accuracy:\t"+str(test_acc[-1])+'%\n')
    print("train_acc", predict[-1])
    print("Testing Accuracy :\n",test_acc[-1])
    f1.close()
