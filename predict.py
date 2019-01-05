import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.saved_model import tag_constants

height = 128
width = 128
n_labels = 353  # number of labels

label_path="C:\\Users\\10253\\Downloads\\SplitAndIngreLabel\\IngredientList.txt"
path_prefix = "C:\\Users\\10253\\Downloads\\ready_chinese_food"
image_path = '/1/4_41.jpg'
image_path = image_path.split('/')
tmp = ''
tmp = os.path.join(tmp, path_prefix)
for ele in image_path:
    tmp = os.path.join(tmp, ele)
label_file=open(label_path,'r')
labels=[]
for line in label_file.readlines():
    labels.append(line.strip())
label_file.close()

im = Image.open(tmp)
dd = im.resize((height, width))
dd = dd.convert('RGB')
im.close()
RGBvalues=np.zeros((height,width,3),dtype=np.uint8)
for j in range(height):
    for k in range(width):
        values = np.array(dd.getpixel((j, k)),dtype=np.uint8)
        RGBvalues[j][k] = values
RGBvalues=np.array([RGBvalues],dtype=np.uint8)

my_graph = tf.Graph()
with my_graph.as_default():
    with tf.Session(graph=my_graph) as sess:
        print("restoring...")
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            'trained02\\saved_model'
        )
        print("loaded")
        X = my_graph.get_tensor_by_name('input_X:0')
        y_truth = my_graph.get_tensor_by_name('input_y:0')
        hold_prob = my_graph.get_tensor_by_name('hold_probability:0')
        prediction = my_graph.get_tensor_by_name('fully_connected_2/BiasAdd:0')
        result=sess.run(prediction,feed_dict={X:RGBvalues,hold_prob:1.0})
        pass

result=result.to_numpy_array().tolist()[0]
sorted_results=result.copy().sort()
tops=sorted_results[0:5]
indexes=[]
for top in tops:
    indexes.append(sorted_results.index(top))
for i in range(5):
    print("No.{} guess:\t{}\t with probability {}.".format(str(i),labels[indexes[i]],tops[i]))
