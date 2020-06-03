'''
## © Copyright (C) 2020-2024 Xilinx, Inc
##
## Licensed under the Apache License, Version 2.0 (the "License"). You may
## not use this file except in compliance with the License. A copy of the
## License is located at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
## Developed by SplineAI (www.spline.ai) in Collaboration with Xilinx
'''

##################################################################
# Evaluation of frozen/quantized graph
#################################################################

import os
import sys
import glob
import argparse
import shutil
import tensorflow as tf
import numpy as np
import cv2
import gc # memory garbage collector #DB

from random import random
from random import shuffle #DB

import tensorflow.contrib.decent_q

from tensorflow.python.platform import gfile
from keras.preprocessing.image import img_to_array


#DB
DATAS_DIR = "./dataset/Pneumonia/rsna/"
TEST_DIR  = os.path.join(DATAS_DIR, "test")
print("\neval_graph runs from ", DATAS_DIR)


def graph_eval(input_graph_def, img_height, num_class, input_node, output_node):

    #Reading image paths
    print("Image Dim: ", img_height) ;
    print("Num_Class: ", num_class) ;

    img_dims = img_height

 #Data generation objects
    input_path = TEST_DIR
    n_normal = 0 
    n_pneumonia = 0
    n_covid = 0
    x_test = list()
    y_test = list()
    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + cond)):
            impath = input_path+cond+img 
            imga = cv2.imread(impath) #, cv2.COLOR_BGR2GRAY)
            imga = cv2.resize(imga, (img_dims, img_dims)) 
            imga = np.dstack([imga])
            if cond=='/NORMAL/':
                label = 0
                n_normal = n_normal + 1
            else: 
                label = 1 
                n_pneumonia = n_pneumonia + 1
                
            x_test.append(imga)
            y_test.append(label) 
    
    test_np = list(zip(x_test, y_test))
    shuffle(test_np)  
    x_test, y_test= zip(*test_np)
    x_test  = np.asarray(x_test)/255.0

    len_xtest = len(x_test)
    print("INFO: test #(NORMAL, PNEUMONIA) = ({}, {})".format(n_normal, n_pneumonia))
    
    batch_size = 64
    nrpts = len_xtest//batch_size 
    NUMEL = nrpts*batch_size
    print("Number of Images:", NUMEL)    

    x_test = np.reshape(x_test, [-1, img_height, img_height, 3])
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_class)
    print(x_test.shape)
    print(y_test.shape)

    tf.import_graph_def(input_graph_def,name = '')

    # Get input placeholders & tensors
    images_in = tf.compat.v1.get_default_graph().get_tensor_by_name(input_node+':0')
    print("Images_in: ", input_node+':0') 
    labels = tf.compat.v1.placeholder(tf.int32,shape = [None, num_class])

    # get output tensors
    logits = tf.compat.v1.get_default_graph().get_tensor_by_name(output_node+':0')

    # top 5 and top 1 accuracy
    in_top1 = tf.nn.in_top_k(predictions=logits, targets=tf.argmax(labels, 1), k=1)
    top1_acc = tf.reduce_mean(tf.cast(in_top1, tf.float32))
    print("INFO: top1_acc: " , top1_acc) ;
    # Create the Computational graph
    sum_acc = 0 
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initializers.global_variables()) 
        for i in range(nrpts):
            a1 = i*batch_size 
            b1 = a1 + batch_size 
            feed_dict={images_in: x_test[a1:b1], labels: y_test[a1:b1]}
            t1_acc = sess.run([top1_acc], feed_dict)
            print (' Top 1 accuracy with validation set: ', t1_acc)
            sum_acc = sum_acc + t1_acc[0]
    print ("Average accuracy = ", sum_acc/nrpts) 
    print ('FINISHED!')
    return


def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.io.gfile.GFile(FLAGS.graph, "rb").read())
    graph_eval(input_graph_def, FLAGS.height, FLAGS.class_num, FLAGS.input_node, FLAGS.output_node)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str,
                        default='./freeze/frozen_graph.pb',
                        help='graph file (.pb) to be evaluated.')
    parser.add_argument('--input_node', type=str,
                        default='input_1',
                        help='input node.')
    parser.add_argument('--output_node', type=str,
                        default='dense_out/Softmax',
                        help='output node.')
    parser.add_argument('--class_num', type=int,
                        default=2,
                        help='number of classes.')
    parser.add_argument('--height', type=int,
                        default=224,
                        help='Height or Width of Image.')
    parser.add_argument('--gpu', type=str,
                        default='0',
                        help='gpu device id.')

    FLAGS, unparsed = parser.parse_known_args()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

