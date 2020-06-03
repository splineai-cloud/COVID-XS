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

'''
 - "How to convert trained Keras model to a single TensorFlow .pb file and make prediction" available from https://github.com/Tony607/keras-tf-pb
'''

# USAGE
# python PnKeras2TFy -c Pnem1 -d Pneumonia-xray

import os
import sys
import shutil
from keras import backend as K
#from tensorflow.keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf

import argparse #DB
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n",  "--network", default="Pnem1", help="input CNN")
args = vars(ap.parse_args())
cnn_name = args["network"]


##############################################
# Set up directories
##############################################
KERAS_MODEL_DIR = "./keras_model/Pneumonia/"   #   cfg.KERAS_MODEL_DIR #DB

WEIGHTS_DIR = os.path.join(KERAS_MODEL_DIR, cnn_name)

CHKPT_MODEL_DIR = "./keras_model/Pneumonia/"


# set learning phase for no training: This line must be executed before loading Keras model
K.set_learning_phase(0)

chkpt_model = os.path.join(WEIGHTS_DIR,"best_chkpt.hdf5")
print(chkpt_model)
# load weights & architecture into new model
model = load_model(os.path.join(WEIGHTS_DIR,"best_chkpt.hdf5"))

#print the CNN structure
model.summary()

# make list of output node names
output_names=[out.op.name for out in model.outputs]

# set up tensorflow saver object
saver = tf.train.Saver()

# fetch the tensorflow session using the Keras backend
sess = K.get_session()

# get the tensorflow session graph
graph_def = sess.graph.as_graph_def()


# Check the input and output name
print ("\n TF input node name:")
print(model.inputs)
print ("\n TF output node name:")
print(model.outputs)

#=====================================
import shutil
save_path = os.path.join(CHKPT_MODEL_DIR, cnn_name) 
try: 
    shutil.rmtree(save_path)
except:
    print("INFO: remove non empty dir: " , save_path)

os.makedirs(save_path)
print("INFO: CNNN Path: ", save_path)
#=====================================

# write out tensorflow checkpoint & inference graph 
save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, cnn_name, "float_model.ckpt"))
#Save Checkpoint as pb
tf.train.write_graph(graph_def, os.path.join(CHKPT_MODEL_DIR, cnn_name), "infer_graph.pb", as_text=False)


print ("\nFINISHED CREATING TF FILES\n")
