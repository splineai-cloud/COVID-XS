#!/bin/bash

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

# freeze trained grap
echo " "
echo "##########################################################################"
echo "FREEZE GRAPH of Pnem1"
echo "##########################################################################"
rm ./freeze/Pneumonia/Pnem1/*  # remove previous results
freeze_graph --input_graph=./keras_model/Pneumonia/Pnem1/infer_graph.pb \
             --input_checkpoint=./keras_model/Pneumonia/Pnem1/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/Pneumonia/Pnem1/frozen_graph.pb \
             --output_node_names=dense_out/Softmax \
             2>&1 | tee rpt/Pneumonia/3a_freeze_graph_Pnem1.log

# check possible input/output node names
echo " "
echo "##########################################################################"
echo "INSPECT FROZEN GRAPH of Pnem1"
echo "##########################################################################"
vai_q_tensorflow inspect --input_frozen_graph=./freeze/Pneumonia/Pnem1/frozen_graph.pb

echo " "
echo "##########################################################################"
echo "FREEZE GRAPH of Pnem2"
echo "##########################################################################"
rm ./freeze/Pneumonia/Pnem2/*  # remove previous results
freeze_graph --input_graph=./keras_model/Pneumonia/Pnem2/infer_graph.pb \
             --input_checkpoint=./keras_model/Pneumonia/Pnem2/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/Pneumonia/Pnem2/frozen_graph.pb \
             --output_node_names=dense_out/Softmax \
             2>&1 | tee rpt/Pneumonia/3a_freeze_graph_Pnem2.log

# check possible input/output node names

echo " "
echo "##########################################################################"
echo "INSPECT FROZEN GRAPH of Pnem2"
echo "##########################################################################"
vai_q_tensorflow inspect --input_frozen_graph=./freeze/Pneumonia/Pnem2/frozen_graph.pb

# freeze trained grap
echo " "
echo "##########################################################################"
echo "FREEZE GRAPH of Pnem3"
echo "##########################################################################"
rm ./freeze/Pneumonia/Pnem3/*  # remove previous results
freeze_graph --input_graph=./keras_model/Pneumonia/Pnem3/infer_graph.pb \
             --input_checkpoint=./keras_model/Pneumonia/Pnem3/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/Pneumonia/Pnem3/frozen_graph.pb \
             --output_node_names=dense_out/Softmax \
             2>&1 | tee rpt/Pneumonia/3a_freeze_graph_Pnem3.log

# check possible input/output node names
echo " "
echo "##########################################################################"
echo "INSPECT FROZEN GRAPH of Pnem3"
echo "##########################################################################"
vai_q_tensorflow inspect --input_frozen_graph=./freeze/Pneumonia/Pnem3/frozen_graph.pb

echo " "
echo "##########################################################################"
echo "FREEZE GRAPH of Pnem4"
echo "##########################################################################"
rm ./freeze/Pneumonia/Pnem4/*  # remove previous results
freeze_graph --input_graph=./keras_model/Pneumonia/Pnem4/infer_graph.pb \
             --input_checkpoint=./keras_model/Pneumonia/Pnem4/float_model.ckpt \
             --input_binary=true \
             --output_graph=./freeze/Pneumonia/Pnem4/frozen_graph.pb \
             --output_node_names=dense_out/Softmax \
             2>&1 | tee rpt/Pneumonia/3a_freeze_graph_Pnem4.log

# check possible input/output node names

echo " "
echo "##########################################################################"
echo "INSPECT FROZEN GRAPH of Pnem4"
echo "##########################################################################"
vai_q_tensorflow inspect --input_frozen_graph=./freeze/Pneumonia/Pnem4/frozen_graph.pb


