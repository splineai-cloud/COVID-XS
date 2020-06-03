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

echo " "
echo "##########################################################################"
echo "EVALUATE FROZEN GRAPH Pnem1"
echo "##########################################################################"
python code/Pneumonia_eval_graph.py --dataset Pneumonia --graph ./freeze/Pneumonia/Pnem1/frozen_graph.pb --height 150 --input_node input_1 --output_node dense_out/Softmax --gpu 0  2>&1 | tee rpt/Pneumonia/3b_evaluate_frozen_graph_Pnem1.log

echo " "
echo "##########################################################################"
echo "EVALUATE FROZEN GRAPH Pnem2"
echo "##########################################################################"
python code/Pneumonia_eval_graph.py --dataset Pneumonia --graph ./freeze/Pneumonia/Pnem2/frozen_graph.pb --height 224 --input_node input_1 --output_node dense_out/Softmax --gpu 0  2>&1 | tee rpt/Pneumonia/3b_evaluate_frozen_graph_Pnem2.log

echo " "
echo "##########################################################################"
echo "EVALUATE FROZEN GRAPH Pnem3"
echo "##########################################################################"
python code/COVID_eval_graph.py --dataset Pneumonia --graph ./freeze/Pneumonia/Pnem3/frozen_graph.pb --height 150 --input_node input_1 --output_node dense_out/Softmax --gpu 0  2>&1 | tee rpt/Pneumonia/3b_evaluate_frozen_graph_Pnem3.log

echo " "
echo "##########################################################################"
echo "EVALUATE FROZEN GRAPH Pnem4"
echo "##########################################################################"
python code/COVID_eval_graph.py --dataset Pneumonia --graph ./freeze/Pneumonia/Pnem4/frozen_graph.pb --height 224 --input_node input_1 --output_node dense_out/Softmax --gpu 0  2>&1 | tee rpt/Pneumonia/3b_evaluate_frozen_graph_Pnem4.log


