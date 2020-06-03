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
echo "EVALUATE QUANTIZED GRAPH of Pnem1 "
echo "##########################################################################"
#Pnem1
python code/Pneumonia_eval_graph.py --dataset Pneumonia --graph ./quantized_results/Pneumonia/Pnem1/quantize_eval_model.pb --height 150 --input_node input_1 --output_node dense_out/Softmax --gpu 0  2>&1 | tee rpt/Pneumonia/4b_evaluate_quantized_graph_Pnem1.log

echo " "
echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of Pnem2 "
echo "##########################################################################"
#Pnem2
python code/Pneumonia_eval_graph.py --dataset Pneumonia --graph ./quantized_results/Pneumonia/Pnem2/quantize_eval_model.pb --height 224 --input_node input_1 --output_node dense_out/Softmax --gpu 0 2>&1 | tee rpt/Pneumonia/4b_evaluate_quantized_graph_Pnem2.log

echo " "
echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of Pnem3 "
echo "##########################################################################"
#Pnem1
python code/COVID_eval_graph.py --dataset Pneumonia --graph ./quantized_results/Pneumonia/Pnem3/quantize_eval_model.pb --height 150 --input_node input_1 --output_node dense_out/Softmax --gpu 0  2>&1 | tee rpt/Pneumonia/4b_evaluate_quantized_graph_Pnem3.log

echo " "
echo "##########################################################################"
echo "EVALUATE QUANTIZED GRAPH of Pnem4 "
echo "##########################################################################"
#Pnem2
python code/COVID_eval_graph.py --dataset Pneumonia --graph ./quantized_results/Pneumonia/Pnem4/quantize_eval_model.pb --height 224 --input_node input_1 --output_node dense_out/Softmax --gpu 0 2>&1 | tee rpt/Pneumonia/4b_evaluate_quantized_graph_Pnem4.log

echo " "


