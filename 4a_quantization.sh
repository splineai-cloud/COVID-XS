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

# activate DECENT_Q Python3.6 virtual environment

cd ./code

# run quantization
echo " "
echo "##########################################################################"
echo "QUANTIZE Pneumonia Model1"
echo "##########################################################################"

vai_q_tensorflow  quantize \
	 --input_frozen_graph ../freeze/Pneumonia/Pnem1/frozen_graph.pb \
	 --input_nodes input_1 \
	 --input_shapes ?,150,150,3 \
	 --output_nodes dense_out/Softmax \
	 --output_dir ../quantized_results/Pneumonia/Pnem1/ \
	 --method 1 \
	 --input_fn Pneumonia_graph_input_150_fn.calib_input \
	 --calib_iter 40 \
	 --gpu 0  2>&1 | tee ../rpt/Pneumonia/4a_quant_150_Pneumonia_Pnem1.log

echo " "
echo "##########################################################################"
echo "QUANTIZE Pneumonia Model2"
echo "##########################################################################"

vai_q_tensorflow quantize \
	 --input_frozen_graph ../freeze/Pneumonia/Pnem2/frozen_graph.pb \
	 --input_nodes input_1 \
	 --input_shapes ?,224,224,3 \
	 --output_nodes dense_out/Softmax \
	 --output_dir ../quantized_results/Pneumonia/Pnem2/ \
	 --method 1 \
	 --input_fn Pneumonia_graph_input_224_fn.calib_input \
	 --calib_iter 40 \
	 --gpu 0  2>&1 | tee ../rpt/Pneumonia/4a_quant_224_Pneumonia_Pnem2.log

echo " "
echo "##########################################################################"
echo "QUANTIZE Pneumonia Model3"
echo "##########################################################################"

vai_q_tensorflow  quantize \
	 --input_frozen_graph ../freeze/Pneumonia/Pnem3/frozen_graph.pb \
	 --input_nodes input_1 \
	 --input_shapes ?,150,150,3 \
	 --output_nodes dense_out/Softmax \
	 --output_dir ../quantized_results/Pneumonia/Pnem3/ \
	 --method 1 \
	 --input_fn COVID_graph_input_150_fn.calib_input \
	 --calib_iter 22 \
	 --gpu 0  2>&1 | tee ../rpt/Pneumonia/4a_quant_150_Pneumonia_Pnem3.log

echo " "
echo "##########################################################################"
echo "QUANTIZE Pneumonia Model4"
echo "##########################################################################"

vai_q_tensorflow quantize \
	 --input_frozen_graph ../freeze/Pneumonia/Pnem4/frozen_graph.pb \
	 --input_nodes input_1 \
	 --input_shapes ?,224,224,3 \
	 --output_nodes dense_out/Softmax \
	 --output_dir ../quantized_results/Pneumonia/Pnem4/ \
	 --method 1 \
	 --input_fn COVID_graph_input_224_fn.calib_input \
	 --calib_iter 22 \
	 --gpu 0  2>&1 | tee ../rpt/Pneumonia/4a_quant_224_Pneumonia_Pnem4.log

# run quantization

cd ..

