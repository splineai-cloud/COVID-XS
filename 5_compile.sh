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


# delete previous results
rm -rf ./compile/Pneumonia/*

mkdir -p ./compile/Pneumonia/Pnem1
mkdir -p ./compile/Pneumonia/Pnem2
mkdir -p ./compile/Pneumonia/Pnem3
mkdir -p ./compile/Pneumonia/Pnem4


# Compile
echo " "
echo "##########################################################################"
echo "COMPILE WITH Vitis AI: Pnem1 "
echo "##########################################################################"
vai_c_tensorflow \
       --frozen_pb=./quantized_results/Pneumonia/Pnem1/deploy_model.pb \
       --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU104/ZCU104.json \
       --output_dir=./compile/Pneumonia/Pnem1 \
       --net_name=Pnem1 \
       --options  "{'mode':'normal'}" \
       2>&1 | tee rpt/Pneumonia/5_vai_compile_Pnem1.log
echo " "
echo "##########################################################################"
echo "COMPILATION COMPLETED  on Pnem1"
echo "##########################################################################"
echo " "

# Compile
echo " "
echo "##########################################################################"
echo "COMPILE WITH Vitis AI: Pnem2 "
echo "##########################################################################"
vai_c_tensorflow \
       --frozen_pb=./quantized_results/Pneumonia/Pnem2/deploy_model.pb \
       --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU104/ZCU104.json \
       --output_dir=./compile/Pneumonia/Pnem2 \
       --net_name=Pnem2 \
       --options  "{'mode':'normal'}" \
       2>&1 | tee rpt/Pneumonia/5_vai_compile_Pnem2.log
echo " "
echo "##########################################################################"
echo "COMPILATION COMPLETED  on Pnem2"
echo "##########################################################################"
echo " "

# Compile
echo " "
echo "##########################################################################"
echo "COMPILE WITH Vitis AI: Pnem3 "
echo "##########################################################################"
vai_c_tensorflow \
       --frozen_pb=./quantized_results/Pneumonia/Pnem3/deploy_model.pb \
       --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU104/ZCU104.json \
       --output_dir=./compile/Pneumonia/Pnem3 \
       --net_name=Pnem3 \
       --options  "{'mode':'normal'}" \
       2>&1 | tee rpt/Pneumonia/5_vai_compile_Pnem3.log
echo " "
echo "##########################################################################"
echo "COMPILATION COMPLETED  on Pnem3"
echo "##########################################################################"
echo " "

# Compile
echo " "
echo "##########################################################################"
echo "COMPILE WITH Vitis AI: Pnem4 "
echo "##########################################################################"
vai_c_tensorflow \
       --frozen_pb=./quantized_results/Pneumonia/Pnem4/deploy_model.pb \
       --arch /opt/vitis_ai/compiler/arch/dpuv2/ZCU104/ZCU104.json \
       --output_dir=./compile/Pneumonia/Pnem4 \
       --net_name=Pnem4 \
       --options  "{'mode':'normal'}" \
       2>&1 | tee rpt/Pneumonia/5_vai_compile_Pnem4.log
echo " "
echo "##########################################################################"
echo "COMPILATION COMPLETED  on Pnem4"
echo "##########################################################################"
echo " "



