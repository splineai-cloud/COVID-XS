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
echo "############################################################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for Pneumonia model1"
echo "############################################################################################"
# convert Keras model into TF inference graph
python code/PnKeras2TF.py -n Pnem1 2>&1 | tee rpt/Pneumonia/2_Pnkeras2TF_graph_conversion_Pnem1.log

echo " "
echo "#################################################################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for Pneumonia model2"
echo "#################################################################################################"
python code/PnKeras2TF.py -n Pnem2 2>&1 | tee rpt/Pneumonia/2_Pnkeras2TF_graph_conversion_Pnem2.log

echo " "
echo "############################################################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for Pneumonia model3"
echo "############################################################################################"
# convert Keras model into TF inference graph
python code/PnKeras2TF.py -n Pnem3 2>&1 | tee rpt/Pneumonia/2_Pnkeras2TF_graph_conversion_Pnem3.log

echo " "
echo "#################################################################################################"
echo "KERAS to TENSORFLOW GRAPH CONVERSION for Pneumonia model4"
echo "#################################################################################################"
python code/PnKeras2TF.py -n Pnem4 2>&1 | tee rpt/Pneumonia/2_Pnkeras2TF_graph_conversion_Pnem4.log



