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
echo "TRAIN & EVAL Pneumonia Model 1 2"
echo "##########################################################################"
python code/train_Pneumonia.py --network Pnem1 --weights ./keras_model/Pneumonia/Pnem1 --height 150 --epochs 300 --batch_size 64 2>&1 | tee rpt/Pneumonia/1_train_Pnem1.log
python code/train_Pneumonia.py --network Pnem2 --weights ./keras_model/Pneumonia/Pnem2 --height 224 --epochs 300 --batch_size 64 2>&1 | tee rpt/Pneumonia/1_train_Pnem2.log


