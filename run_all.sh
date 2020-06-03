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


: '
# clean up previous log files
rm -f *.log
if [ ! rpt ]; then
    mkdir rpt
else
    rm ./rpt/*.log
fi
'

: '

#source 1_Pneumonia_train.sh
#source 1_COVID_train.sh
source 2_keras2tf.sh
source 3a_freeze.sh
source 3b_evaluate_frozen_graph.sh
source 4a_quantization.sh
source 4b_evaluate_quantized_graph.sh
source 5_compile.sh
source run_all.sh

