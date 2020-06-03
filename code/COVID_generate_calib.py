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

import os
import shutil
#import cv2
import random

import argparse #DB
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d",  "--dataset", default="covid_data",  help="input dataset")

args = vars(ap.parse_args())
dataset_name = args["dataset"]

input_path = "./dataset/Pneumonia/" + dataset_name + "/"
print("Input Data Set Path: ", input_path) 
destpath = "./dataset/Pneumonia/calib_covid/"

calib_file = destpath + "calib_list.txt"

print("Calib File: ", calib_file) 

fcalib = open(calib_file ,"w+") 

lib_lims=[600, 400, 100] # Selecting 600 NORMAL, 400 PNEUMONIA and 100 COVID images
i = 0 
calib_list = list()

for cond in ['/NORMAL/', '/PNEUMONIA/', '/COVID/']:
    listimg = os.listdir(input_path + 'train' + cond)
    random.shuffle(listimg)
    nlims = lib_lims[i]
    listimg1 = listimg[0:nlims]
    i = i + 1
    for img in listimg1:
        impath = input_path+'train'+cond+img
        
        if cond=='/NORMAL/':
            calib_list.append("NORMAL_" + img + "\n")
            shutil.copyfile(impath, destpath+"NORMAL_" +img) 
        elif cond=='/PNEUMONIA/':
            calib_list.append("PNEUMONIA_" + img + "\n")
            shutil.copyfile(impath, destpath+"PNEUMONIA_" +img) 
        elif cond=='/COVID/':
            calib_list.append("COVID_" + img + "\n")
            shutil.copyfile(impath, destpath+"COVID_" +img) 
        else:
            print("ERROR: Invalid option") 

random.shuffle(calib_list)
for item in calib_list:
    fcalib.write(item) 


fcalib.close() 


