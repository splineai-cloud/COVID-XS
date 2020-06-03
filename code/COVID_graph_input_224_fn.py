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

import cv2
import os
import numpy as np
from keras.preprocessing.image import img_to_array


calib_image_dir  = "./../dataset/Pneumonia/calib_covid/"
calib_image_list = calib_image_dir +  "calib_list.txt"

print("CALIB DIR ", calib_image_dir)

calib_batch_size = 50
img_dims = 224

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]

def resize_shortest_edge(image, size):
  H, W = image.shape[:2]
  if H >= W:
    nW = size
    nH = int(float(H)/W * size)
  else:
    nH = size
    nW = int(float(W)/H * size)
  return cv2.resize(image,(nW,nH))

def mean_image_subtraction(image, means):
  B, G, R = cv2.split(image)
  B = B - means[0]
  G = G - means[1]
  R = R - means[2]
  image = cv2.merge([R, G, B])
  return image

def BGR2RGB(image):
  B, G, R = cv2.split(image)
  image = cv2.merge([R, G, B])
  return image

def central_crop(image, crop_height, crop_width):
  image_height = image.shape[0]
  image_width = image.shape[1]
  offset_height = (image_height - crop_height) // 2
  offset_width = (image_width - crop_width) // 2
  return image[offset_height:offset_height + crop_height, offset_width:
               offset_width + crop_width, :]

def normalize(image):
  image=image/255.0
  image=image-0.5
  image=image*2
  return image


def calib_input(iter):
    images = []
    line = open(calib_image_list).readlines()
    for index in range(0, calib_batch_size):
        curline = line[iter * calib_batch_size + index]
        calib_image_name = curline.strip()
        image = cv2.imread(calib_image_dir + calib_image_name)  
        image = cv2.resize(image, (img_dims, img_dims), 0, 0, cv2.INTER_LINEAR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = np.dstack([image])
        image = img_to_array(image, data_format=None)
        image = image/255.0     #Normalization used in Model training
        images.append(image)
    return {"input_1": images}

#######################################################

def main():
  calib_input(20)


if __name__ == "__main__":
    main()
