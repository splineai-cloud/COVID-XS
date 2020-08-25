import json
import logging
import platform
import sys
import time
import os
from ctypes import *  
import numpy as np
import cv2
import greengrasssdk


# Setup logging to stdout
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Creating a greengrass core sdk client
client = greengrasssdk.client("iot-data")

# Counter to keep track of invocations of the function_handler
my_counter = 0

# Retrieving platform information to send from Greengrass Core
my_platform = platform.platform()

AWS_ACCESS_KEY_ID = <aws-access-key-id>
AWS_SECRET_ACCESS_KEY = <aws-secret-access-key>
region="us-west-2"

KERNEL_CONV = os.environ['KERNEL_CONV']                         #"Pnem1_0"  
KERNEL_CONV_INPUT = os.environ['KERNEL_CONV_INPUT']             #"conv2d_1_convolution"
KERNEL_FC_OUTPUT = os.environ['KERNEL_FC_OUTPUT']               #"dense_out_MatMul" 
IMG_DIMS=   int(os.environ['IMG_DIMS'])                         #150
LABEL_FILE=    os.environ['LABEL_FILE']                         #"pnlabels.txt"
DPU_DIR=    os.environ['DPU_DIR']                               #"/dpu_ml_models/"
BUCKET='imgxrays'
BUCKETDPU = 'covidxs'
LOCAL_IMG_DIR= DPU_DIR+ BUCKET

dpu_elf = "dpu_" + KERNEL_CONV +  ".elf" 
#dpu_elf = "dpu_Pnem1_0.elf"
label_file= LABEL_FILE


elf_file= DPU_DIR + dpu_elf
lable_path= DPU_DIR + label_file
image_folder = DPU_DIR + BUCKET + '/'
bit_path = DPU_DIR + "dpu.bit"
elf_path =  DPU_DIR + dpu_elf
label_path= DPU_DIR + LABEL_FILE

session = boto3.session.Session(region_name=region) 
s3_client = session.client('s3', 
     config=boto3.session.Config(signature_version='s3v4'),
     aws_access_key_id=AWS_ACCESS_KEY_ID,
     aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

#Accesing the bit file from the S3
s3_client.download_file(BUCKETDPU, "dpu.bit",    DPU_DIR + "dpu.bit")
s3_client.download_file(BUCKETDPU, "dpu.hwh",    DPU_DIR + "dpu.hwh")
s3_client.download_file(BUCKETDPU, "dpu.xclbin", DPU_DIR + "dpu.xclbin")
s3_client.download_file(BUCKETDPU, dpu_elf,      DPU_DIR + dpu_elf)
s3_client.download_file(BUCKETDPU, LABEL_FILE,   DPU_DIR + LABEL_FILE)

from pynq_dpu import DpuOverlay
overlay = DpuOverlay(bit_path)
overlay.load_model(elf_path)

from dnndk import n2cube
from pynq_dpu import dputils 
 

n2cube.dpuOpen()
kernel = n2cube.dpuLoadKernel(KERNEL_CONV) 

with open(lable_path, "r") as f:
    lines = f.readlines()
slabels = lines

def predict_label(imfile):
    task = n2cube.dpuCreateTask(kernel, 0)

    # Set client to get file from S3 
    s3_client.download_file(BUCKET, imfile, image_folder + imfile)
    img_obj = os.path.join(image_folder, imfile)
    
    #To get it from local path
    #img_file = os.path.join(image_folder, imfile)
    
    img = cv2.imread(img_obj) 
    img = cv2.resize(img, (IMG_DIMS, IMG_DIMS))
    img = img.astype(np.float32)
    img = (img/255.0) 
        
    """Get input Tensor"""
    tensor = n2cube.dpuGetInputTensor(task, KERNEL_CONV_INPUT)
    input_len = n2cube.dpuGetInputTensorSize(task, KERNEL_CONV_INPUT)   
        
    """Set input Tesor"""
    n2cube.dpuSetInputTensorInHWCFP32(task, KERNEL_CONV_INPUT, img, input_len)

    """Model run on DPU"""
    n2cube.dpuRunTask(task)
        
    """Get the output tensor size from FC output"""
    size = n2cube.dpuGetOutputTensorSize(task, KERNEL_FC_OUTPUT)

    """Get the output tensor channel from FC output"""
    channel = n2cube.dpuGetOutputTensorChannel(task, KERNEL_FC_OUTPUT)

    softmax = np.zeros(size,dtype=np.float32)

    """Get FC result"""
    conf = n2cube.dpuGetOutputTensorAddress(task, KERNEL_FC_OUTPUT)

    """Get output scale of FC"""
    outputScale = n2cube.dpuGetOutputTensorScale(task, KERNEL_FC_OUTPUT)

    """Run softmax"""
    softmax = n2cube.dpuRunSoftmax(conf, channel, size // channel, outputScale)
     
    #print("softmax =", softmax)

    n2cube.dpuDestroyTask(task)
    
    return slabels[np.argmax(softmax)]


#listimage = [i for i in os.listdir(image_folder) if "NORMAL" in i or "PNEUMONIA" in i ]
#imfile = listimage[0]
imfile = 'COVID_COVID-00006.jpg' 
path_imfile = DPU_DIR
#output=predict_label(imfile)


TOPIC="DP28/i28"
def function_handler(event, context):
    global my_counter
    global slabels
    global imfile 
    global image_folder
    my_counter = my_counter + 1
    imfile_ev = imfile
    output = ""
    
    try:
        if not my_platform:
            client.publish(
                topic=TOPIC,
                queueFullPolicy="AllOrException",
                payload=json.dumps(
                    {"message": "PN! Sent from Greengrass Core.  Invocation Count: {}".format(my_counter)}
                ),
            )
        else: 
            if ('imgfile' in event.keys()):
                imfile_ev = event['imgfile'] 
                
            output=predict_label(imfile_ev)
            client.publish(
                topic=TOPIC,
                queueFullPolicy="AllOrException",
                payload=json.dumps(
                    {
                        "message": "PN! Sent from Greengrass Core running on platform: {}.".format(my_platform)
                                   + " Invocation Count: {}".format(my_counter) 
                                   + " Image = {}".format(imfile_ev) + " Output= {}".format(output)
                    }
                ),
            )
    except Exception as e:
        logger.error("Failed to publish message: " + repr(e))
    #time.sleep(20)
    response = {
        "statusCode": 200,
        "body": json.dumps({" Image = {}".format(imfile) : " Output= {}".format(output)})
    }
    return response





