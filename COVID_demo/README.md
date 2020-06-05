### Install OpenCV >= 4.0 to view the Heatmaps of the inference output from the Deep Learning models. 
[Vitis AI 1.1](https://developer.xilinx.com/en/get-started/ai.html) includes the native OpenCV 3.4. 
This version of OpenCV will not be enough for displaying the Heatmap. 

The steps to install the OpenCV >=4.0 is given below:

```bash
./docker_run.sh xilinx/vitis-ai-gpu:latest # enter into the docker VAI tools image
sudo su # you must be root
conda activate vitis-ai-tensorflow # as root, enter into Vitis AI TF (anaconda-based) virtual environment
conda uninstall OpenCV 
conda install -c conda-forge opencv   #This will install OpenCV >= 4.0
conda deactivate
exit # to exit from root
conda activate vitis-ai-tensorflow # as normal user, enter into Vitis AI TF (anaconda-based) virtual environment
```
Note that if you exit from the current Docker Vitis AI tools image you will lose all the installed packages, so to save all changes in a new docker image open a new terminal and run the following commands:

```bash
sudo docker ps -l # To get the Docker CONTAINER ID
```
you will see the following text (the container ID might have a different number):
```text
CONTAINER ID        IMAGE                        COMMAND                CREATED             STATUS              PORTS               NAMES
7f865e2ddd33        xilinx/vitis-ai-gpu:latest   "/etc/login.sh bash"   something           something                          ecstatic_kirch
```
now save the modified docker image:
```bash
sudo docker commit -m "new image: added pydicom and xlrd used in data generation" \
        7f865e2ddd33        xilinx/vitis-ai-gpu:latest
```
