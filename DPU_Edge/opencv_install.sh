#!/bin/bash

# References
# https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html

set -x
set -e

if [ "$1" == "a1" ]; then
    threads=16
elif [ "$1" == "u96" ]; then
    threads=4
elif [ "$1" == "zcu104" ]; then
    threads=4
elif [ "$1" == "install" ]; then
	make install
	ldconfig
else
    echo '$0 <a1|u96|install>'
    exit 0
fi 


# if needed
# sudo apt-get update
# sudo apt-get -y install build-essential cmake zip python3-opencv python3-dev python3-numpy

opencv_version=4.2.0
wget -O opencv.zip https://github.com/opencv/opencv/archive/${opencv_version}.zip  
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${opencv_version}.zip

unzip opencv.zip 
unzip opencv_contrib.zip

mkdir opencv-${opencv_version}/build
cd opencv-${opencv_version}/build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D BUILD_WITH_DEBUG_INFO=OFF \
	-D BUILD_DOCS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D BUILD_TESTS=ON \
	-D BUILD_opencv_ts=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D ENABLE_NEON=ON \
	-D WITH_LIBV4L=ON \
	-D WITH_GSTREAMER=ON \
	-D BUILD_opencv_dnn=OFF \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${opencv_version}/modules \
        ../

make -j$threads
cd ../..

