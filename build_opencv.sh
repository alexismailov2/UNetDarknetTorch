cd downloads
#git clone https://github.com/opencv/opencv.git
#cd opencv
#git checkout 4.5.3
#cd ..

#git clone https://github.com/opencv/opencv_contrib.git
#cd opencv_contrib
#git checkout 4.5.3
#cd ..

cd opencv
rm -rf build
mkdir build
cd build

#sudo apt install libjpeg-dev libpng-dev libtiff-dev
#sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
#sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
#sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
#sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
#sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev

# Cameras programming interface
#sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
#cd /usr/include/linux
#sudo ln -s -f ../libv4l1-videodev.h videodev.h
#cd ~

#sudo apt-get install libgtk-3-dev

#sudo apt-get install libtbb-dev

#sudo apt-get install libatlas-base-dev gfortran

cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/home/alex/WORK/smile/UNetDarknetTorch/downloads/opencv_4_5_2_cuda_release \
 -D INSTALL_C_EXAMPLES=OFF \
 -D WITH_TBB=ON \
 -D WITH_CUDA=ON \
 -D BUILD_opencv_cudacodec=OFF \
 -D ENABLE_FAST_MATH=1 \
 -D CUDA_FAST_MATH=1 \
 -D WITH_CUBLAS=1 \
 -D WITH_CUDNN=ON \
 -D OPENCV_DNN_CUDA=ON \
 -D CUDA_ARCH_BIN=8.6 \
 -D WITH_V4L=ON \
 -D WITH_QT=OFF \
 -D WITH_OPENGL=ON \
 -D WITH_GSTREAMER=ON \
 -D OPENCV_GENERATE_PKGCONFIG=ON \
 -D OPENCV_PC_FILE_NAME=opencv.pc \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
 -D BUILD_EXAMPLES=OFF \
 -D BUILD_PERF_TESTS=OFF \
 -D BUILD_TESTS=OFF ..
make -j4
