# UNetTorchTrain
Project for training unet with libtorch in darknet format.

#To avoid spending time on reading next one just try to run build.sh script for building the project:
```
./build.sh
```

# And Run test after run.sh
```
run.sh
```

# Steps for getting successful build:
##1. Clone this project
Hope you know how to do that...

##2. First of all download libtorch for your system:
Consider that there is two different versions with CUDA support and without.
You should use needed version according to your hardware support CUDA and CUDA tool kit installed previous.
This reference describe how to install CUDA and all info about needed hardware for it.

After you will be ready to decide which version you will need download one of them:
- With CUDA - https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip
- CPU only - https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip

##3. Unpack selected libtorch wherever you want into your system
Hope you know how to do that, do not afraid project settings does not depend of some path, and can be 
flexible be set to needed distributive without patching your system by unneeded dependencies(definitely for now).

##4. Install or download binaries or even build OpenCV for your system if you need
I suggest to use 4.4.0 version because some of new DNN extentions was created there and it will be needed
for testing results of this training UNet project.
Also path can be flexible passed to the project without installing it to the system.

In addition in case CUDA supported by your system you can build OpenCV with CUDA support too:
- https://cuda-chen.github.io/image%20processing/programming/2020/02/22/build-opencv-dnn-module-with-nvidia-gpu-support-on-ubuntu-1804.html

##6. Did I say that needed compiler which supports c++17?
Yeah it is true because in my project I have used std::filesystem(do not arguing me on that)

##7. Did I say that need CMake 3.14 or higer?
You can change it to lower version but I suggest to install the most newest version which there is.
You should understand that XCode, and Visual Studio changed every year and add new features and CMake 
project gets updates for it. It will safe your time of unneded finding what will going wrong if some problem 
will be occurred on different platform.

##8. May be will be some in addition soon

##9. Generate project for needed build system(read all variants before you start)
The most flexible way to do it with next cmake command(if OpenCV and libtorch installed to your system):

- Debug version(by default -DCMAKE_BUILD_TYPE=Debug):
```
cmake -Bbuild_host
```

- Release version(change build type with -DCMAKE_BUILD_TYPE=Release):
```
cmake -Bbuild_host -DCMAKE_BUILD_TYPE=Release
```

- But as I said previous it is not necessary to be installed OpenCV and libtorch. Instead you can pass 
pathes to the CMake with your own locations for that.
``` 
cmake -Bbuild_host \
-DCUSTOM_TORCH_BUILD_PATH="<path-to-the-root-folder-of-unpacked-libtorch>" \
-DCUSTOM_OPENCV_BUILD_PATH="<path-to-the-root-folder-of-your-opencv>" \
```

Note:
build_host - it is a name of folder where CMake will create and put generated project and needed 
artifacts for build.

##10. If it will be finished ok(Tested only on Linux now, on MacOS and Windows will be tested soon)
Start building with asking cmake to delegate build to generated project(not only make).
```
cmake --build build_host --build --target all
```

##11. If it will be ok you can use command line and run produced executable "train_unet_darknet2d" with --help
```
./build_host/train_unet_2d --help
```

##12. Enjoy

