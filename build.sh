torch_path=$2
opencv_path=$1

if [ -z "$2" ]; then
    echo "Parameter path to libtorch was not set";

if [ -d "./downloads/libtorch" ]; then
    echo "There is libtorch unpacked just use it"
else
    mkdir -p downloads
    cd downloads
    #wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip
    #wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip
    #wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip
    #unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111.zip
    #unzip libtorch-macos-1.6.0.zip
    cd ..
fi
  torch_path=./downloads/libtorch
else
  echo "Just start building";
fi

cmake . -Bbuild_host -DCUSTOM_TORCH_BUILD_PATH=${torch_path} -DCUSTOM_OPENCV_BUILD_PATH=${opencv_path} && cmake --build build_host --target all