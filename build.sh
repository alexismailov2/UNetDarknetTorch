torch_path=$1
opencv_path=$2

if [ -z "$1" ]; then
    echo "Parameter path to libtorch was not set";

if [ -d "./downloads/libtorch" ]; then
    echo "There is libtorch unpacked just use it"
else
    mkdir -p downloads
    cd downloads
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip
    cd ..
fi
  torch_path=./downloads/libtorch
else
  echo "Just start building";
fi

cmake . -Bbuild_host -DCUSTOM_TORCH_BUILD_PATH=${torch_path} -DCUSTOM_OPENCV_BUILD_PATH=${opencv_path} && cmake --build build_host --target all