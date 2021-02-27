"cmake.exe" . ^
 -Bbuild_host_w ^
 -DCUSTOM_TORCH_BUILD_PATH=downloads/libtorch ^
 -DCUSTOM_OPENCV_BUILD_PATH=C:\OpenCV\opencv\build\x64\vc14\lib ^
 -DTIFF_LIBRARY=downloads/libtiff/lib ^
 -DTIFF_INCLUDE_DIR=downloads/libtiff/include

"cmake.exe" --build build_host_w --target ALL_BUILD

copy downloads\libtorch\lib\asmjit.dll build_host_w
copy downloads\libtorch\lib\c10.dll build_host_w
copy downloads\libtorch\lib\fbgemm.dll build_host_w
copy downloads\libtorch\lib\libiomp5md.dll build_host_w
copy downloads\libtorch\lib\torch_cpu.dll build_host_w
copy C:\OpenCV\opencv\build\x64\vc14\lib\opencv_world440d.dll build_host_w
