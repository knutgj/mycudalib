#run cmake ../src/ in ../build/, then make (evt. move created lib to ../) 
cmake_minimum_required (VERSION 2.6) 

set(CMAKE_MODULE_PATH "/CMake/cuda")
find_package(Cuda)

#cuda_add_executable(executable_name source1.cpp source2.cpp source3.cu)
#CUDA_ADD_LIBRARY(mylib SHARED file1.cu file2.cu file3.cu my_cp_file.cc ...)
CUDA_ADD_LIBRARY(mylib SHARED mycudalib.cu )
