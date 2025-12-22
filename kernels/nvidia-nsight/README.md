# 针对 RTX 3060 (Ampere 架构)
export TORCH_CUDA_ARCH_LIST="8.6"
```
# compile relu.cu with debug info
nvcc -arch=sm_86 -o relu.bin --generate-line-info -g relu.cu
nvcc -arch=sm_86 -o elementwise.bin --generate-line-info -g elementwise.cu

# use nsys cli to export timeline profile
nsys profile --stats=true -t cuda,osrt,nvtx -o relu.prof -f true relu.bin
nsys profile --stats=true -t cuda,osrt,nvtx -o elementwise.prof -f true elementwise.bin

# use ncu cli to export kernel profile (include SASS/PTX)
# ERROR: 权限不足, 尚未解决
ncu -o relu.prof -f relu.bin
ncu -o elementwise.prof -f elementwise.bin

# run bin
./relu.bin
./elementwise.bin
```
