# DiskTracer-GPU

CUDA C Implementation of DiskTracer.jl



## GPU architectures

This code has been tested against the following GPUs.

## Nvidia Tesla K80

    Compute capability 3.7
    Max # grids / device : 32
    Max # of threads / block: 1024
    Warp size: 32
    Max # of resident blocks / multiprocessor: 16
    Max # of resident warps / multiprocessor: 64
    Shared mem per multiprocessor : 112KB
    Shared mem per thread block : 48KB

    CUDA cores	4992 ( 2496 per GPU)


## GeForce GTX 980

    CUDA Driver Version / Runtime Version          9.1 / 9.1
    CUDA Capability Major/Minor version number:    5.2
    Total amount of global memory:                 4038 MBytes (4233887744 bytes)
    (16) Multiprocessors, (128) CUDA Cores/MP:     2048 CUDA Cores
    GPU Max Clock rate:                            1278 MHz (1.28 GHz)
    Memory Clock rate:                             3505 Mhz
    Memory Bus Width:                              256-bit
    L2 Cache Size:                                 2097152 bytes
    Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
    Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
    Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
    Total amount of constant memory:               65536 bytes
    Total amount of shared memory per block:       49152 bytes
    Total number of registers available per block: 65536
    Warp size:                                     32
    Maximum number of threads per multiprocessor:  2048
    Maximum number of threads per block:           1024
    Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
    Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
    Maximum memory pitch:                          2147483647 bytes
    Texture alignment:                             512 bytes
    Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
    Run time limit on kernels:                     Yes
    Integrated GPU sharing Host Memory:            No
    Support host page-locked memory mapping:       Yes
    Alignment requirement for Surfaces:            Yes
    Device has ECC support:                        Disabled
    Device supports Unified Addressing (UVA):      Yes
    Supports Cooperative Kernel Launch:            No
    Supports MultiDevice Co-op Kernel Launch:      No
    Device PCI Domain ID / Bus ID / location ID:   0 / 4 / 0
    Compute Mode:
       < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
