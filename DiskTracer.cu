#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


#include "constants.h"

// #include <helper_cuda.h>

// Define constants as constant memory.
// __constant__ double or something
// __device__ double


// 1 grid of N blocks, each with M threads
// grid

// GPI cruncher
// Total amount of constant memory:               65536 bytes
// Total amount of shared memory per block:       49152 bytes
// Total number of registers available per block: 65536
// Warp size:                                     32
// 16 multiprocessors
// Maximum number of threads per multiprocessor:  2048
// Maximum number of threads per block:           1024
// Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
// Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)

// no dimension of launch blocks may exceed 65,535
// We can have 1024 threads per block, so 65,535 * 1024 = 6.7e7
// So, we should be good for our problem if the image size is less than 1024 x 1024 pixels.
// 512 * 512 * 64 = 1.67e7
// 1024 * 1024 * 64 = 6.71e7

// If it exceeds this, then we're going to have to do multiple pixels per thread.

// image size n_pix
// The giant 3D image cube will be (vel, Y, X)
// And this is just written directly to (global) device memory


// I think it makes sense to have one dimension of the block be the frequency dimension,
// Then we have to break each (Y, X) image into sectors.

// Say we have 512 x 512 pixels in each (Y, X) image.
// Then, to break this into sectors

// If we're doing square blocks, then this can be 32 x 32 pixels
// Or, we could just do a column at a time, of 512 pixels or 1024 pixels.

// pixels per channel = n_pix * n_pix
// if we can have 1024 threads per block, then we

__global__ void
vectorAdd(double *img, int numElements)
{
    // int i = blockDim.x * blockIdx.x + threadIdx.x;

    int i_vel = blockIdx.x;
    int i_col = blockIdx.y;
    int i_row = threadIdx.x;

    int index = i_vel * n_pix * n_pix + i_col * n_pix + i_row;

    if (index < numElements)
    {
        img[index] = (double) index; // just put the index for now.
    }
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    init_constants();
    // print the value of a constant
    printf("M_sun is %e [g]\n", M_sun);


    // Determine the size of the image, and create memory to hold it.
    int n_vel = 4;
    int n_pix = 128;

    int numElements = n_vel * n_pix * n_pix;

    size_t size = numElements * sizeof(double);

    // Allocate the host image memory
    double *h_img = (double *)malloc(size);

    // Verify that allocations succeeded
    if (h_img == NULL)
    {
        fprintf(stderr, "Failed to allocate host image memory!\n");
        exit(EXIT_FAILURE);
    }


    // Allocate the device image memory
    double *d_img = NULL;
    err = cudaMalloc((void **)&d_img, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // A maximum of 65,000 can be called in each dimension.
    // For us, this is 256 x 256
    // So, most grids will be need to be id.x, id.y,

    // number of blocks in a single launch is limited to 65,535
    // neither dimension of a grid can exceed 65,535

    // a thread block can contain up to 1024 threads.
    // a kernel can be executed by multiple equally-shaped thread blocks, e.g., a grid

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = n_pix; // 128
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    dim3 numBlocks(n_vel, n_pix);
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_img, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy image from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print the results
    for (int i = 0; i < numElements; ++i)
    {
        printf("i=%d, h_img[%d]=%f\n", i, i, h_img[i]);
    }

    // Free device global memory
    err = cudaFree(d_img);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Free host memory
    free(h_img);

    printf("Done\n");
    return 0;
}
