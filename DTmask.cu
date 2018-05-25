
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


#include "constants.h"

// #include <helper_cuda.h>

// Define constants as constant memory.
// __constant__ double or something
// __device__ double 

__global__ void
vectorAdd(double *img, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        img[i] = M_sun; // just put the index for now.
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

 
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    // printf("Copy input data from the host memory to the CUDA device\n");
    // err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_img, numElements);
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

