#include <stdio.h>
#include <vector>

#define SIZE 10000

#define TILE 16

inline void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void multiplyMatrix(float *a, float *b, float *c, int size) {
    
    // TODO: Compute the global row and column indices for this thread. Use blockIdx, blockDim, and threadIdx. It is a 2D grid of 2D blocks, watch slide if not sure how to do this.
    int row = 0;
    int col = 0;


    // Compute a value in the result
    // Line multiplied by column
    if (row < size && col < size) {
        float sum = 0;
        for (int i = 0; i < size; i++) {
            sum += a[row * size + i] * b[i * size + col];
        }
        c[row * size + col] = sum;
    }
}

__global__ void betterMultiplyMatrix(float *a, float *b, float *c, int size) {
    
    // TODO: Compute the global row and column indices for this thread. Use blockIdx, blockDim, and threadIdx.
    int row = 0; 
    int col = 0;

    // I used here a more efficient memory shared between threads in the same block, to reduce the number of accesses to global memory.
    __shared__ float aTile[TILE][TILE];
    __shared__ float bTile[TILE][TILE];

    float sum = 0;

    // TODO: this loop runs "size / TILE" times using integer division.
    // Add the missing bound check so that the last blocks of threads don't make out-of-bounds accesses to a and b.


    for (int i = 0; i < size / TILE; i ++) {
        // Load the tiles from global memory to shared memory
        aTile[threadIdx.y][threadIdx.x] = a[row * size + (i * TILE + threadIdx.x)];
        bTile[threadIdx.y][threadIdx.x] = b[(i * TILE + threadIdx.y) * size + col];
        __syncthreads();
        // Compute smaller matrix multiplication
        for (int j = 0; j < TILE; j++) {
            sum += aTile[threadIdx.y][j] * bTile[j][threadIdx.x];
        }
        __syncthreads();
    }

    c[row * size + col] = sum;
}

int main(void) {
    float *aDev, *bDev, *cDev;
    std::vector<float> aHost(SIZE * SIZE, 1), bHost(SIZE * SIZE, 1), cHost(SIZE * SIZE, 0);

    cudaError_t err;

    err = cudaMalloc((void**)&aDev, SIZE * SIZE * sizeof(float));
    checkCuda(err);

    err = cudaMalloc((void**)&bDev, SIZE * SIZE * sizeof(float));
    checkCuda(err);

    err = cudaMalloc((void**)&cDev, SIZE * SIZE * sizeof(float));
    checkCuda(err);

    err = cudaMemcpy(aDev, aHost.data(), SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);
    checkCuda(err);

    err = cudaMemcpy(bDev, bHost.data(), SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);
    checkCuda(err);

    err = cudaMemset(cDev, 0, SIZE * SIZE * sizeof(float));
    checkCuda(err);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((SIZE + dimBlock.x - 1) / dimBlock.x, (SIZE + dimBlock.y - 1) / dimBlock.y);

    // Also use events to measure time
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    checkCuda(err);

    err = cudaEventCreate(&stop);
    checkCuda(err);

    err = cudaEventRecord(start);
    checkCuda(err);

    // TODO: try launching multiplyMatrix (the naive version) here instead of
    // betterMultiplyMatrix, keeping everything else the same, and compare the time
    betterMultiplyMatrix<<<dimGrid, dimBlock>>>(aDev, bDev, cDev, SIZE);

    // TODO: a kernel launch can fail silently (bad grid/block configuration,
    // out-of-bounds shared memory, etc.). Add a cudaGetLastError() + checkCuda()
    // call right here, immediately after the launch.

    err = cudaEventRecord(stop);
    checkCuda(err);

    err = cudaEventSynchronize(stop);
    checkCuda(err);

    err = cudaMemcpy(cHost.data(), cDev, SIZE * SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    checkCuda(err);

    cudaFree(aDev);
    cudaFree(bDev);
    cudaFree(cDev);

    // TODO: cudaEventCreate() allocates resources, add matching cudaEventDestroy()

    printf("cHost[0] = %f\n", cHost[0]);
    printf("cHost[SIZE * SIZE - 1] = %f\n", cHost[SIZE * SIZE - 1]);

    float ms;
    err = cudaEventElapsedTime(&ms, start, stop);
    checkCuda(err);

    printf("Time: %f ms\n", ms);
}
