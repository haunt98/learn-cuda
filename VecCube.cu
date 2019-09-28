#include <stdio.h>

#define CHECK(call)                                                                        \
    {                                                                                      \
        cudaError_t err = call;                                                            \
        if (err != cudaSuccess)                                                            \
        {                                                                                  \
            printf("%s in %s at line %d!\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    }

__global__ void cubeVecKernel(int *in, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = in[i] * in[i] * in[i];
    }
}

void cubeVec(int *in, int *out, int n, bool useDevice = false)
{
    if (useDevice == false)
    {
        for (int i = 0; i < n; i++)
        {
            out[i] = in[i] * in[i] * in[i];
        }
    }
    else // Use device
    {
        // Host allocates memories on device
        int *d_in, *d_out;
        CHECK(cudaMalloc(&d_in, n * sizeof(int)));
        CHECK(cudaMalloc(&d_out, n * sizeof(int)));

        // Host copies data to device memories
        CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

        // Host invokes kernel function to add vectors on device
        dim3 blockSize(512);
        dim3 gridSize((n - 1) / blockSize.x + 1);
        cubeVecKernel<<<gridSize, blockSize>>>(d_in, d_out, n);

        // Host copies result from device memory
        CHECK(cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

        // Host frees device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
    }
}

int main()
{
    int n;                 // Vector size
    int *in;               // Input vectors
    int *out, *correctOut; // Output vector

    // Input data into "n"
    n = 100000;

    // Allocate memories for "in", "out"
    in = (int *)malloc(n * sizeof(int));
    out = (int *)malloc(n * sizeof(int));
    correctOut = (int *)malloc(n * sizeof(int));

    // Input data into "in"
    for (int i = 0; i < n; i++)
    {
        in[i] = rand() & 0xff; // Random int in [0, 255]
    }

    // Cube vec on host
    cubeVec(in, correctOut, n);

    // Cube vec on device
    cubeVec(in, out, n, true);

    // Check correctness
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return 1;
        }
    }
    printf("CORRECT :)\n");
}
