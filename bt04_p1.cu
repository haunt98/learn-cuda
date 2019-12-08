#include <stdio.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
  }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

/*
Scan within each block's data (work-inefficient), write results to "out", and
write each block's sum to "blkSums" if "blkSums" is not NULL.
*/
__global__ void scanBlkKernel(int *in, int n, int *out, int *blkSums) {
  // TODO
  extern __shared__ int section[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // inclusive scan
  if (i < n) {
    section[threadIdx.x] = in[i];
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    // copy temp section[threadIdx.x - stride] before changed
    int temp = 0;
    if (stride <= threadIdx.x)
      temp = section[threadIdx.x - stride];
    __syncthreads();

    section[threadIdx.x] += temp;
    __syncthreads();
  }

  __syncthreads();
  if (i < n) {
    out[i] = section[threadIdx.x];
  }

  // copy to blkSums
  __syncthreads();
  if (blkSums != NULL && threadIdx.x == 0) {
    blkSums[blockIdx.x] = section[blockDim.x - 1];
  }
}

// TODO: You can define necessary functions here

// add one of the blkSums elements to all in elements
__global__ void addScannedBlkSums(int *out, int n, int *blkSums) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // skip first block index in "out"
  if (i < n && blockIdx.x > 0) {
    out[i] += blkSums[blockIdx.x - 1];
  }
}

void scan(int *in, int n, int *out, bool useDevice = false,
          dim3 blkSize = dim3(1)) {
  GpuTimer timer;
  timer.Start();
  if (useDevice == false) {
    printf("\nScan by host\n");
    out[0] = in[0];
    for (int i = 1; i < n; i++) {
      out[i] = out[i - 1] + in[i];
    }
  } else // Use device
  {
    printf("\nScan by device\n");
    // TODO
    dim3 gridSize((n - 1) / blkSize.x + 1);

    int *d_in, *d_out, *d_blkSums;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
    CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

    // scan on in and write to out and blkSums
    scanBlkKernel<<<gridSize, blkSize, blkSize.x * sizeof(int)>>>(
        d_in, n, d_out, d_blkSums);
    CHECK(cudaDeviceSynchronize());

    // temp blkSums
    int *blkSums = (int *)malloc(gridSize.x * sizeof(int));
    CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize.x * sizeof(int),
                     cudaMemcpyDeviceToHost));

    // Inclusive scan blkSums
    for (int i = 1; i < gridSize.x; i += 1)
      blkSums[i] += blkSums[i - 1];

    CHECK(cudaMemcpy(d_blkSums, blkSums, gridSize.x * sizeof(int),
                     cudaMemcpyHostToDevice));
    free(blkSums);

    // add blkSums to out
    addScannedBlkSums<<<gridSize, blkSize>>>(d_out, n, d_blkSums);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_blkSums));
  }
  timer.Stop();
  printf("Processing time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo() {
  cudaDeviceProp devProv;
  CHECK(cudaGetDeviceProperties(&devProv, 0));
  printf("**********GPU info**********\n");
  printf("Name: %s\n", devProv.name);
  printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
  printf("Num SMs: %d\n", devProv.multiProcessorCount);
  printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
  printf("Max num warps per SM: %d\n",
         devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
  printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
  printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
  printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
  printf("****************************\n");
}

void checkCorrectness(int *out, int *correctOut, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != correctOut[i]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}

void printArr(int *arr, int n) {
  for (int i = 0; i < n; i += 1) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

int main(int argc, char **argv) {
  // PRINT OUT DEVICE INFO
  printDeviceInfo();

  // SET UP INPUT SIZE
  int n = (1 << 24) + 1;
  printf("\nInput size: %d\n", n);

  // ALLOCATE MEMORIES
  size_t bytes = n * sizeof(int);
  int *in = (int *)malloc(bytes);
  int *out = (int *)malloc(bytes);        // Device result
  int *correctOut = (int *)malloc(bytes); // Host result

  // SET UP INPUT DATA
  for (int i = 0; i < n; i++)
    in[i] = (int)(rand() & 0xFF) - 127; // random int in [-127, 128]

  // DETERMINE BLOCK SIZE
  dim3 blockSize(512);
  if (argc == 2) {
    blockSize.x = atoi(argv[1]);
  }

  // TEST
  // printArr(in, n);

  // SCAN BY HOST
  scan(in, n, correctOut);

  // TEST
  // printArr(correctOut, n);

  // SCAN BY DEVICE
  scan(in, n, out, true, blockSize);
  checkCorrectness(out, correctOut, n);

  // TEST
  // printArr(out, n);

  // FREE MEMORIES
  free(in);
  free(out);
  free(correctOut);

  return EXIT_SUCCESS;
}
