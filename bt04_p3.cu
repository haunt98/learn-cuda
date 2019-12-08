#include <stdint.h>
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

// Sequential radix sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
void sortByHost(const uint32_t *in, int n, uint32_t *out, int nBits) {
  int nBins = 1 << nBits; // 2^nBits
  int *hist = (int *)malloc(nBins * sizeof(int));
  int *histScan = (int *)malloc(nBins * sizeof(int));

  // In each counting sort, we sort data in "src" and write result to "dst"
  // Then, we swap these 2 pointers and go to the next counting sort
  // At first, we assign "src = in" and "dest = out"
  // However, the data pointed by "in" is read-only
  // --> we create a copy of this data and assign "src" to the address of this
  // copy
  uint32_t *src = (uint32_t *)malloc(n * sizeof(uint32_t));
  memcpy(src, in, n * sizeof(uint32_t));
  uint32_t *originalSrc = src; // Use originalSrc to free memory later
  uint32_t *dst = out;

  // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
  // (Each digit consists of nBits bits)
  // In each loop, sort elements according to the current digit
  // (using STABLE counting sort)
  for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    // TODO: Compute "hist" of the current digit
    memset(hist, 0, nBins * sizeof(int));
    for (int i = 0; i < n; i += 1) {
      // >> de xoa cac bit o ben phai khong o vi tri "bit"
      // & de xoa cac bit khong o vi tri "bit"
      int bin = (src[i] >> bit) & (nBins - 1);
      hist[bin] += 1;
    }

    // TODO: Scan "hist" (exclusively) and save the result to "histScan"
    histScan[0] = 0;
    for (int bin = 1; bin < nBins; bin += 1) {
      histScan[bin] = histScan[bin - 1] + hist[bin - 1];
    }

    // TODO: From "histScan", scatter elements in "src" to correct locations in
    // "dst"
    for (int i = 0; i < n; i += 1) {
      int bin = (src[i] >> bit) & (nBins - 1);
      // histScan[bin] la vi tri cua src[i] trong dst
      dst[histScan[bin]] = src[i];
      histScan[bin] += 1;
    }

    // TODO: Swap "src" and "dst"
    uint32_t *temp = src;
    src = dst;
    dst = temp;
  }

  // TODO: Copy result to "out"
  memcpy(out, src, n * sizeof(uint32_t));

  // Free memories
  free(hist);
  free(histScan);
  free(originalSrc);
}

// CUDA compute hist only for radix sort
__global__ void computeHistKernel(uint32_t *in, int n, int *hist, int nBins,
                                  int bit) {
  // TODO
  // sHist[nBins]
  extern __shared__ int sHist[];
  // current threadIdx control threadIdx.x(+..blockDim.x) index in sHist
  for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
    sHist[bin] = 0;
  }
  __syncthreads();

  // Each block computes its local hist using atomic on SMEM
  // i index in "in"
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // >> de xoa cac bit o ben phai khong o vi tri "bit"
    // & de xoa cac bit khong o vi tri "bit"
    int bin = (in[i] >> bit) & (nBins - 1);
    atomicAdd(&sHist[bin], 1);
  }
  __syncthreads();

  // Each block adds its local hist to global hist using atomic on GMEM
  // add sHist to hist
  for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
    atomicAdd(&hist[bin], sHist[bin]);
  }
}

// compute scan blk kernel only for radix sort (exclusive)
__global__ void scanBlkKernel(int *in, int n, int *out, int *blkSums) {
  // TODO
  extern __shared__ int section[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // exclusive scan
  if (i < n && threadIdx.x != 0) {
    section[threadIdx.x] = in[i - 1];
  } else {
    section[threadIdx.x] = 0;
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

  // exclusive missing final index in 1 block "in"
  __syncthreads();
  if (i < n && threadIdx.x == blockDim.x - 1) {
    blkSums[blockIdx.x] += in[i];
  }
}

// add blkSums only for radix sort
__global__ void addScannedBlkSums(int *out, int n, int *blkSums) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // skip first block index in "out"
  if (i < n && blockIdx.x > 0) {
    out[i] += blkSums[blockIdx.x - 1];
  }
}

// (Partially) Parallel radix sort: implement parallel histogram and parallel
// scan in counting sort Assume: nBits (k in slides) in {1, 2, 4, 8, 16} Why
// "int * blockSizes"? Because we may want different block sizes for diffrent
// kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByDevice(const uint32_t *in, int n, uint32_t *out, int nBits,
                  int *blockSizes) {
  // TODO
  int nBins = 1 << nBits; // 2^nBits
  int *hist = (int *)malloc(nBins * sizeof(int));
  int *histScan = (int *)malloc(nBins * sizeof(int));

  // CUDA
  int histogramBlockSize = blockSizes[0];
  dim3 histogramGridSize((n - 1) / histogramBlockSize + 1);
  int scanBlockSize = blockSizes[1];
  dim3 scanGridSize((n - 1) / scanBlockSize + 1);

  // CUDA
  int *d_hist, *d_histScan, *d_blkSums;
  CHECK(cudaMalloc(&d_hist, nBins * sizeof(int)));
  CHECK(cudaMalloc(&d_histScan, nBins * sizeof(int)));
  CHECK(cudaMalloc(&d_blkSums, scanGridSize.x * sizeof(int)));

  // In each counting sort, we sort data in "src" and write result to "dst"
  // Then, we swap these 2 pointers and go to the next counting sort
  // At first, we assign "src = in" and "dest = out"
  // However, the data pointed by "in" is read-only
  // --> we create a copy of this data and assign "src" to the address of this
  // copy
  uint32_t *src = (uint32_t *)malloc(n * sizeof(uint32_t));
  memcpy(src, in, n * sizeof(uint32_t));
  uint32_t *originalSrc = src; // Use originalSrc to free memory later
  uint32_t *dst = out;

  // CUDA
  uint32_t *d_src;
  CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));

  // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
  // (Each digit consists of nBits bits)
  // In each loop, sort elements according to the current digit
  // (using STABLE counting sort)
  for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    // TODO: Compute "hist" of the current digit
    // Compute hist or kernel
    // CUDA
    // copy src to cuda
    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice))
    // clear cuda hist
    CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)))
    computeHistKernel<<<histogramGridSize, histogramBlockSize,
                        nBins * sizeof(int)>>>(d_src, n, d_hist, nBins, bit);
    CHECK(cudaDeviceSynchronize());

    // TODO: Scan "hist" (exclusively) and save the result to "histScan"
    // CUDA
    // scan on "d_hist" and write to "d_histScan" and "d_blkSums"
    scanBlkKernel<<<scanGridSize, scanBlockSize,
                    scanBlockSize * sizeof(int)>>>(d_hist, nBins, d_histScan,
                                                     d_blkSums);
    CHECK(cudaDeviceSynchronize());

    // temp blkSums
    int *blkSums = (int *)malloc(scanGridSize.x * sizeof(int));
    CHECK(cudaMemcpy(blkSums, d_blkSums, scanGridSize.x * sizeof(int),
                     cudaMemcpyDeviceToHost));

    // Inclusive scan blkSums
    for (int i = 1; i < scanGridSize.x; i += 1)
      blkSums[i] += blkSums[i - 1];

    CHECK(cudaMemcpy(d_blkSums, blkSums, scanGridSize.x * sizeof(int),
                     cudaMemcpyHostToDevice));
    free(blkSums);

    // add blkSums to out
    addScannedBlkSums<<<scanGridSize, scanBlockSize>>>(d_histScan, nBins,
                                                       d_blkSums);
    CHECK(cudaDeviceSynchronize());

    // copy cuda histScan to host histScan
    CHECK(cudaMemcpy(histScan, d_histScan, nBins * sizeof(int),
                     cudaMemcpyDeviceToHost));

    // TODO: From "histScan", scatter elements in "src" to correct locations in
    // "dst"
    for (int i = 0; i < n; i += 1) {
      int bin = (src[i] >> bit) & (nBins - 1);
      // histScan[bin] la vi tri cua src[i] trong dst
      dst[histScan[bin]] = src[i];
      histScan[bin] += 1;
    }

    // TODO: Swap "src" and "dst"
    uint32_t *temp = src;
    src = dst;
    dst = temp;
  }

  // TODO: Copy result to "out"
  memcpy(out, src, n * sizeof(uint32_t));

  // Free memories
  free(hist);
  free(histScan);
  free(originalSrc);

  CHECK(cudaFree(d_hist));
  CHECK(cudaFree(d_histScan));
  CHECK(cudaFree(d_blkSums));
  CHECK(cudaFree(d_src));
}

// Radix sort
void sort(const uint32_t *in, int n, uint32_t *out, int nBits,
          bool useDevice = false, int *blockSizes = NULL) {
  GpuTimer timer;
  timer.Start();

  if (useDevice == false) {
    printf("\nRadix sort by host\n");
    sortByHost(in, n, out, nBits);
  } else // use device
  {
    printf("\nRadix sort by device\n");
    sortByDevice(in, n, out, nBits, blockSizes);
  }

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
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

void checkCorrectnessHost(uint32_t *out, int n) {
  for (int i = 0; i < n - 1; i += 1) {
    if (out[i] > out[i + 1]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}

void checkCorrectness(uint32_t *out, uint32_t *correctOut, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != correctOut[i]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}

void printArray(uint32_t *a, int n) {
  for (int i = 0; i < n; i++)
    printf("%i ", a[i]);
  printf("\n");
}

int main(int argc, char **argv) {
  // PRINT OUT DEVICE INFO
  printDeviceInfo();

  // SET UP INPUT SIZE
  int n = (1 << 24) + 1;
  // TEST HOST
  // n = 10;
  printf("\nInput size: %d\n", n);

  // ALLOCATE MEMORIES
  size_t bytes = n * sizeof(uint32_t);
  uint32_t *in = (uint32_t *)malloc(bytes);
  uint32_t *out = (uint32_t *)malloc(bytes);        // Device result
  uint32_t *correctOut = (uint32_t *)malloc(bytes); // Host result

  // SET UP INPUT DATA
  for (int i = 0; i < n; i++)
    in[i] = rand();
  // printArray(in, n);

  // SET UP NBITS
  int nBits = 4; // Default
  if (argc > 1)
    nBits = atoi(argv[1]);
  printf("\nNum bits per digit: %d\n", nBits);

  // DETERMINE BLOCK SIZES
  int blockSizes[2] = {512, 512}; // One for histogram, one for scan
  if (argc == 4) {
    blockSizes[0] = atoi(argv[2]);
    blockSizes[1] = atoi(argv[3]);
  }
  printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0],
         blockSizes[1]);

  // SORT BY HOST
  sort(in, n, correctOut, nBits);
  // TEST HOST
  // checkCorrectnessHost(correctOut, n);
  // printArray(correctOut, n);

  // SORT BY DEVICE
  sort(in, n, out, nBits, true, blockSizes);
  checkCorrectness(out, correctOut, n);

  // FREE MEMORIES
  free(in);
  free(out);
  free(correctOut);

  return EXIT_SUCCESS;
}
