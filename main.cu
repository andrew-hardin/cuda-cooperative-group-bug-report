#include <algorithm>
#include <iostream>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template<int kThreadCount>
__global__ void Kernel(const int* values, int count, int* global_min) {

  // CUDA programming guide 8.4.2.1: Thread Block Tile
  //
  // From the notes, thread blocks larger than 32 need a small amount
  // of shared memory allocated for CC <= 7.5.
  __shared__ cg::block_tile_memory<kThreadCount> for_reduction;
  cg::thread_block thread_block = cg::this_thread_block(for_reduction);

  // Calculate thread-local minimum value.
  int minimum = std::numeric_limits<int>::max();
  for(int x = thread_block.thread_rank(); x < count; x += kThreadCount) {
    int v = values[x];
    minimum = min(v, minimum);
  }

  // Calculate the minimum across all threads in the block.
  minimum = cg::reduce(cg::tiled_partition<kThreadCount>(thread_block), minimum, cg::less<int>());
  if(thread_block.thread_rank() == 0) {
    global_min[0] = minimum;
  }
}

// Thank you StackOverflow.
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CHECK_CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int, char**) {

  // Generate some random integer values in managed memory.
  static const int kSize = 5000;
  int* samples;
  CHECK_CUDA(cudaMallocManaged(&samples, sizeof(int) * kSize));
  for(int i = 0; i < kSize; i++) samples[i] = (rand() % 100) + 10;

  // Allocate output.
  int* gpu_min;
  CHECK_CUDA(cudaMallocManaged(&gpu_min, sizeof(int)));

  // Calculate the minimum.
  static const int kThreadCount = 256;
  Kernel<kThreadCount><<<1, kThreadCount>>>(samples, kSize, gpu_min);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Check that the min matches the host minimum.
  int host_min = *std::min_element(samples, samples + kSize);
  std::cout << "Host min = " << host_min << "; Device min = " << *gpu_min << std::endl;
  if(host_min == *gpu_min) {
    std::cout << "Host and device calculation match - we're good!" << std::endl;
    return EXIT_SUCCESS;
  } else {
    std::cout << "Host and device calculation don't match - there's a problem here." << std::endl;
    return EXIT_FAILURE;
  }
}