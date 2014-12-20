#include "cuda_global.h"
#include <stdio.h>

// executions per thread
#define N_PER_T 16
#define BLOCK_SIZE 256

template<uint32_t BLK_SIZE>
__global__ void labelMap_kernel(uint32_t *z, int32_t* map, uint32_t N)
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
    z[id] = map[z[id]];
};

extern void labelMapGpu(uint32_t *d_z, int32_t* d_Map, uint32_t N)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;
  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  labelMap_kernel<BLOCK_SIZE><<<blocks,threads>>>(d_z,d_Map,N); 
  checkCudaErrors(cudaDeviceSynchronize());
};
