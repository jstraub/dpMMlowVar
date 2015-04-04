/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <dpMMlowVar/cuda_global.h>

// executions per thread
#define N_PER_T 32
#define BLOCK_SIZE 256

template<typename T, uint32_t D, uint32_t BLK_SIZE>
__global__ void copy_kernel(T *d_from, T *d_to,
    uint32_t N, uint32_t step, uint32_t offset) 
{
//  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
#pragma unroll
    for (int32_t d=0; d<D; ++d)
    {
      d_to[id*D+d] = d_from[id*step+offset+d];
    }
  }
};

extern void copy_gpu( double *d_from, double *d_to , uint32_t N, 
    uint32_t step, uint32_t offset, uint32_t D)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(D == 1){
    copy_kernel<double,1,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==2){
    copy_kernel<double,2,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==3){
    copy_kernel<double,3,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==4){
    copy_kernel<double,4,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==5){
    copy_kernel<double,5,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==6){
    copy_kernel<double,6,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else{
    assert(false);
  }
}

extern void copy_gpu( float *d_from, float *d_to , uint32_t N, 
    uint32_t step, uint32_t offset, uint32_t D)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(D == 1){
    copy_kernel<float,1,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==2){
    copy_kernel<float,2,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==3){
    copy_kernel<float,3,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==4){
    copy_kernel<float,4,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==5){
    copy_kernel<float,5,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==6){
    copy_kernel<float,6,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else{
    assert(false);
  }
}

extern void copy_gpu( uint32_t *d_from, uint32_t *d_to , uint32_t N, 
    uint32_t step, uint32_t offset, uint32_t D)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(D == 1){
    copy_kernel<uint32_t,1,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==2){
    copy_kernel<uint32_t,2,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==3){
    copy_kernel<uint32_t,3,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==4){
    copy_kernel<uint32_t,4,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==5){
    copy_kernel<uint32_t,5,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else if(D==6){
    copy_kernel<uint32_t,6,BLK_SIZE><<<blocks,threads>>>(
        d_from, d_to, N, step,offset);
  }else{
    assert(false);
  }
}
