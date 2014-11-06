
#include "cuda_global.h"

// executions per thread
#define N_PER_T 32
#define BLOCK_SIZE 256
//#define K 6
#define SS_DIM (DIM+1)

template<typename T, uint32_t K, uint32_t BLK_SIZE>
__global__ void vectorSum_kernel(T *d_x,
    uint32_t *z, uint32_t N, uint32_t k0, T *SSs) 
{
  // sufficient statistics for whole blocksize
  // 2 (x in TpS @north) + 1 (count) + 4 (outer product in TpS @north)
  // all fo that times 6 for the different axes
  __shared__ T xSSs[BLK_SIZE*SS_DIM*K];

  //const int tid = threadIdx.x;
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching 
#pragma unroll
  for(int s=0; s< K*SS_DIM; ++s) {
    // this is almost certainly bad ordering
    xSSs[tid*K*SS_DIM+s] = 0.0f;
  }
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    int32_t k = z[id]-k0;
    if(0 <= k && k < K)
    {
      // input sufficient statistics
      // because Eigen is col major by default
      xSSs[tid*SS_DIM*K+k*SS_DIM+0] += d_x[id*DIM+0];
      xSSs[tid*SS_DIM*K+k*SS_DIM+1] += d_x[id*DIM+1];
      xSSs[tid*SS_DIM*K+k*SS_DIM+2] += d_x[id*DIM+2];
//      xSSs[tid*SS_DIM*K+k*SS_DIM+0] += d_x[id];
//      xSSs[tid*SS_DIM*K+k*SS_DIM+1] += d_x[N+id];
//      xSSs[tid*SS_DIM*K+k*SS_DIM+2] += d_x[2*N+id];
      xSSs[tid*SS_DIM*K+k*SS_DIM+3] += 1.0f;
    }
  }

  // old reduction.....
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      const uint32_t si = s*K*SS_DIM;
      const uint32_t tidk = tid*K*SS_DIM;
#pragma unroll
      for( int k=0; k<K*SS_DIM; ++k) {
        xSSs[tidk+k] += xSSs[si+tidk+k];
      }
    }
    __syncthreads();
  }
  if(tid < K*SS_DIM) {
    // sum the last two remaining matrixes directly into global memory
    atomicAdd_<T>(&SSs[tid],xSSs[tid]+xSSs[tid+K*SS_DIM]);
  }
}

extern void vectorSum_gpu( double *d_x, uint32_t *d_z , uint32_t N, 
    uint32_t k0, uint32_t K, double *d_SSs)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    vectorSum_kernel<double,1,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==2){
    vectorSum_kernel<double,2,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==3){
    vectorSum_kernel<double,3,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==4){
    vectorSum_kernel<double,4,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==5){
    vectorSum_kernel<double,5,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==6){
    vectorSum_kernel<double,6,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void vectorSum_gpu(float *d_x, uint32_t *d_z, 
    uint32_t N, uint32_t k0, uint32_t K, float *d_SSs)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    vectorSum_kernel<float,1,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==2){
    vectorSum_kernel<float,2,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==3){
    vectorSum_kernel<float,3,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==4){
    vectorSum_kernel<float,4,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==5){
    vectorSum_kernel<float,5,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else if(K==6){
    vectorSum_kernel<float,6,BLK_SIZE><<<blocks,threads>>>(
        d_x, d_z,N,k0,d_SSs);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};
