#include "cuda_global.h"
#include <stdio.h>

// executions per thread
#define N_PER_T 16
#define BLOCK_SIZE 256

#define K_MAX 100

template<typename T, uint32_t K, uint32_t BLK_SIZE>
__global__ void spkmLabelAssign_kernel(T *d_q, T *d_p, uint32_t *z, 
    uint32_t N)
{
  __shared__ T p[DIM*K];
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching and init
  if(tid < DIM*K) p[tid] = d_p[tid];
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    uint32_t z_i = UNASSIGNED;
    T sim_closest = -2.0;
    T* p_k = p;
    T q_i[DIM];
    q_i[0] = d_q[id*DIM];
    q_i[1] = d_q[id*DIM+1];
    q_i[2] = d_q[id*DIM+2];
    if (q_i[0]!=q_i[0] || q_i[1]!=q_i[1] || q_i[2]!=q_i[2])
    {
#pragma unroll
      for (uint32_t k=0; k<K; ++k)
      {
        T sim_k = min(1.0,max(-1.0,q_i[0]*p_k[0] + q_i[1]*p_k[1] + q_i[2]*p_k[2]));
        if(sim_k > sim_closest)
        {
          sim_closest = sim_k;
          z_i = k;
        }
        p_k += DIM;
      }
    }
    z[id] = z_i;
  }
  //TODO: could add a reduction to compute the cost function
}

template<typename T, uint32_t BLK_SIZE>
__global__ void spkmLabelAssignFlexK_kernel(T *d_q, T *d_p, uint32_t *z, 
    uint32_t K, uint32_t N)
{
  __shared__ T p[DIM*K_MAX];
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching and init
  if(tid < DIM*K) p[tid] = d_p[tid];
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    uint32_t z_i = UNASSIGNED;
    T sim_closest = -2.0;
    T* p_k = p;
    T q_i[DIM];
    q_i[0] = d_q[id*DIM];
    q_i[1] = d_q[id*DIM+1];
    q_i[2] = d_q[id*DIM+2];
    if (q_i[0]!=q_i[0] || q_i[1]!=q_i[1] || q_i[2]!=q_i[2])
    {
      for (uint32_t k=0; k<K; ++k)
      {
        T sim_k = min(1.0,max(-1.0,q_i[0]*p_k[0] + q_i[1]*p_k[1] + q_i[2]*p_k[2]));
        if(sim_k > sim_closest)
        {
          sim_closest = sim_k;
          z_i = k;
        }
        p_k += DIM;
      }
    }
    z[id] = z_i;
  }
  //TODO: could add a reduction to compute the cost function
}

void spkmLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z, 
    uint32_t K, uint32_t N)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    spkmLabelAssign_kernel<double,1,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);  
  }else if(K == 2){
    spkmLabelAssign_kernel<double,2,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 3){
    spkmLabelAssign_kernel<double,3,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 4){
    spkmLabelAssign_kernel<double,4,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 5){
    spkmLabelAssign_kernel<double,5,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 6){
    spkmLabelAssign_kernel<double,6,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 7){
    spkmLabelAssign_kernel<double,7,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 8){
    spkmLabelAssign_kernel<double,8,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 9){
    spkmLabelAssign_kernel<double,9,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 10){
    spkmLabelAssign_kernel<double,10,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 11){
    spkmLabelAssign_kernel<double,11,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 12){
    spkmLabelAssign_kernel<double,12,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 13){
    spkmLabelAssign_kernel<double,13,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 14){
    spkmLabelAssign_kernel<double,14,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 15){
    spkmLabelAssign_kernel<double,15,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 16){
    spkmLabelAssign_kernel<double,16,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 17){
    spkmLabelAssign_kernel<double,17,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 18){
    spkmLabelAssign_kernel<double,18,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 19){
    spkmLabelAssign_kernel<double,19,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 20){
    spkmLabelAssign_kernel<double,20,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 21){
    spkmLabelAssign_kernel<double,21,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else{
    spkmLabelAssignFlexK_kernel<double,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, K, N);
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

void spkmLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, 
    uint32_t K, uint32_t N)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    spkmLabelAssign_kernel<float,1,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);  
  }else if(K == 2){
    spkmLabelAssign_kernel<float,2,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 3){
    spkmLabelAssign_kernel<float,3,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 4){
    spkmLabelAssign_kernel<float,4,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 5){
    spkmLabelAssign_kernel<float,5,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 6){
    spkmLabelAssign_kernel<float,6,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 7){
    spkmLabelAssign_kernel<float,7,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 8){
    spkmLabelAssign_kernel<float,8,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 9){
    spkmLabelAssign_kernel<float,9,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 10){
    spkmLabelAssign_kernel<float,10,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 11){
    spkmLabelAssign_kernel<float,11,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 12){
    spkmLabelAssign_kernel<float,12,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 13){
    spkmLabelAssign_kernel<float,13,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 14){
    spkmLabelAssign_kernel<float,14,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 15){
    spkmLabelAssign_kernel<float,15,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 16){
    spkmLabelAssign_kernel<float,16,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 17){
    spkmLabelAssign_kernel<float,17,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 18){
    spkmLabelAssign_kernel<float,18,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 19){
    spkmLabelAssign_kernel<float,19,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 20){
    spkmLabelAssign_kernel<float,20,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else if(K == 21){
    spkmLabelAssign_kernel<float,21,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, N);
  }else{
    spkmLabelAssignFlexK_kernel<float,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, K, N);
  }
  checkCudaErrors(cudaDeviceSynchronize());
}
