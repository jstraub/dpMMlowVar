/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <stdio.h>
#include <jsCore/cuda_global.h>

#define DIM 3
#include <dpMMlowVar/ddpvMF_cuda_helper.h>
// executions per thread
#define N_PER_T 16
#define BLOCK_SIZE 256


template<typename T, uint32_t K, uint32_t BLK_SIZE>
__global__ void ddpvMFlabelAssign_kernel(T *d_q, T *d_p, uint32_t *z,
    uint32_t *d_Ns, T *d_ages, T *d_ws, T lambda, T beta, T Q, uint32_t *d_iAction,
    uint32_t i0, uint32_t N)
{
  __shared__ T p[DIM*(K+1)]; // K+1 because K might be 0 and 0 size arrays are not appreciated
//  __shared__ T ages[K+1];
  __shared__ T Ns[K+1];
//  __shared__ T ws[K+1];
  __shared__ uint32_t iAction[BLK_SIZE]; // id of first action (revieval/new) for one core

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching and init
  iAction[tid] = UNASSIGNED;
  if(tid < DIM*K) p[tid] = d_p[tid];
//  if(tid < K) ages[tid] = d_ages[tid];
//  if (K>=1) return;
  if(tid < K) Ns[tid] = d_Ns[tid];
//  if(tid < K) ws[tid] = d_ws[tid];
  __syncthreads(); // make sure that ys have been cached


  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    uint32_t z_i = K;
    T sim_closest = lambda + 1.;
    T sim_k = 0.;
    T* p_k = p;
    T q_i[DIM];
    q_i[0] = d_q[id*DIM];
    q_i[1] = d_q[id*DIM+1];
    q_i[2] = d_q[id*DIM+2];
    if (q_i[0]!=q_i[0] || q_i[1]!=q_i[1] || q_i[2]!=q_i[2])
    {
      // normal is nan -> break out here
      z[id] = UNASSIGNED;
    }else{
      for (uint32_t k=0; k<K; ++k)
      {
        T dot = min(1.0,max(-1.0,q_i[0]*p_k[0] + q_i[1]*p_k[1]
              + q_i[2]*p_k[2]));
        T zeta = acos(dot);

        if(Ns[k] == 0)
        {// cluster not instantiated yet in this timestep
          T age = d_ages[k]; // TODO d_ages size is not always = K
          //TODO: using small angle approximation here!
          sim_k = distToUninstantiatedSmallAngleApprox<T>(zeta,
              age,beta,d_ws[k],Q);
//          sim_k = distToUninstantiated<T,10>(zeta,age,beta,d_ws[k],Q,1e-6);
        }else{ // cluster instantiated
          sim_k = dot;
        }
        if(sim_k > sim_closest)
        {
          sim_closest = sim_k;
          z_i = k;
        }
        p_k += DIM;
      }
      if (z_i == K || Ns[z_i] == 0)
      {
        iAction[tid] = id;
        break; // save id at which an action occured and break out because after
        // that id anything more would be invalid.
      }
      z[id] = z_i;
    }
  }

  // min() reduction
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      iAction[tid] = min(iAction[tid], iAction[s+tid]);
    }
    __syncthreads();
  }
  if(tid == 0) {
    // reduce the last two remaining matrixes directly into global memory
    atomicMin(d_iAction, min(iAction[0],iAction[1]));
  }
};


void ddpvMFlabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z,
    uint32_t *d_Ns, double *d_ages, double *d_ws, double lambda, double beta,
    double Q, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 0){
    *d_iAction =0;
  }else if(K == 1){
    ddpvMFlabelAssign_kernel<double,1,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==2){
    ddpvMFlabelAssign_kernel<double,2,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==3){
    ddpvMFlabelAssign_kernel<double,3,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==4){
    ddpvMFlabelAssign_kernel<double,4,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==5){
    ddpvMFlabelAssign_kernel<double,5,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==6){
    ddpvMFlabelAssign_kernel<double,6,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==7){
    ddpvMFlabelAssign_kernel<double,7,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==8){
    ddpvMFlabelAssign_kernel<double,8,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==9){
    ddpvMFlabelAssign_kernel<double,9,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==10){
    ddpvMFlabelAssign_kernel<double,10,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==11){
    ddpvMFlabelAssign_kernel<double,11,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==12){
    ddpvMFlabelAssign_kernel<double,12,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==13){
    ddpvMFlabelAssign_kernel<double,13,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==14){
    ddpvMFlabelAssign_kernel<double,14,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==15){
    ddpvMFlabelAssign_kernel<double,15,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==16){
    ddpvMFlabelAssign_kernel<double,16,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};


void ddpvMFlabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z,
    uint32_t *d_Ns, float *d_ages, float *d_ws, float lambda, float beta,
    float Q, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 0){
    *d_iAction = 0;
  }else if(K == 1){
    ddpvMFlabelAssign_kernel<float,1,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==2){
    ddpvMFlabelAssign_kernel<float,2,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==3){
    ddpvMFlabelAssign_kernel<float,3,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==4){
    ddpvMFlabelAssign_kernel<float,4,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==5){
    ddpvMFlabelAssign_kernel<float,5,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==6){
    ddpvMFlabelAssign_kernel<float,6,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==7){
    ddpvMFlabelAssign_kernel<float,7,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==8){
    ddpvMFlabelAssign_kernel<float,8,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==9){
    ddpvMFlabelAssign_kernel<float,9,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==10){
    ddpvMFlabelAssign_kernel<float,10,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==11){
    ddpvMFlabelAssign_kernel<float,11,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==12){
    ddpvMFlabelAssign_kernel<float,12,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==13){
    ddpvMFlabelAssign_kernel<float,13,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==14){
    ddpvMFlabelAssign_kernel<float,14,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==15){
    ddpvMFlabelAssign_kernel<float,15,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else if(K==16){
    ddpvMFlabelAssign_kernel<float,16,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, beta, Q, d_iAction, i0, N);
  }else{
    assert(false);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

