
#include <stdio.h>
#include <float.h>
#include <dpMMlowVar/cuda_global.h>

// executions per thread
#define K_MAX 50
#define N_PER_T 16
#define BLOCK_SIZE 256

template<typename T>
__device__ inline T distToUninstantiated( T distsq, T age, T w, T Q, T tau, T thresh)
{
  return Q*age+1.0/(1.0+1.0/w+age*tau)*distsq;
}

template<typename T, uint32_t BLK_SIZE>
__global__ void ddpLabelAssignSpecial_kernel(T *d_q, T *d_oldp, T
    *d_ages, T *d_ws, T lambda, T Q, T tau, uint32_t *d_asgnIdces,
    uint32_t N, uint32_t K)
{
  //__shared__ T oldp[DIM*K];
  __shared__ uint32_t asgnIdces[K_MAX*BLK_SIZE]; //for each thread, index selected for each old K
  __shared__ T oldp[K_MAX*DIM];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching and init
  for(int k = 0; k < K; k++){
    asgnIdces[K*tid+k] = UNASSIGNED;
    if(tid < DIM) oldp[k*DIM+tid] = d_oldp[k*DIM+tid];
  }
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    T max_sim_k = FLT_MAX;
    uint32_t max_k = UNASSIGNED;
    T sim_k = 0.;
    T* p_k = oldp;
    T q_i[DIM];
    q_i[0] = d_q[id*DIM];
    q_i[1] = d_q[id*DIM+1];
    q_i[2] = d_q[id*DIM+2];
    if (q_i[0] ==q_i[0] && q_i[1] ==q_i[1] && q_i[2]==q_i[2])// only do this for q not nan
    {
      for (uint32_t k=0; k<K; ++k) {
        T distsq = (q_i[0] - p_k[0])*(q_i[0] - p_k[0])
        		 +(q_i[1] - p_k[1])*(q_i[1] - p_k[1])
        		 +(q_i[2] - p_k[2])*(q_i[2] - p_k[2]);
        sim_k = distToUninstantiated<T>(distsq,d_ages[k],d_ws[k],Q,tau,1e-6);
        if(sim_k < lambda && max_sim_k > sim_k)
        {
          max_sim_k = sim_k;
          max_k = k;
        }
        p_k += DIM;
      }
      if(max_k < K && id < asgnIdces[K*tid+max_k]){
        asgnIdces[K*tid+max_k] = id;
      }
    }
  }
  // min() reduction
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      for(uint32_t k = 0; k < K; ++k){
        if(asgnIdces[K*tid+k] > asgnIdces[K*(s+tid)+k]){
          asgnIdces[K*tid+k] = asgnIdces[K*(s+tid)+k];
        } 
      }
    }
    __syncthreads();
  }

  //reduce the 2 remaining into the output d_asgnCosts/d_asgnIndices
  if(tid < K) {
    if(asgnIdces[tid] < asgnIdces[K+tid]){
      // leads to the smallest index of minimal cost value (but only minimal wrt to its block)
      // this is not the argmin over all values - that is probably not possible atomically
      atomicMin(&d_asgnIdces[tid], asgnIdces[tid]);
    } else {
      atomicMin(&d_asgnIdces[tid], asgnIdces[K+tid]);
    }
  }

};

template<typename T, uint32_t K, uint32_t BLK_SIZE>
__global__ void ddpLabelAssignSpecial_kernel(T *d_q, T *d_oldp, T *d_ages, T *d_ws, T lambda, T Q, T tau, uint32_t *d_asgnIdces, uint32_t N)
{
  //__shared__ T oldp[DIM*K];
  __shared__ uint32_t asgnIdces[K*BLK_SIZE]; //for each thread, index selected for each old K
  __shared__ T oldp[K*DIM];

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // caching and init
  for(int k = 0; k < K; k++){
    asgnIdces[K*tid+k] = UNASSIGNED;
    if(tid < DIM) oldp[k*DIM+tid] = d_oldp[k*DIM+tid];
  }
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    T max_sim_k = FLT_MAX;
    uint32_t max_k = UNASSIGNED;
    T sim_k = 0.;
    T* p_k = oldp;
    T q_i[DIM];
    q_i[0] = d_q[id*DIM];
    q_i[1] = d_q[id*DIM+1];
    q_i[2] = d_q[id*DIM+2];
    if (q_i[0] ==q_i[0] && q_i[1] ==q_i[1] && q_i[2]==q_i[2])// only do this for q not nan
    {
      for (uint32_t k=0; k<K; ++k) {
        T distsq = (q_i[0] - p_k[0])*(q_i[0] - p_k[0])
        		 +(q_i[1] - p_k[1])*(q_i[1] - p_k[1])
        		 +(q_i[2] - p_k[2])*(q_i[2] - p_k[2]);
        sim_k = distToUninstantiated<T>(distsq,d_ages[k],d_ws[k],Q,tau,1e-6);
        if(sim_k < lambda && max_sim_k > sim_k)
        {
          max_sim_k = sim_k;
          max_k = k;
        }
        p_k += DIM;
      }
      if(max_k < K && id < asgnIdces[K*tid+max_k]){
        asgnIdces[K*tid+max_k] = id;
      }
    }
  }
  // min() reduction
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
      for(uint32_t k = 0; k < K; ++k){
        if(asgnIdces[K*tid+k] > asgnIdces[K*(s+tid)+k]){
          asgnIdces[K*tid+k] = asgnIdces[K*(s+tid)+k];
        } 
      }
    }
    __syncthreads();
  }

  //reduce the 2 remaining into the output d_asgnCosts/d_asgnIndices
  if(tid < K) {
    if(asgnIdces[tid] < asgnIdces[K+tid]){
      // leads to the smallest index of minimal cost value (but only minimal wrt to its block)
      // this is not the argmin over all values - that is probably not possible atomically
      atomicMin(&d_asgnIdces[tid], asgnIdces[tid]);
    } else {
      atomicMin(&d_asgnIdces[tid], asgnIdces[K+tid]);
    }
  }

};

extern void ddpLabelsSpecial_gpu( double *d_q,  double *d_oldp, double *d_ages, double *d_ws, double lambda, double Q, 
    double tau, uint32_t K, uint32_t N, uint32_t *d_asgnIdces)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(K >= 1);//only run the special kernel if there is at least one old cluster
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    ddpLabelAssignSpecial_kernel<double,1, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==2){
    ddpLabelAssignSpecial_kernel<double,2, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==3){
    ddpLabelAssignSpecial_kernel<double,3, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==4){
    ddpLabelAssignSpecial_kernel<double,4, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==5){
    ddpLabelAssignSpecial_kernel<double,5, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==6){
    ddpLabelAssignSpecial_kernel<double,6, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==7){
    ddpLabelAssignSpecial_kernel<double,7, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==8){
    ddpLabelAssignSpecial_kernel<double,8, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==9){
    ddpLabelAssignSpecial_kernel<double,9, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==10){
    ddpLabelAssignSpecial_kernel<double,10, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==11){
    ddpLabelAssignSpecial_kernel<double,11, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces,  N);
  }else if(K==12){
    ddpLabelAssignSpecial_kernel<double,12, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==13){
    ddpLabelAssignSpecial_kernel<double,13, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==14){
    ddpLabelAssignSpecial_kernel<double,14, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==15){
    ddpLabelAssignSpecial_kernel<double,15, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==16){
    ddpLabelAssignSpecial_kernel<double,16, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else{
    ddpLabelAssignSpecial_kernel<double,BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N, K);
  }
  checkCudaErrors(cudaDeviceSynchronize());

};


extern void ddpLabelsSpecial_gpu( float *d_q,  float *d_oldp, float *d_ages,
    float *d_ws, float lambda, float Q, float tau, uint32_t K, uint32_t N,
    uint32_t *d_asgnIdces)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
  assert(K >= 1);//only run the special kernel if there is at least one old cluster
  assert(BLK_SIZE > DIM*K+DIM*(DIM-1)*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 1){
    ddpLabelAssignSpecial_kernel<float,1, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==2){
    ddpLabelAssignSpecial_kernel<float,2, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==3){
    ddpLabelAssignSpecial_kernel<float,3, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==4){
    ddpLabelAssignSpecial_kernel<float,4, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==5){
    ddpLabelAssignSpecial_kernel<float,5, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==6){
    ddpLabelAssignSpecial_kernel<float,6, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==7){
    ddpLabelAssignSpecial_kernel<float,7, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==8){
    ddpLabelAssignSpecial_kernel<float,8, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==9){
    ddpLabelAssignSpecial_kernel<float,9, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==10){
    ddpLabelAssignSpecial_kernel<float,10, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==11){
    ddpLabelAssignSpecial_kernel<float,11, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces,  N);
  }else if(K==12){
    ddpLabelAssignSpecial_kernel<float,12, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==13){
    ddpLabelAssignSpecial_kernel<float,13, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==14){
    ddpLabelAssignSpecial_kernel<float,14, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==15){
    ddpLabelAssignSpecial_kernel<float,15, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else if(K==16){
    ddpLabelAssignSpecial_kernel<float,16, BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N);
  }else{
    ddpLabelAssignSpecial_kernel<float,BLK_SIZE><<<blocks, threads>>>(
        d_q, d_oldp, d_ages, d_ws, lambda, Q, tau, d_asgnIdces, N, K);
  }
  checkCudaErrors(cudaDeviceSynchronize());

};

