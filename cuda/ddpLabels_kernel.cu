
#include <stdio.h>

#include <dpMMlowVar/cuda_global.h>

// executions per thread
#define N_PER_T 16
#define BLOCK_SIZE 256

template<typename T>
__device__ inline T distToUninstantiated( T distsq, T age, T w, T Q, T tau, T thresh)
{
  return Q*age+1.0/(1.0+1.0/w+age*tau)*distsq;
}

template<typename T, uint32_t BLK_SIZE>
__global__ void ddpLabelAssign_kernel(T *d_q, T *d_p, uint32_t *z, 
    uint32_t *d_Ns, T *d_ages, T *d_ws, T lambda, T Q, T tau, uint32_t *d_iAction, 
    uint32_t i0, uint32_t N, uint32_t K)
{
//  __shared__ T p[DIM*(K+1)];
////  __shared__ T ages[K+1];
//  __shared__ T Ns[K+1];
//  __shared__ T ws[K+1];
  __shared__ uint32_t iAction[BLK_SIZE]; // id of first action (revieval/new) for one core

  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

//  if(idx==0){printf("CALLED ddpLabelAssign_kernel\n");}
  __syncthreads();

  // caching and init
  iAction[tid] = UNASSIGNED;
//  if(tid < DIM*K) p[tid] = d_p[tid];
//  if(tid < K) ages[tid] = d_ages[tid];
//  if (K>=1) return;
//  if(tid < K) Ns[tid] = d_Ns[tid];
//  if(tid < K) ws[tid] = d_ws[tid];
  __syncthreads(); // make sure that ys have been cached

  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)
  {
    uint32_t z_i = K;
    T sim_closest = lambda;
    T sim_k = 0.;
    T* p_k = d_p;
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
        //      T dot = (this->ps_.col(k).transpose()*x_i)(0);
        T distsq = (q_i[0] - p_k[0])*(q_i[0] - p_k[0])
        		 +(q_i[1] - p_k[1])*(q_i[1] - p_k[1])
        		 +(q_i[2] - p_k[2])*(q_i[2] - p_k[2]);
        if(d_Ns[k] == 0)
        {// cluster not instantiated yet in this timestep                           
          T age = d_ages[k]; // TODO d_ages size is not always = K
          //TODO: using small angle approximation here!
          sim_k = distToUninstantiated<T>(distsq,age,d_ws[k],Q,tau,1e-6);
//          sim_k = distToUninstantiated<T,10>(zeta,age,beta,d_ws[k],Q,1e-6);
        }else{ // cluster instantiated                                              
          sim_k = distsq;
        }
        if(sim_k < sim_closest)
        {
          sim_closest = sim_k;
          z_i = k;
        }
        p_k += DIM;
      }
      if (z_i == K || d_Ns[z_i] == 0)
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

template<typename T, uint32_t K, uint32_t BLK_SIZE>
__global__ void ddpLabelAssign_kernel(T *d_q, T *d_p, uint32_t *z, 
    uint32_t *d_Ns, T *d_ages, T *d_ws, T lambda, T Q, T tau, uint32_t *d_iAction, 
    uint32_t i0, uint32_t N)
{
  __shared__ T p[DIM*(K+1)];
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
    T sim_closest = lambda;
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
        //      T dot = (this->ps_.col(k).transpose()*x_i)(0);
        T distsq = (q_i[0] - p_k[0])*(q_i[0] - p_k[0])
        		 +(q_i[1] - p_k[1])*(q_i[1] - p_k[1])
        		 +(q_i[2] - p_k[2])*(q_i[2] - p_k[2]);
        if(Ns[k] == 0)
        {// cluster not instantiated yet in this timestep                           
          T age = d_ages[k]; // TODO d_ages size is not always = K
          //TODO: using small angle approximation here!
          sim_k = distToUninstantiated<T>(distsq,age,d_ws[k],Q,tau,1e-6);
//          sim_k = distToUninstantiated<T,10>(zeta,age,beta,d_ws[k],Q,1e-6);
        }else{ // cluster instantiated                                              
          sim_k = distsq;
        }
        if(sim_k < sim_closest)
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


extern void ddpLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, double *d_ages, double *d_ws, double lambda, double Q, 
    double tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE/2;
//  assert(BLK_SIZE > DIM*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 0){
    ddpLabelAssign_kernel<double,0,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K == 1){
    ddpLabelAssign_kernel<double,1,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==2){
    ddpLabelAssign_kernel<double,2,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==3){
    ddpLabelAssign_kernel<double,3,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==4){
    ddpLabelAssign_kernel<double,4,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==5){
    ddpLabelAssign_kernel<double,5,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==6){
    ddpLabelAssign_kernel<double,6,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==7){
    ddpLabelAssign_kernel<double,7,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==8){
    ddpLabelAssign_kernel<double,8,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==9){
    ddpLabelAssign_kernel<double,9,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==10){
    ddpLabelAssign_kernel<double,10,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==11){
    ddpLabelAssign_kernel<double,11,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==12){
    ddpLabelAssign_kernel<double,12,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==13){
    ddpLabelAssign_kernel<double,13,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==14){
    ddpLabelAssign_kernel<double,14,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==15){
    ddpLabelAssign_kernel<double,15,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==16){
    ddpLabelAssign_kernel<double,16,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else{
    ddpLabelAssign_kernel<double,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N, K);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};


extern void ddpLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, float *d_ages, float *d_ws, float lambda, float Q, 
    float tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction)
{
  const uint32_t BLK_SIZE = BLOCK_SIZE;
//  assert(BLK_SIZE > DIM*K);

  dim3 threads(BLK_SIZE,1,1);
  dim3 blocks(N/(BLK_SIZE*N_PER_T)+(N%(BLK_SIZE*N_PER_T)>0?1:0),1,1);
  if(K == 0){
    ddpLabelAssign_kernel<float,0,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K == 1){
    ddpLabelAssign_kernel<float,1,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==2){
    ddpLabelAssign_kernel<float,2,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==3){
    ddpLabelAssign_kernel<float,3,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==4){
    ddpLabelAssign_kernel<float,4,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==5){
    ddpLabelAssign_kernel<float,5,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==6){
    ddpLabelAssign_kernel<float,6,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==7){
    ddpLabelAssign_kernel<float,7,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==8){
    ddpLabelAssign_kernel<float,8,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==9){
    ddpLabelAssign_kernel<float,9,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==10){
    ddpLabelAssign_kernel<float,10,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==11){
    ddpLabelAssign_kernel<float,11,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==12){
    ddpLabelAssign_kernel<float,12,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==13){
    ddpLabelAssign_kernel<float,13,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==14){
    ddpLabelAssign_kernel<float,14,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==15){
    ddpLabelAssign_kernel<float,15,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else if(K==16){
    ddpLabelAssign_kernel<float,16,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N);
  }else{
    ddpLabelAssign_kernel<float,BLK_SIZE><<<blocks,threads>>>(
        d_q, d_p, d_z, d_Ns, d_ages, d_ws, lambda, Q, tau, d_iAction, i0, N, K);
  }
  checkCudaErrors(cudaDeviceSynchronize());
};

