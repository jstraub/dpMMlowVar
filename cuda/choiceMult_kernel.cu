#include <stdint.h>

#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>

__device__ uint wang_hash(uint seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}

__device__ uint rand_xorshift(uint rng_state)
{
  // Xorshift algorithm from George Marsaglia's paper
  rng_state ^= (rng_state << 13);
  rng_state ^= (rng_state >> 17);
  rng_state ^= (rng_state << 5);
  return rng_state;
}

#define N_PER_THREAD 1

template<typename T>
__global__ void choiceMult_kernel(T* pdfs, uint32_t *z, uint32_t N, uint32_t M,
  uint32_t seed)
{
  //int tid = threadIdx.x;
  int idx = (threadIdx.x + blockIdx.x*blockDim.x)*N_PER_THREAD;
  if(idx <= N-N_PER_THREAD)
  {
#pragma unroll
    for(uint32_t k=0; k<N_PER_THREAD; ++k)
    {
      int idk = idx+k;
      // obtain 32 bit random int and map it into 0.0 to 1.0
      T rnd = (wang_hash(idk+seed)*2.3283064365386963e-10);
      T cdf = pdfs[idk];
      uint32_t z_i = M-1;
      for (int i=1; i<M; ++i)
      {
        if(rnd <= cdf)
        {
          z_i = i-1;
          break;
        }
        cdf += pdfs[idk+i*N];
      }
      z[idk] = z_i;
    }
  }
}

template<typename T>
__global__ void choiceMultLogPdf_kernel(T* logPdfs, uint32_t *z, 
    uint32_t N, uint32_t M, uint32_t seed)
{
  //int tid = threadIdx.x;
  int idx = (threadIdx.x + blockIdx.x*blockDim.x)*N_PER_THREAD;
  if(idx <= N-N_PER_THREAD)
  {
#pragma unroll
    for(uint32_t k=0; k<N_PER_THREAD; ++k)
    {
      int idk = idx+k;
      // obtain 32 bit random int and map it into 0.0 to 1.0
      T rnd = (wang_hash(idk+seed)*2.3283064365386963e-10);
      T cdf = exp(logPdfs[idk]);
      uint32_t z_i = M-1;
      for (int i=1; i<M; ++i)
      {
        if(rnd <= cdf)
        {
          z_i = i-1;
          break;
        }
        cdf += exp(logPdfs[idk+i*N]);
      }
      z[idk] = z_i;
    }
  }
}

template<typename T>
__global__ void choiceMultLogPdfUnnormalizedGpu_kernel(T* pdfs, uint32_t *z,
  uint32_t N, uint32_t M, uint32_t seed)
{
  //int tid = threadIdx.x;
  int idx = (threadIdx.x + blockIdx.x*blockDim.x)*N_PER_THREAD;
  if(idx <= N-N_PER_THREAD)
  {
#pragma unroll
    for(uint32_t k=0; k<N_PER_THREAD; ++k)
    {
      int idk = idx+k;
      // obtain 32 bit random int and map it into 0.0 to 1.0
      T rnd = (wang_hash(idk+seed)*2.3283064365386963e-10);
      // normalizer for logPdf
      T maxLog = -9999999.; 
      for (int k=0; k<M; ++k)
        if(maxLog < pdfs[idk + k*N]) 
          maxLog = pdfs[idk + k*N];
      T normalizer = 0;
      for (int k=0; k<M; ++k)
        normalizer += exp(pdfs[idk + k*N] - maxLog);
      normalizer = log(normalizer) + maxLog;
      
      T cdf = exp(pdfs[idk]-normalizer);
      uint32_t z_i = M-1;
      for (int i=1; i<M; ++i)
      {
        if(rnd <= cdf)
        {
          z_i = i-1;
          break;
        }
        cdf += exp(pdfs[idk+i*N]-normalizer);
      }
      z[idk] = z_i;
    }
  }
}

template<typename T>
__global__ void unif_kernel(T* u, uint32_t N, uint32_t seed)
{
  //int tid = threadIdx.x;
  int idx = (threadIdx.x + blockIdx.x*blockDim.x)*N_PER_THREAD;
  if(idx <= N-N_PER_THREAD)
  {
    // obtain 32 bit random int and map it into 0.0 to 1.0
//    u[idx] = (wang_hash(idx+seed)*2.3283064365386963e-10);
//    u[idx+N/4] = (wang_hash(idx+seed)*2.3283064365386963e-10);
//    u[idx+N/2] = (wang_hash(idx+seed)*2.3283064365386963e-10);
//    u[idx+(3*N)/4] = (wang_hash(idx+seed)*2.3283064365386963e-10);

#pragma unroll
  for(uint32_t i=0; i<N_PER_THREAD; ++i)
    u[idx+i] = (wang_hash(idx+i+seed)*2.3283064365386963e-10);
//    u[idx+1] = (wang_hash(idx+1+seed)*2.3283064365386963e-10);
//    u[idx+2] = (wang_hash(idx+2+seed)*2.3283064365386963e-10);
//    u[idx+3] = (wang_hash(idx+3+seed)*2.3283064365386963e-10);
  }
}


// assumes that pdfs are copied to device already
extern void choiceMultGpu(double* d_pdf, uint32_t* d_z, uint32_t N, uint32_t M,
  uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  choiceMult_kernel<double><<<blocks,threads>>>(d_pdf,d_z,N,M,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};
extern void choiceMultGpu(float* d_pdf, uint32_t* d_z, uint32_t N, uint32_t M,
  uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  choiceMult_kernel<float><<<blocks,threads>>>(d_pdf,d_z,N,M,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void choiceMultLogPdfGpu(double* d_logPdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  choiceMultLogPdf_kernel<double><<<blocks,threads>>>(d_logPdf,d_z,N,M,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};
extern void choiceMultLogPdfGpu(float* d_logPdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  choiceMultLogPdf_kernel<float><<<blocks,threads>>>(d_logPdf,d_z,N,M,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void choiceMultLogPdfUnNormalizedGpu(double* d_pdf, uint32_t* d_z, 
  uint32_t N, uint32_t M, uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  choiceMultLogPdfUnnormalizedGpu_kernel<double><<<blocks,threads>>>(d_pdf,d_z,N,
    M,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};
extern void choiceMultLogPdfUnNormalizedGpu(float* d_pdf, uint32_t* d_z, 
  uint32_t N, uint32_t M, uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  choiceMultLogPdfUnnormalizedGpu_kernel<float><<<blocks,threads>>>(d_pdf,d_z,N,
    M,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void unifGpu(float* d_u, uint32_t N, uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  unif_kernel<float><<<blocks,threads>>>(d_u,N,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};
extern void unifGpu(double* d_u, uint32_t N, uint32_t seed)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  unif_kernel<double><<<blocks,threads>>>(d_u,N,seed);
  checkCudaErrors(cudaDeviceSynchronize());
};



//TODO: fast summing up of logPdfs indicated by z
template<typename T>
__global__ void sampleLikelihood_kernel(T* logPdfs, uint32_t *z, uint32_t N, uint32_t M,
  uint32_t seed)
{
  const int tid = threadIdx.x;
  const int idx = (threadIdx.x + blockIdx.x*blockDim.x)*N_PER_THREAD;
  __shared__ T sum[256]; // TODO

  if(idx <= N-N_PER_THREAD)
  {
#pragma unroll
    for(uint32_t k=0; k<N_PER_THREAD; ++k)
    {
      int idk = idx+k;
      // obtain 32 bit random int and map it into 0.0 to 1.0
      sum[tid] += logPdfs[idk+N*z[idk]];
    }
  }
}

#define K_MAX 256
/* 
 * computes the logsumexp over chuncks of rows of d_logPdf
 * each chunck has dk cols
 * d_logPdf has N rows and K cols
 * d_logNormalizer has N rows and K/dk cols
 */
template<typename T>
__global__ void logNormalizer(T* d_logPdf, T* d_logNormalizer,
    uint32_t dk, uint32_t K, uint32_t N) 
{
//  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  for(uint32_t id=idx*N_PER_THREAD; id<min(N,(idx+1)*N_PER_THREAD); ++id)
  {
    T pdf[K_MAX]; 
    for(uint32_t k=0; k<K; ++k)
      pdf[k] = d_logPdf[id+N*k];

    uint32_t kBlock=0;
    for(uint32_t k0=0; k0<K; k0+=dk)
    {
      T maxPdf = pdf[k0];
#pragma unroll
      for(uint32_t k=k0+1; k<k0+dk; ++k)
        if(maxPdf < pdf[k]) maxPdf = pdf[k];
      T logsumexp = exp(pdf[k0]-maxPdf);
#pragma unroll
      for(uint32_t k=k0+1; k<k0+dk; ++k)
        logsumexp += exp(pdf[k]-maxPdf);
      d_logNormalizer[id + N*(kBlock++)] = - log(logsumexp)+maxPdf;
    }
//#pragma unroll
//    for(uint32_t k=0; k<K; ++k)
//      d_logPdf[id+N*k] = exp(pdf[k]-logsumexp);
  }
}

/* 
 * log normalizes over chuncks of cols of d_logPdf
 * each chunck has dk cols
 * d_logPdf has N rows and K cols
 * d_logNormalizer has N rows and K/dk cols
 */
template<typename T>
__global__ void logNormalize(T* d_logPdf,
    uint32_t dk, uint32_t K, uint32_t N) 
{
//  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  for(uint32_t id=idx*N_PER_THREAD; id<min(N,(idx+1)*N_PER_THREAD); ++id)
  {
    T pdf[K_MAX]; 
    for(uint32_t k=0; k<K; ++k)
      pdf[k] = d_logPdf[id+N*k];

    for(uint32_t k0=0; k0<K; k0+=dk)
    {
      T maxPdf = pdf[k0];
#pragma unroll
      for(uint32_t k=k0+1; k<k0+dk; ++k)
        if(maxPdf < pdf[k]) maxPdf = pdf[k];
      T logsumexp = exp(pdf[k0]-maxPdf);
#pragma unroll
      for(uint32_t k=k0+1; k<k0+dk; ++k)
        logsumexp += exp(pdf[k]-maxPdf);
      logsumexp = log(logsumexp)+maxPdf;
#pragma unroll
      for(uint32_t k=k0; k<k0+dk; ++k)
        d_logPdf[id+N*k] = pdf[k]-logsumexp;
    }
//#pragma unroll
//    for(uint32_t k=0; k<K; ++k)
//      d_logPdf[id+N*k] = exp(pdf[k]-logsumexp);
  }
}

/* add logPi + logNormalizer to logPdf for each block of dk columns
 * logPi is length K/dk
 * d_logPdf has N rows and K cols
 * d_logNormalizer has N rows and K/dk cols
 */
template<typename T>
__global__ void logAddTopLevel(T* d_logPdf, T* d_logNormalizer, T* d_logPi,
    uint32_t dk, uint32_t K, uint32_t N) 
{
//  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  for(uint32_t id=idx*N_PER_THREAD; id<min(N,(idx+1)*N_PER_THREAD); ++id)
  {
#pragma unroll
    for(uint32_t k=0; k<K; ++k)
    {
      uint32_t kBlock = k/dk;
      d_logPdf[id+ N*k] += d_logPi[kBlock] + d_logNormalizer[id + N*kBlock];
    }
//#pragma unroll
//    for(uint32_t k=0; k<K; ++k)
//      d_logPdf[id+N*k] = exp(pdf[k]-logsumexp);
  }
}

extern void logNormalizerGpu(float* d_logPdf, float* d_logNormalizer, 
    uint32_t dk, uint32_t K, uint32_t N)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  logNormalizer<float><<<blocks,threads>>>(d_logPdf,d_logNormalizer,dk,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void logNormalizerGpu(double* d_logPdf, double* d_logNormalizer, 
    uint32_t dk, uint32_t K, uint32_t N)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  logNormalizer<double><<<blocks,threads>>>(d_logPdf,d_logNormalizer,dk,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void logNormalizeGpu(float* d_logPdf,
    uint32_t dk, uint32_t K, uint32_t N)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  logNormalize<float><<<blocks,threads>>>(d_logPdf,dk,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void logNormalizeGpu(double* d_logPdf, 
    uint32_t dk, uint32_t K, uint32_t N)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  logNormalize<double><<<blocks,threads>>>(d_logPdf,dk,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void logAddTopLevelGpu(float* d_logPdf, float* d_logNormalizer, 
    float* d_logPi, uint32_t dk, uint32_t K, uint32_t N)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  logAddTopLevel<float><<<blocks,threads>>>(d_logPdf,d_logNormalizer,d_logPi, 
      dk,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};

extern void logAddTopLevelGpu(double* d_logPdf, double* d_logNormalizer, 
    double* d_logPi, uint32_t dk, uint32_t K, uint32_t N)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/(256*N_PER_THREAD)+(N%(256*N_PER_THREAD)>0?1:0), 1,1);
  logAddTopLevel<double><<<blocks,threads>>>(d_logPdf,d_logNormalizer,d_logPi, 
      dk,K,N);
  checkCudaErrors(cudaDeviceSynchronize());
};
