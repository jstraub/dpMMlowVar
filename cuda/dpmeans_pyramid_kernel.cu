/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

__local__ inline float square(float a) {
  return a*a;
}

template<uint16_t D>
__local__ float squaredDist(float* a, float* b) {
  float sqDist = 0.f;
#pragma unroll
  for (uint16_t d=0; d<D; ++d)
    sqDist += square(a[d]-b[d]);
  return sqDist;
}

template<uint16_t D>
__local__ uint16_t indOfClosestCluster(float* xi, float* sim_closest,
    float* mus, float lambda, uint16_t K) {
  uint16_t z_i = K;
  *sim_closest = lambda;
  for (uint16_t k=0; k<K; ++k) {
    float sim_k = squaredDist<D>(mus+k*D, xi);
    if(sim_k < sim_closest) {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

template<uint16_t D, uint32_t BLK_SIZE>
__global__ void dpLabelAssign_kernel()
{
  __shared__ float mus[BLK_SIZE]; 
  __shared__ uint16_t zs[BLK_SIZE]; 
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  uint16_t K=0;
  float * xs = x[];
  for (uint16_t i=0; i<W; ++i) 
    for (uint16_t j=0; j<H; ++j) {
      float sim_closest = 0;
      uint16_t z = indOfClosestCluster<D>(xs_[i,j], &sim_closest, mus,
          lambda, K);
      if (z == K) {
        mus_.push_back(xs_[i]);
        ++K;
      }
      zs_[i] = z;
    }
  // accumulate data
  for (uint16_t i=0; i<W; ++i) 
    for (uint16_t j=0; j<H; ++j) {

    }
  
  // update means
  for (uint16_t k=0; k<K; ++k) {
#pragma unroll
    for (uint16_t d=0; d<D; ++d)
    mus[k] = 
  }
}
