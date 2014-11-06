#pragma once

#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>

#define PI 3.141592653589793
#define DIM 3

#define MIN_DOT -0.95
#define MAX_DOT 0.95

#define MAX_UINT32 4294967295
#define INVALID_LABEL MAX_UINT32 //4294967294

#if DIM==3
template<typename T>
__device__ inline void Log_p(T *p, T *q, T *x)
{
  T dot = min(1.0,max(-1.0,q[0]*p[0] + q[1]*p[1] + q[2]*p[2]));
  // 2nd order taylor expansions for the limit cases obtained via mathematica
  T invSinc = 0.0;
  if(static_cast<T>(MIN_DOT) < dot && dot < static_cast<T>(MAX_DOT))
    invSinc = acos(dot)/sqrt(1.-dot*dot);
  else if(dot <= static_cast<T>(MIN_DOT))
    invSinc = PI/(sqrt(2.)*sqrt(dot+1.)) -1. + PI*sqrt(dot+1.)/(4.*sqrt(2.))
      -(dot+1.)/3. + 3.*PI*(dot+1.)*sqrt(dot+1.)/(32.*sqrt(2.)) 
      - 2./15.*(dot+1.)*(dot+1.);
  else if(dot >= static_cast<T>(MAX_DOT))
    invSinc = 1. - (dot-1)/3. + 2./5.*(dot-1.)*(dot-1.);
  x[0] = (q[0]-p[0]*dot)*invSinc;
  x[1] = (q[1]-p[1]*dot)*invSinc;
  x[2] = (q[2]-p[2]*dot)*invSinc;
}


template<typename T>
__device__ inline void Ab(T *A, T *b)
{

};

/* computes b^TAb; 
 * assumes that A and b are in one piece in memory
 * written up for row-major A; but works for col major as well since
 * (b^TAb)^T = b^TA^Tb and row -> col major is transposing
 */
template<typename T>
__device__ inline T bTAb_3D(T *A, T *b)
{
  return b[0]*b[0]*A[0] + b[0]*b[1]*A[1] + b[0]*b[2]*A[2]
        +b[1]*b[0]*A[3] + b[1]*b[1]*A[4] + b[1]*b[2]*A[5]
        +b[2]*b[0]*A[6] + b[2]*b[1]*A[7] + b[2]*b[2]*A[8];
};

/* computes b^TAb; 
 * assumes that A and b are in one piece in memory
 * written up for row-major A; but works for col major as well since
 * (b^TAb)^T = b^TA^Tb and row -> col major is transposing
 */
template<typename T>
__device__ inline T bTAb_2D(T *A, T *b)
{
  return b[0]*b[0]*A[0] + b[0]*b[1]*A[1]
        +b[1]*b[0]*A[2] + b[1]*b[1]*A[3];
};

#else
template<typename T>
__device__ inline void Log_p(T *p, T *q, T *x)
{
#pragma unroll
  for(int i=1; i<DIM; ++i)
    q[i] = d_q[id*DIM+i]; 
  T dot = q[0]*p[0];
#pragma unroll
  for(int i=1; i<DIM; ++i)
    dot += q[i]*p[i]; 
  dot = min(0.99999f,max(-0.99999f,dot));
  T theta = acosf(dot);
  T sinc=1.0f;
  if(theta > 1.e-8)
    sinc = theta/sinf(theta);
#pragma unroll
  for(int i=0; i<DIM; ++i)
    x[i] = (q[i] - p[i]*dot)*sinc;
}

/* computes b^TAb; 
 * assumes that A and b are in one piece in memory
 * written up for row-major A; but works for col major as well since
 * (b^TAb)^T = b^TA^Tb and row -> col major is transposing
 */
// TODO
//__device__ inline void bTAb(T *A, T *b)
//{
//  return b[0]*b[0]*A[0] + b[0]*b[1]*A[1] + b[0]*b[2]*A[2]
//        +b[1]*b[0]*A[3] + b[1]*b[1]*A[4] + b[1]*b[2]*A[5]
//        +b[2]*b[0]*A[6] + b[2]*b[1]*A[7] + b[2]*b[2]*A[8];
//};
#endif

/* just base function - empty because we are specializing if you look down */
template<typename T>
__device__ inline T atomicAdd_(T* address, T val)
{};

/* atomic add for double */
template<>
__device__ inline double atomicAdd_<double>(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
};

template<>
__device__ inline float atomicAdd_<float>(float* address, float val)
{
  return atomicAdd(address,val);
};

//template<>
//__device__ inline float atomicAdd_<int>(int* address, int val)
//{
//  return atomicAdd(address,val);
//};


