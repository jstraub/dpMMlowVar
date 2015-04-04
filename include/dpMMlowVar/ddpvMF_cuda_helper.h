#pragma once

#include <stdio.h>
#include <jsCore/cuda_global.h>

#define MAX_DOT 0.99
#define MIN_DOT -0.99

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
#endif

template<typename T, uint32_t I>
__device__ inline T distToUninstantiated( T zeta, T age, T beta, T w, T Q, T thresh)
{
  // solveProblem2(x_i, zeta, this->ts_[k], this->ws_[k], phi,theta,eta);
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  zeta = T phi + theta
  // for phi and theta

  T phi =  0.0;
  T dPhi = 0.0;
  for (uint32_t i=0; i< I; ++i)
  {
    T sinPhi = sin(phi);
    T cosPhi = cos(phi);
    T a = asin(beta*sinPhi);
    T b = asin(beta/w *sinPhi);
    T f = -zeta + asin(beta*sinPhi) + age * phi + asin(beta/w *sinPhi);
    T df = age + (beta*cosPhi)/sqrt(1.-beta*beta*sinPhi*sinPhi)
      + (beta*cosPhi)/sqrt(w*w - beta*beta*sinPhi*sinPhi);

    T phiPrev = phi;
    T dPhiPrev = dPhi;

    dPhi = f/df;
    phi = phi - dPhi; // Newton iteration
    printf("i=%d: prev: dPhi=%f; phi=%f; curr: dPhi=%f phi=%f zeta=%f; w=%f; Q=%f; f=%f; df=%f; sinPhi=%f; beta=%f %f %f \n",i,dPhiPrev,phiPrev,dPhi,phi,zeta,w,Q,f,df,sinPhi,beta,a,b);
//    printf("i=%d: dPhi=%f zeta=%f; age=%f; beta=%f; w=%f; Q=%f; thresh=%f; \n",i,dPhi,zeta,age,beta,w,Q,thresh);
    if(fabs(dPhi) < thresh) break;
  }

  T theta = asin(beta/w *sin(phi));
  T eta = asin(beta*sin(phi));

  return w*(cos(theta)-1.0) + age*(Q+beta*(cos(phi)-1.)) + cos(eta);
}

template<typename T >
__device__ inline T distToUninstantiatedSmallAngleApprox( T zeta, T age, T beta, T w, T Q)
{
  // solveProblem2(x_i, zeta, this->ts_[k], this->ws_[k], phi,theta,eta);
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  zeta = T phi + theta
  // for phi and theta

  T phi = zeta/ (beta*(1.+1./w) + age);
  T theta = zeta/( 1.+ w*(1. + age/beta) );
  T eta = zeta/(1. + 1./w + age/beta);

  return w*(cos(theta)-1.0) + age*(Q+beta*(cos(phi)-1.)) + cos(eta);
}
