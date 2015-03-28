#pragma once

#include <stdio.h>
#include <dpMMlowVar/cuda_global.h>

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

template<typename T, uint32_t I>
__device__ inline T distToUninstantiatedSmallAngleApprox( T zeta, T age, T beta, T w, T Q, T thresh)
{
  // solveProblem2(x_i, zeta, this->ts_[k], this->ws_[k], phi,theta,eta);
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  zeta = T phi + theta
  // for phi and theta


  T phi = zeta/ (beta*(1.+1./w) + age);
  T theta = zeta/( 1.+ w*(1. + age/beta) );
  T eta = zeta/(1. + 1./w + age/beta);
  // DONE: hacky -- could solve this analytically!

//  T phi =  0.0;
//  T dPhi = 0.0;
//  for (uint32_t i=0; i< I; ++i)
//  {
////    T a = (beta*phi);
////    T b = (beta/w *phi);
//    T f = -zeta + (beta*phi) + (age * phi) + (beta/w *phi);
//    T df = beta + age + (beta/w);
////    T df = age + (beta*cosPhi)/sqrt(1.-beta*beta*sinPhi*sinPhi)
////      + (beta*cosPhi)/sqrt(w*w - beta*beta*sinPhi*sinPhi);
//
////    T phiPrev = phi;
////    T dPhiPrev = dPhi;
//
//    dPhi = f/df;
//    phi = phi - dPhi; // Newton iteration
////    printf("i=%d: prev: dPhi=%f; phi=%f; curr: dPhi=%f phi=%f zeta=%f; w=%f; Q=%f; f=%f; df=%f; beta=%f %f %f \n",i,dPhiPrev,phiPrev,dPhi,phi,zeta,w,Q,f,df,beta,a,b);
////    printf("i=%d: dPhi=%f zeta=%f; age=%f; beta=%f; w=%f; Q=%f; thresh=%f; \n",i,dPhi,zeta,age,beta,w,Q,thresh);
//    if(fabs(dPhi) < thresh) break;
//  }
//
//  T theta = asin(beta/w *sin(phi));
//  T eta = asin(beta*sin(phi));

  return w*(cos(theta)-1.0) + age*(Q+beta*(cos(phi)-1.)) + cos(eta);
}
