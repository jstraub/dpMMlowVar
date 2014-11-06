
#pragma once

#include <stdint.h>
#include <Eigen/Dense>

#include <boost/random/mersenne_twister.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "global.hpp"

#ifndef PI
#  define PI 3.141592653589793
#endif
#define LOG_PI 1.1447298858494002
#define LOG_2 0.69314718055994529
#define LOG_2PI 1.8378770664093453

using namespace Eigen;

template<typename T>
class Distribution
{
public:
  Distribution(boost::mt19937* pRndGen) : pRndGen_(pRndGen)
  {};
  virtual ~Distribution()
  {};

//  virtual logProb()
  boost::mt19937* pRndGen_;
private:
};


template<typename T, typename T2>
inline Matrix<T,Dynamic,1> counts(const Matrix<T2,Dynamic,1> & z, T2 K)
{
  Matrix<T,Dynamic,1> N(K);
  N.setZero(K);
  for (T2 i=0; i<z.size(); ++i)
    N(z(i))++;
  return N;
};

//inline VectorXd counts(const VectorXu& z, uint32_t K)
//{
//  VectorXd N(K);
//  N.setZero(K);
//  for (uint32_t i=0; i<z.size(); ++i)
//    N(z(i))++;
//  return N;
//};

/* multivariate gamma function of dimension p */
inline double lgamma_mult(double x,uint32_t p)
{
  assert(x+0.5*(1.-p) > 0.);
  double lgam_p = p*(p-1.)*0.25*LOG_PI;
  for (uint32_t i=1; i<p+1; ++i)
  {
//    cout<<"digamma_mult of "<<(x + (1.0-double(i))/2)<<" = "<<digamma(x + (1.0-double(i))/2)<<endl;
    lgam_p += boost::math::lgamma(x + 0.5*(1.0-double(i)));
  }
  return lgam_p;
}

template<typename T>
inline T logsumexp(T x1, T x2)                                   
{                                                                               
   if (x1>x2)                                                                   
      return x1 + log(1.+exp(x2-x1));                                            
   else                                                                         
      return x2 + log(1.+exp(x1-x2));                                            
}
