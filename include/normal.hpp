#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <algorithm>

#include <iostream>

#include <boost/random/normal_distribution.hpp>

#include "distribution.hpp"

using namespace Eigen;
using std::cout;
using std::endl;
using std::min;

#ifdef BOOST_OLD
using boost::normal_distribution;
#else
using boost::random::normal_distribution;
#endif

template<typename T>
class Normal : public Distribution<T>
{
public:
  uint32_t  D_;
  Matrix<T,Dynamic,1> mu_;

  Normal(const Matrix<T,Dynamic,1>& mu, const Matrix<T,Dynamic,Dynamic>& Sigma,
      boost::mt19937 *pRndGen);
  Normal(const Matrix<T,Dynamic,Dynamic>& Sigma, boost::mt19937 *pRndGen);
  Normal(uint32_t D, boost::mt19937 *pRndGen);
  Normal(const Normal<T>& other);
  ~Normal();

  T logPdf(const Matrix<T,Dynamic,Dynamic>& x) const;
  T logPdfSlower(const Matrix<T,Dynamic,Dynamic>& x) const;
  T logPdf(const Matrix<T,Dynamic,Dynamic>& scatter, 
      const Matrix<T,Dynamic,1>& mean, T count) const;
  T logPdf(const Matrix<T,Dynamic,Dynamic>& scatter, T count) const;

  Matrix<T,Dynamic,1> sample();

  void print() const;

  const Matrix<T,Dynamic,Dynamic>& Sigma() const {return Sigma_;};
  void setSigma(const Matrix<T,Dynamic,Dynamic>& Sigma)
  { Sigma_ = Sigma; SigmaLDLT_.compute(Sigma_); 
    logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();};
  T logDetSigma() const {return logDetSigma_;};
  const LDLT<Matrix<T,Dynamic,Dynamic> >& SigmaLDLT() const {return SigmaLDLT_;};

private:

  Matrix<T,Dynamic,Dynamic> Sigma_;
  // helpers for fast computation
  T logDetSigma_;
  LDLT<Matrix<T,Dynamic,Dynamic> > SigmaLDLT_;

  normal_distribution<> gauss_;
};

typedef Normal<double> Normald;
typedef Normal<float> Normalf;

template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClusters(
    Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t K);

template<class T>
inline Matrix<T,Dynamic,Dynamic> sampleClusters(
    Matrix<T,Dynamic,Dynamic>& x, VectorXu& z, uint32_t K)
{
  uint32_t N = x.cols();
  uint32_t D = x.rows();
  boost::mt19937 rndGen(9119);

  Matrix<T,Dynamic,Dynamic> Sigma = Matrix<T,Dynamic,Dynamic>::Identity(D,D);
  Sigma *= 5.0;
  Normal<T> meanPrior(Sigma,&rndGen);

  Sigma *= 1.0/20.;
  Matrix<T,Dynamic,Dynamic> mus(D,K);
  for(uint32_t k=0; k<K; ++k)
  {
    mus.col(k) = meanPrior.sample();
    Normal<T> gauss_k(mus.col(k),Sigma,&rndGen);
    for (uint32_t i=k*(N/K); i<min(N,(k+1)*(N/K)+N%K); ++i) 
    {
      x.col(i) = gauss_k.sample();
      z(i) = k;
    }
  }
  return mus;
};

