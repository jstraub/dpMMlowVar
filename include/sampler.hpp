#pragma once

#include <iostream>
#include <time.h>
#include <Eigen/Dense>

#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_int_distribution.hpp>
//#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>

#include "distribution.hpp"
#include "global.hpp"

#ifdef CUDA
#include "gpuMatrix.hpp"
#endif

using namespace Eigen;
using std::cout;
using std::endl;


#ifdef BOOST_OLD
#define mt19937 boost::mt19937
#else
using boost::mt19937;
#endif

/* Sampler for CPU */
template<typename T=double>
class Sampler
{
protected:
  mt19937* pRndGen_;
  bool selfManaged_;
  boost::uniform_01<> unif_;

public:
  Sampler(mt19937* pRndGen=NULL);
  virtual ~Sampler();

  virtual void sampleUnif(Matrix<T,Dynamic,1>& r);
  virtual Matrix<T,Dynamic,1> sampleUnif(uint32_t N)
  {
    Matrix<T,Dynamic,1> u(N);
    sampleUnif(u);
    return u;
  };

  virtual void sampleDiscLogPdfUnNormalized(const Matrix<T,Dynamic,Dynamic>& pdfs,
    VectorXu& z);
  virtual VectorXu sampleDiscLogPdfUnNormalized(
    const Matrix<T,Dynamic,Dynamic>& pdfs)
  { 
    VectorXu z(pdfs.rows());
    sampleDiscLogPdfUnNormalized(pdfs,z);
    return z;
  };

  virtual void sampleDiscPdf(T *d_pdfs, const spVectorXu& z){ assert(false);};
  virtual void sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, VectorXu& z);
  virtual VectorXu sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs)
  {
    VectorXu z(pdfs.rows());
    sampleDiscPdf(pdfs,z);
    return z;
  };
};

#ifdef CUDA
#include "gpuMatrix.hpp"

extern void choiceMultGpu(double* d_pdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed);
extern void choiceMultGpu(float* d_pdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed);
extern void choiceMultLogPdfGpu(double* d_logPdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed);
extern void choiceMultLogPdfGpu(float* d_logPdf, uint32_t* d_z, uint32_t N, 
    uint32_t M, uint32_t seed);
extern void choiceMultLogPdfUnNormalizedGpu(double* d_pdf, uint32_t* d_z, 
  uint32_t N, uint32_t M, uint32_t seed);
extern void choiceMultLogPdfUnNormalizedGpu(float* d_pdf, uint32_t* d_z, 
  uint32_t N, uint32_t M, uint32_t seed);
extern void unifGpu(double* d_u, uint32_t N, uint32_t seed);
extern void unifGpu(float* d_u, uint32_t N, uint32_t seed);

extern void logNormalizerGpu(float* d_logPdf, float* d_logNormalizer, 
    uint32_t dk, uint32_t K, uint32_t N);
extern void logNormalizerGpu(double* d_logPdf, double* d_logNormalizer, 
    uint32_t dk, uint32_t K, uint32_t N);
extern void logNormalizeGpu(float* d_logPdf, uint32_t dk, uint32_t K, uint32_t N);
extern void logNormalizeGpu(double* d_logPdf, uint32_t dk, uint32_t K, uint32_t N);
extern void logAddTopLevelGpu(float* d_logPdf, float* d_logNormalizer, 
    float* d_logPi, uint32_t dk, uint32_t K, uint32_t N);
extern void logAddTopLevelGpu(double* d_logPdf, double* d_logNormalizer, 
    double* d_logPi, uint32_t dk, uint32_t K, uint32_t N);


/* Sampler for GPU */
template<typename T=float>
class SamplerGpu : public Sampler<T>
{
  shared_ptr<GpuMatrix<T> > pdfs_; // one pdf per row
  GpuMatrix<T> logNormalizers_; // one pdf per row
  GpuMatrix<uint32_t> z_; // samples from pdfs
  GpuMatrix<T> r_; // unif random numbers

public:
  SamplerGpu(uint32_t N, uint32_t K, mt19937* pRndGen=NULL);
  ~SamplerGpu();

  void sampleUnif(Matrix<T,Dynamic,1>& r);
  Matrix<T,Dynamic,1> sampleUnif()
  {
    Matrix<T,Dynamic,1> u(r_.rows());
    sampleUnif(u);
    return u;
  };

  virtual void setPdfs(const shared_ptr<GpuMatrix<T> >& pdfs, bool logScale);

  virtual void sampleDiscLogPdfUnNormalized(
      const Matrix<T,Dynamic,Dynamic>& pdfs, VectorXu& z);
  /* use pdfs already prestored in GPU memory */
  void sampleDiscPdf(T *d_pdfs, const spVectorXu& z, bool logScale=false);
//  void sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, const spVectorXu& z);
  void sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, VectorXu& z, 
      bool logScale = false);
  VectorXu sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, 
      bool logScale=false)
  {
    VectorXu z(pdfs.rows());
    sampleDiscPdf(pdfs,z,logScale);
    return z;
  };
  // sample from internal pdfs_
  void sampleDiscPdf();

  // compute logNormalizer over blocks of dk columns
  void logNormalizer(uint32_t dk, uint32_t K);
  // log normalize the distributions in pdfs_ in blocks of dk cols
  void logNormalize(uint32_t dk, uint32_t K);
  // add logNormalizer+pi to pdfs_ -> add one layer of hierarchy
  void addTopLevel(const Matrix<T,Dynamic,1>& pi,uint32_t dk);

  // get Z from memory
  void getZ( VectorXu& z){ z_.get(z);};
};
#endif

