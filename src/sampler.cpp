#include "sampler.hpp"

// ------------------------ impl ---------------------------------------------
template<typename T>
Sampler<T>::Sampler(boost::mt19937* pRndGen)
  : pRndGen_(pRndGen), selfManaged_(false)
{
  if(pRndGen_ == NULL)
  {
    selfManaged_ = true;
    pRndGen_ = new boost::mt19937(time(0));
  }
};

template<typename T>
Sampler<T>::~Sampler()
{ 
  if(selfManaged_) delete pRndGen_;
};


template<typename T>
void Sampler<T>::sampleUnif(Matrix<T,Dynamic,1>& r)
{
  for(uint32_t i=0; i<r.size(); ++i)
    r(i) = unif_(*pRndGen_);
};

template<typename T>
void Sampler<T>::sampleDiscLogPdfUnNormalized(const 
  Matrix<T,Dynamic,Dynamic>& logPdfs,VectorXu& z)
{
  uint32_t K = logPdfs.cols();
  if(K==1) {z.setZero(); return;}
  for(uint32_t i=0; i<logPdfs.rows(); ++i)
  {
    //Matrix<T,1,Dynamic> pdf(K);
    T normalizer = logSumExp<T>(logPdfs.row(i));
//    cout<<(logPdfs.row(i).array() - normalizer).array().exp()<<endl;
    
    T r = unif_(*pRndGen_);
    T cdf = exp(logPdfs(i,0)-normalizer);
    z(i) = K - 1; // if we do not break from the loop below
    for (uint32_t k=1; k<K; ++k)
    {
//      cout<<cdf<<" ";
      if(r <= cdf)
      {
        z(i) = k-1;
        break;
      }
      cdf += exp(logPdfs(i,k)-normalizer);
    }
//    cout<<endl;
  }
}

template<typename T>
void Sampler<T>::sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, VectorXu& z)
{
  uint32_t K = pdfs.cols();
  if(K==1) {z.setZero(); return;}
  for(uint32_t i=0; i<pdfs.rows(); ++i)
  {
    //cout<<pdfs.row(i)<<endl;
    T r = unif_(*pRndGen_);
    T cdf = pdfs(i,0);
    z(i) = K -1; // if we do not break from the loop below
    for (uint32_t k=1; k<K; ++k)
    {
      if(r <= cdf)
      {
        z(i) = k-1;
        break;
      }
      cdf += pdfs(i,k);
    }
  }
};

template class Sampler<double>;
template class Sampler<float>;


#ifdef CUDA
// ----------------------------------------------------------------------------
template<typename T>
SamplerGpu<T>::SamplerGpu(uint32_t N, uint32_t K, boost::mt19937* pRndGen)
  : Sampler<T>(pRndGen), pdfs_(new GpuMatrix<T>(N,K)), logNormalizers_(N,1),
  z_(N), r_(N)
{};

template<typename T>
SamplerGpu<T>::~SamplerGpu()
{};

template<typename T>
void SamplerGpu<T>::setPdfs(const boost::shared_ptr<GpuMatrix<T> >& pdfs, bool logScale)
{
  pdfs_ = pdfs;
  z_.resize(pdfs_->rows(),1);
};

template<typename T>
void SamplerGpu<T>::sampleUnif(Matrix<T,Dynamic,1>& r)
{
  assert(r.size() == r_.rows());
  unifGpu(r_.data(), r_.rows(), 
    static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));
  r_.get(r);
};

template<typename T>
void SamplerGpu<T>::sampleDiscPdf(T *d_pdfs, const spVectorXu& z, bool logScale)
{
  if(logScale)
    choiceMultLogPdfGpu(d_pdfs, z_.data(), pdfs_->rows(), pdfs_->cols(),
      static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));
  else
    choiceMultGpu(d_pdfs, z_.data(), pdfs_->rows(), pdfs_->cols(),
      static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));
  z_.get(z);
};

template<typename T>
void SamplerGpu<T>::sampleDiscPdf()
{
//  pdfs_->print();
//  z_.print();
  if(!z_.isInit()){
    z_.resize(pdfs_->rows(),1);
    z_.setZero();
  }
  // internally we use logPdf
  choiceMultLogPdfGpu(pdfs_->data(), z_.data(), pdfs_->rows(), pdfs_->cols(),
      static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));
};

template<typename T>
void SamplerGpu<T>::sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, 
    VectorXu& z, bool logScale)
{
  if(pdfs.cols()==1) {z.setZero(); return;}
  pdfs_->set(pdfs);
  if(!z_.isInit()){z_.set(z);}
//cout<<pdfs<<endl;
  if(logScale)
    choiceMultLogPdfGpu(pdfs_->data(), z_.data(), pdfs_->rows(), pdfs_->cols(),
      static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));
  else
    choiceMultGpu(pdfs_->data(), z_.data(), pdfs_->rows(), pdfs_->cols(),
      static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));

  z_.get(z);
//cout<<z<<endl;
};

//template<typename T>
//void SamplerGpu<T>::sampleDiscPdf(const Matrix<T,Dynamic,Dynamic>& pdfs, 
//    const spVectorXu& z)
//{
//  if(pdfs.cols()==1) {z.setZero(); return;}
//  pdfs_->set(pdfs);
//  if(!z_.isInit()){z_.set(z);}
////cout<<pdfs<<endl;
//  choiceMultGpu(pdfs_->data(), z_.data(), pdfs_->rows(), pdfs_->cols(),
//    static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));
//  z_.get(z);
////  VectorXu zz(z_.rows());
////  z_.get(zz);
////  cout<<(*z).transpose()<<endl;
////  cout<<(zz).transpose()<<endl;
//};

template<typename T>
void SamplerGpu<T>::sampleDiscLogPdfUnNormalized(const 
  Matrix<T,Dynamic,Dynamic>& logPdfs,VectorXu& z)
{
  if(logPdfs.cols()==1) {z.setZero(); return;}
  pdfs_->set(logPdfs);
  if(!z_.isInit()){z_.set(z);}
  choiceMultLogPdfUnNormalizedGpu(pdfs_->data(), z_.data(), pdfs_->rows(), 
    pdfs_->cols(), static_cast<uint32_t>(floor(unif_(*this->pRndGen_)*4294967296)));
  z_.get(z);
}


template<typename T>
void SamplerGpu<T>::logNormalizer(uint32_t dk, uint32_t K)
{
  logNormalizers_.resize(pdfs_->rows(),K/dk);
  logNormalizers_.setZero();
  logNormalizerGpu(pdfs_->data(),logNormalizers_.data(),dk,K,pdfs_->rows());
};

template<typename T>
void SamplerGpu<T>::logNormalize(uint32_t dk, uint32_t K)
{
  logNormalizeGpu(pdfs_->data(),dk,K,pdfs_->rows());
};

template<typename T>
void SamplerGpu<T>::addTopLevel(const Matrix<T,Dynamic,1>& pi,uint32_t dk)
{ 
  uint32_t K = pi.rows()*dk;
  assert(pdfs_->cols() == K);
//  logNormalizer(dk,K); // compute logNormalizers over the dk sets
  cout<<logNormalizers_.get().topRows(30)<<endl;

  // add the logNormalizers and logPi into the pdfs
  GpuMatrix<T> d_pi(pi);
  logAddTopLevelGpu(pdfs_->data(),logNormalizers_.data(),d_pi.data(),dk,K,
      pdfs_->rows());
};

template class SamplerGpu<double>;
template class SamplerGpu<float>;

#endif 
