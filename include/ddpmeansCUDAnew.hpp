#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "ddpmeans.hpp"
#include "clDataGpu.hpp"
#include "gpuMatrix.hpp"


using namespace Eigen;
using std::cout;
using std::endl;


extern void ddpLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, double *d_ages, double *d_ws, double lambda,
    double Q, double tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);
extern void ddpLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, float *d_ages, float *d_ws, float lambda, float Q, 
    float tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);

template<class T, class DS>
class DDPMeansCUDA : public DDPMeans<T,DS>
{
public:
//  DDPMeansCUDA(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx,
//      T lambda, T Q, T tau);
  DDPMeansCUDA(const shared_ptr<ClDataGpu<T> >& cld,
      T lambda, T Q, T tau);
  virtual ~DDPMeansCUDA();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

//  virtual void updateLabelsParallel();
//  virtual void updateLabels();
//  virtual void updateCenters();
//  virtual void nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);

  //TODO: this should be solved using constructors for ClData!
//  virtual void nextTimeStep(T* d_x, uint32_t N, uint32_t step, uint32_t offset);

//  virtual void updateState(); // after converging for a single time instant
//  virtual uint32_t indOfClosestCluster(int32_t i);
//
//  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
//  virtual bool closer(T a, T b);
  
  void getZfromGpu() {this->cld_->z();};
  uint32_t* d_z(){ return this->cdl_->d_z();};
  
protected:

  static const uint32_t MAX_UINT32 = 4294967295;

//  GpuMatrix<T> d_x_;
//  GpuMatrix<uint32_t> d_z_;
  GpuMatrix<uint32_t> d_iAction_;
  GpuMatrix<T> d_ages_;
  GpuMatrix<T> d_ws_;
  GpuMatrix<uint32_t> d_Ns_;
  GpuMatrix<T> d_p_;

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);
//  virtual void computeSums(uint32_t k0, uint32_t K); 
//  virtual void computeSums(void); // updates internal xSums_ 
  uint32_t computeLabelsGPU(uint32_t i0);

  // TODO: implement
//  virtual void reInstantiatedOldCluster(const Matrix<T,Dynamic,1>& xSum, uint32_t k);

};
// --------------------------- impl -------------------------------------------

//template<class T, class DS>
//DDPMeansCUDA<T,DS>::DDPMeansCUDA(
//    const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, T lambda, T Q, T tau, 
//    mt19937* pRndGen)
//  : DDPMeans<T,DS>(spx,lambda,Q,tau), d_x_(spx), d_z_(this->N_),
//  d_iAction_(1), d_ages_(1), d_ws_(1), d_Ns_(1), d_p_(this->D_,1)
//{}

template<class T, class DS>
DDPMeansCUDA<T,DS>::DDPMeansCUDA(const shared_ptr<ClDataGpu<T> >& cld,
      T lambda, T Q, T tau)
  : DDPMeans<T,DS>(cld,lambda,Q,tau), //d_x_(cld), d_z_(this->N_),
  d_iAction_(1), d_ages_(1), d_ws_(1), d_Ns_(1), d_p_(this->D_,1)
{}

template<class T, class DS>
DDPMeansCUDA<T,DS>::~DDPMeansCUDA()
{}

//template<class T, class DS>
//void DDPMeansCUDA<T,DS>::computeSums(uint32_t k0, uint32_t K)
//{
////  cout<<"CUDA ::computeSums for k0="<<k0<<" K="<<K<<" N="<<this->N_<<endl;
//  Matrix<T,Dynamic,Dynamic> xSums = Matrix<T,Dynamic,Dynamic>::Zero(
//      this->D_+1,K);
//  GpuMatrix<T> d_xSums(xSums);
//
////  d_x_.print();
////  d_z_.print();
//  
//  vectorSum_gpu(d_x_.data(),d_z_.data(),this->N_,k0,K,d_xSums.data());
//  d_xSums.get(xSums); 
////  cout<<xSums<<endl; 
//  for (uint32_t k=0; k<K; ++k)
//  {
////    this->Ns_(k+k0) = Ss(3,k); //TODO do I need counts?
//    this->xSums_.col(k+k0) = xSums.block(0,k,3,1);
//    this->Ns_(k+k0) = xSums(3,k);
//  }
//}
//
//
//template<class T, class DS>
//void DDPMeansCUDA<T,DS>::computeSums(void)
//{
//  prevNs_ =  this->Ns_;
//
//  this->xSums_ = Matrix<T,Dynamic,Dynamic>::Zero(this->D_, this->K_);
//  uint32_t k0 = 0;
//  if(this->K_ <= 6)
//  {
//    computeSums(0,this->K_); // max 6 SSs per kernel due to shared mem
//  }else{
//    for (k0=0; k0<this->K_; k0+=6)
//      computeSums(k0,min(this->K_-k0,uint32_t(6))); // max 6 SSs per kernel due to shared mem
//  }
//}
//
//template<class T, class DS>
//void DDPMeansCUDA<T,DS>::nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
//{
//  DDPMeans<T,DS>::nextTimeStep(spx);
//  d_x_.set(this->spx_); // copy to GPU
//  d_z_.set(this->z_);
//};

//template<class T, class DS>
//void DDPMeansCUDA<T,DS>::nextTimeStep(T* d_x, uint32_t N, uint32_t step, uint32_t offset)
//{
//  //TODO: this should be solved using constructors for ClData!
//  assert(false);
//// copy from other array with N cols/elems and "step" rows 
////  d_x_.copyFromGpu(d_x,N,step,offset,3);
////  this->spx_->resize(3,N);
////  d_x_.get(*(this->spx_)); // copy it for parallel labeling
////
////  DDPMeans<T,DS>::nextTimeStep(this->spx_);
////
////  d_z_.set(this->z_);
//};

template<class T, class DS>
uint32_t DDPMeansCUDA<T,DS>::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = MAX_UINT32;
  d_iAction_.set(iAction);
  d_Ns_.set(this->counts());
  d_ages_.set(this->ages());
  d_ws_.set(this->weights());

  // TODO not too too sure about this
  Matrix<T,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->isInstantiated())
      ps.col(k) = this->cls_[k]->centroid();
    else if(!this->cls_[k]->isInstantiated() && !this->cls_[k]->isNew())
      ps.col(k) = this->clsPrev_[k]->centroid();
  d_p_.set(ps);

  assert(this->K_ < 17); // limitation of kernel at this point

//  cout<<"ddpvMFlabels_gpu K="<<this->K_<<endl;
//  cout<<ps<<endl;
//  d_p_.print();
//  d_ages_.print();
//  d_Ns_.print();
//  cout<<"d_Ns_ "<<d_Ns_.get().transpose()<<endl;
//  cout<<"d_ages_ "<<d_ages_.get().transpose()<<endl;

//  cout << "******************BEFORE*******************"<<endl;
  ddpLabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(), 
      d_Ns_.data(), d_ages_.data(), d_ws_.data(), this->cl0_.lambda(), 
      this->cl0_.Q(), this->cl0_.tau(), 0, this->K_, i0, this->N_-i0, 
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<class T, class DS>
uint32_t DDPMeansCUDA<T,DS>::optimisticLabelsAssign(uint32_t i0)
{
  return computeLabelsGPU(0); // TODO make passing i0 work!
};

//template<class T, class DS>
//void DDPMeansCUDA<T,DS>::updateCenters()
//{
//  d_z_.get(this->z_);
//  DDPMeans<T,DS>::updateCenters();
//};



 
