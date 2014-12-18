#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "ddpmeans.hpp"
#include "gpuMatrix.hpp"


using namespace Eigen;
using std::cout;
using std::endl;


extern void vectorSum_gpu( double *d_x, uint32_t *d_z , uint32_t N, 
    uint32_t k0, uint32_t K, double *d_SSs);
extern void vectorSum_gpu(float *d_x, uint32_t *d_z, 
    uint32_t N, uint32_t k0, uint32_t K, float *d_SSs);

extern void ddpLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, double *d_ages, double *d_ws, double lambda,
    double Q, double tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);
extern void ddpLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, float *d_ages, float *d_ws, float lambda, float Q, 
    float tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);

template<class T>
class DDPMeansCUDA : public DDPMeans<T>
{
public:
  DDPMeansCUDA(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx,
      T lambda, T Q, T tau, mt19937* pRndGen);
  virtual ~DDPMeansCUDA();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

//  virtual void updateLabelsParallel();
  virtual void updateLabels();
//  virtual void updateCenters();
  virtual void nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);
  virtual void nextTimeStep(T* d_x, uint32_t N, uint32_t step, uint32_t offset);
//  virtual void updateState(); // after converging for a single time instant
//  virtual uint32_t indOfClosestCluster(int32_t i);
//
//  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
//  virtual bool closer(T a, T b);
  
  // TODO approximate !
  virtual bool converged(T eps=1e-6) 
  {
    return this->Ns_.size() > 0 && this->Ns_.size() == prevNs_.size() && (prevNs_.array() == this->Ns_.array()).all();
  };

  void getZfromGpu(){this->z_.resize(d_z_.rows()); d_z_.get(this->z_);};
  uint32_t* d_z(){ return d_z_.data();};
  
protected:

  static const uint32_t MAX_UINT32 = 4294967295;

  GpuMatrix<T> d_x_;
  GpuMatrix<uint32_t> d_z_;
  GpuMatrix<uint32_t> d_iAction_;
  GpuMatrix<T> d_ages_;
  GpuMatrix<T> d_ws_;
  GpuMatrix<uint32_t> d_Ns_;
  GpuMatrix<T> d_p_;

  VectorXu prevNs_;

  // TODO initialize propperly!
  Matrix<T,Dynamic,Dynamic> xSums_;
  std::vector<uint32_t> globalInd_; // for each non-dead cluster the global id of it;
  uint32_t globalMaxInd_;

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);
  virtual void computeSums(uint32_t k0, uint32_t K); 
  virtual void computeSums(void); // updates internal xSums_ 
  uint32_t computeLabelsGPU(uint32_t i0);

  // TODO: implement
  virtual void reInstantiatedOldCluster(const Matrix<T,Dynamic,1>& xSum, uint32_t k);

};
// --------------------------- impl -------------------------------------------

template<class T>
DDPMeansCUDA<T>::DDPMeansCUDA(
    const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, T lambda, T Q, T tau, 
    mt19937* pRndGen)
  : DDPMeans<T>(spx,lambda,Q,tau,pRndGen), d_x_(spx), d_z_(this->N_),
  d_iAction_(1), d_ages_(1), d_ws_(1), d_Ns_(1), d_p_(this->D_,1)
{}

template<class T>
DDPMeansCUDA<T>::~DDPMeansCUDA()
{}

template<class T>
void DDPMeansCUDA<T>::computeSums(uint32_t k0, uint32_t K)
{
//  cout<<"CUDA ::computeSums for k0="<<k0<<" K="<<K<<" N="<<this->N_<<endl;
  Matrix<T,Dynamic,Dynamic> xSums = Matrix<T,Dynamic,Dynamic>::Zero(
      this->D_+1,K);
  GpuMatrix<T> d_xSums(xSums);

//  d_x_.print();
//  d_z_.print();
  
  vectorSum_gpu(d_x_.data(),d_z_.data(),this->N_,k0,K,d_xSums.data());
  d_xSums.get(xSums); 
//  cout<<xSums<<endl; 
  for (uint32_t k=0; k<K; ++k)
  {
//    this->Ns_(k+k0) = Ss(3,k); //TODO do I need counts?
    this->xSums_.col(k+k0) = xSums.block(0,k,3,1);
    this->Ns_(k+k0) = xSums(3,k);
  }
}


template<class T>
void DDPMeansCUDA<T>::computeSums(void)
{
  prevNs_ =  this->Ns_;

  this->xSums_ = Matrix<T,Dynamic,Dynamic>::Zero(this->D_, this->K_);
  uint32_t k0 = 0;
  if(this->K_ <= 6)
  {
    computeSums(0,this->K_); // max 6 SSs per kernel due to shared mem
  }else{
    for (k0=0; k0<this->K_; k0+=6)
      computeSums(k0,min(this->K_-k0,uint32_t(6))); // max 6 SSs per kernel due to shared mem
  }
}

template<class T>
void DDPMeansCUDA<T>::nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
{
  DDPMeans<T>::nextTimeStep(spx);
//  cout<<"copy x"<<endl;
//  d_x_.print();
//  d_z_.print();
  d_x_.set(this->spx_); // copy to GPU
//  d_z_.set(this->z_);
//  cout<<"resize z"<<endl;
  d_z_.resize(this->N_,1);
  if (!d_z_.isInit())
  {
    d_z_.setZero(); // to make sure we are initialized
//    cout<<"set z to Zero"<<endl;
  }
};

template<class T>
void DDPMeansCUDA<T>::nextTimeStep(T* d_x, uint32_t N, uint32_t step, uint32_t offset)
{
  this->psPrev_ = this->ps_;
  this->Kprev_ = this->K_;
  // TODO: hopefully this does not mess stuff up
//  DDPMeans<T>::nextTimeStep(spx);
  this->N_ = N;
//  cout<<"copy x"<<endl;
//  d_x_.print();
//  d_z_.print();
//  d_x_.set(this->spx_); 
// copy from other array with N cols/elems and "step" rows 
  d_x_.copyFromGpu(d_x,N,step,offset,3);
  this->spx_->resize(3,this->N_);
  d_x_.get(*(this->spx_)); // copy it for parallel labeling
//  d_z_.set(this->z_);
//  cout<<"resize z"<<endl;
//  d_z_.copyFromGpu(d_z,N,1,0,1);
  d_z_.resize(this->N_,1);
  if (!d_z_.isInit())
  {
    d_z_.setZero(); // to make sure we are initialized
//    cout<<"set z to Zero"<<endl;
  }
//  d_x_.print();
//  d_z_.print();
};

//template<class T>
//void DDPMeansCUDA<T>::updateLabels()
//{
//  DDPMeans<T>::updateLabels();
////  d_z_.set(this->z_); // TODO i dont think I need to copy back
//};


template<class T>
uint32_t DDPMeansCUDA<T>::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = MAX_UINT32;
  d_iAction_.set(iAction);
  d_Ns_.set(this->Ns_);
  d_ages_.set(this->ts_);
  d_ws_.set(this->ws_);
  Matrix<T,Dynamic,Dynamic> ps = this->ps_;
  for(uint32_t k=0; k<this->ts_.size(); ++k)
    if(this->Ns_[k] == 0)
      ps.col(k) = this->psPrev_.col(k);
  d_p_.set(ps);

  assert(this->K_ < 17); // limitation of kernel at this point

//  cout<<"ddpvMFlabels_gpu K="<<this->K_<<endl;
//  cout<<this->ps_<<endl;
//  d_x_.print();
//  d_p_.print();
//  d_z_.print();
//  d_ages_.print();
//  d_Ns_.print();

  ddpLabels_gpu( d_x_.data(),  d_p_.data(),  d_z_.data(), 
      d_Ns_.data(), d_ages_.data(), d_ws_.data(), this->lambda_, this->Q_, 
      this->tau_, 0, this->K_, i0, this->N_-i0, d_iAction_.data());
  d_iAction_.get(iAction); 
  return iAction;
}

template<class T>
uint32_t DDPMeansCUDA<T>::optimisticLabelsAssign(uint32_t i0)
{
  return computeLabelsGPU(0); // TODO make passing i0 work!
};


template<class T>
void DDPMeansCUDA<T>::updateLabels()
{
  uint32_t idAction = UNASSIGNED;
  uint32_t i0 = 0;
//  cout<<"::updateLabelsParallel"<<endl;
  do{
    idAction = optimisticLabelsAssign(i0);
//  cout<<"::updateLabelsParallel:  idAction: "<<idAction<<endl;
    if(idAction != UNASSIGNED)
    {
      T sim = 0.;
      uint32_t z_i = this->indOfClosestCluster(idAction,sim);
      if(z_i == this->K_) 
      { // start a new cluster
        this->ps_.conservativeResize(this->D_,this->K_+1);
        this->Ns_.conservativeResize(this->K_+1); 
        this->ps_.col(this->K_) = this->spx_->col(idAction);
        this->Ns_(z_i) = 1.;
        this->globalInd_.push_back(this->globalMaxInd_++);
        this->K_ ++;
      } 
      else if(this->Ns_[z_i] == 0)
      { // instantiated an old cluster
        reInstantiatedOldCluster(this->spx_->col(idAction), z_i);
        this->Ns_(z_i) = 1.; // set Ns of revived cluster to 1 tosignal
        // computeLabelsGPU to use the cluster;
      }
      i0 = idAction;
    }
    cout<<" K="<<this->K_<<" Ns="<<this->Ns_.transpose()<< " i0="<<i0<<endl;
  }while(idAction != UNASSIGNED);
//  this->z_.resize(this->N_);
//  d_z_.set(this->z_); // TODO i dont think I need to copy back

//  // TODO: this cost only works for a single time slice
//  T cost =  0.0;
//  for(uint32_t k=0; k<this->K_; ++k)
//    if(this->Ns_(k) == 1.) cost += this->lambda_;
//
//  //TODO get counts from GPU
//  this->Ns_.fill(0);
//#pragma omp parallel for reduction(+:cost)
//  for(uint32_t k=0; k<this->K_; ++k)
//    for(uint32_t i=0; i<this->N_; ++i)
//      if(this->z_(i) == k)
//      {
//        this->Ns_(k) ++; 
//        T sim_closest = dist(this->ps_.col(k), this->spx_->col(i));
//        cost += sim_closest;
//      }
//  this->prevCost_ = this->cost_;
//  this->cost_ = cost;
};

//template<class T>
//void DDPMeansCUDA<T>::updateLabelsParallel()
//{
//  uint32_t idAction = MAX_UINT32;
////  cout<<"::updateLabelsParallel"<<endl;
//  do{
////    if(idAction == MAX_UINT32)
////    {
//      idAction = computeLabelsGPU(0);
////    }else{
////      idAction = computeLabelsGPU(idAction-1);
////    }
////  cout<<"::updateLabelsParallel:  idAction: "<<idAction<<endl;
//    if(idAction != MAX_UINT32)
//    {
//      uint32_t z_i = this->indOfClosestCluster(idAction);
//      if(z_i == this->K_) 
//      { // start a new cluster
//        this->ps_.conservativeResize(this->D_,this->K_+1);
//        this->Ns_.conservativeResize(this->K_+1); 
//        this->ps_.col(this->K_) = this->spx_->col(idAction);
//        this->Ns_(z_i) = 1.;
//        this->K_ ++;
//      } 
//      else if(this->Ns_[z_i] == 0)
//      { // instantiated an old cluster
//        reInstantiatedOldCluster(this->spx_->col(idAction), z_i);
//        this->Ns_(z_i) = 1.; // set Ns of revived cluster to 1 tosignal
//        // computeLabelsGPU to use the cluster;
//      }
//    }
//    cout<<" K="<<this->K_<<" Ns="<<this->Ns_.transpose()<<endl;
//  }while(idAction != MAX_UINT32);
//};

 
