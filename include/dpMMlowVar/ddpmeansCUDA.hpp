#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <jsCore/global.hpp>
#include <jsCore/clDataGpu.hpp>
#include <jsCore/gpuMatrix.hpp>

#include <dpMMlowVar/ddpmeans.hpp>
#include <dpMMlowVar/euclideanData.hpp>
#include <dpMMlowVar/sphericalData.hpp>


using namespace Eigen;
using std::cout;
using std::endl;

// for spherical space 
void ddpvMFlabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z,
    uint32_t *d_Ns, double *d_ages, double *d_ws, double lambda, double beta,
    double Q, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t
    *d_iAction);
void ddpvMFlabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, uint32_t
    *d_Ns, float *d_ages, float *d_ws, float lambda, float beta, float Q,
    uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);


// for Eucledian space
extern void ddpLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, double *d_ages, double *d_ws, double lambda,
    double Q, double tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);
extern void ddpLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, float *d_ages, float *d_ws, float lambda, float Q, 
    float tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);

extern void ddpLabelsSpecial_gpu( double *d_q,  double *d_oldp, double *d_ages,
    double *d_ws, double lambda, double Q, double tau, uint32_t K, uint32_t N,
    uint32_t *d_asgnIdces);
extern void ddpLabelsSpecial_gpu( float *d_q,  float *d_oldp, float *d_ages,
    float *d_ws, float lambda, float Q, float tau, uint32_t K, uint32_t N,
    uint32_t *d_asgnIdces);

extern void ddpvMFLabelsSpecial_gpu( double *d_q,  double *d_oldp, double *d_ages,
    double *d_ws, double lambda, double Q, double tau, uint32_t K, uint32_t N,
    uint32_t *d_asgnIdces);
extern void ddpvMFLabelsSpecial_gpu( float *d_q,  float *d_oldp, float *d_ages,
    float *d_ws, float lambda, float Q, float tau, uint32_t K, uint32_t N,
    uint32_t *d_asgnIdces);


namespace dplv {

template<class T, class DS>
class DDPMeansCUDA : public DDPMeans<T,DS>
{
public:
  DDPMeansCUDA(const shared_ptr<jsc::ClDataGpu<T> >& cld,
      T lambda, T Q, T tau);
  virtual ~DDPMeansCUDA();
  
  virtual void nextTimeStepGpu(T* d_x, uint32_t N, uint32_t step, 
      uint32_t offset);

  void getZfromGpu() {this->cld_->z();};
  uint32_t* d_z(){ return this->cdl_->d_z();};
  
protected:

//  static const uint32_t MAX_UINT32 = 4294967295;
  jsc::GpuMatrix<uint32_t> d_iAction_;
  jsc::GpuMatrix<T> d_ages_;
  jsc::GpuMatrix<T> d_ws_;
  jsc::GpuMatrix<uint32_t> d_Ns_;
  jsc::GpuMatrix<T> d_p_;

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);

  void setupComputeLabelsGPU(uint32_t iAction);
  uint32_t computeLabelsGPU(uint32_t i0);

  virtual VectorXu initLabels();
};
// --------------------------- impl -------------------------------------------

template<class T, class DS>
DDPMeansCUDA<T,DS>::DDPMeansCUDA(const shared_ptr<jsc::ClDataGpu<T> >& cld,
      T lambda, T Q, T tau)
  : DDPMeans<T,DS>(cld,lambda,Q,tau), //d_x_(cld), d_z_(this->N_),
  d_iAction_(1), d_ages_(1), d_ws_(1), d_Ns_(1), d_p_(this->D_,1)
{}

template<class T, class DS>
DDPMeansCUDA<T,DS>::~DDPMeansCUDA()
{};

template<class T, class DS> void DDPMeansCUDA<T,DS>::nextTimeStepGpu(T*
    d_x, uint32_t N, uint32_t step, uint32_t offset) {
//  this->clsPrev_.clear();
  for (uint32_t k =0; k< this->K_; ++k)
  {
//    this->clsPrev_.push_back(shared_ptr<typename
//        DS::DependentCluster>(this->cls_[k]->clone())); 
    this->cls_[k]->nextTimeStep();
  }

  this->Kprev_ = this->K_;
  this->cld_->updateData(d_x,N,step,offset);
  this->N_ = this->cld_->N();

  if(this->K_ > 0)
  { // seemed to slow down the algorithms convergence
    VectorXu idActions = initLabels();
    for(uint32_t k=0; k<this->K_; ++k)
      if(idActions(k) != UNASSIGNED && !this->cls_[k]->isInstantiated())
      { // instantiated an old cluster
        cout<<"revieve cluster "<<k<<" from point "<<idActions(k)<<endl;
        cout<<(this->cld_->x()->col(idActions(k))).transpose()<<endl;
        this->cls_[k]->reInstantiate(this->cld_->x()->col(idActions(k)));
      }
  }

//  this->initRevive();
};

template<class T, class DS>
void DDPMeansCUDA<T,DS>::setupComputeLabelsGPU(uint32_t iAction)
{
  d_iAction_.set(iAction);
  d_Ns_.set(this->counts());
  d_ages_.set(this->ages());
  d_ws_.set(this->weights());

  Matrix<T,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->isInstantiated())
      ps.col(k) = this->cls_[k]->centroid();
    else if(!this->cls_[k]->isInstantiated() && !this->cls_[k]->isNew())
      ps.col(k) = this->cls_[k]->prevCentroid();
//      ps.col(k) = this->clsPrev_[k]->centroid();
  d_p_.set(ps);

//  cout<<"ddpvMFlabels_gpu K="<<this->K_<<endl;
//  cout<<ps<<endl;
//  d_p_.print();
//  d_ages_.print();
//  d_Ns_.print();
//  cout<<"d_Ns_ "<<d_Ns_.get().transpose()<<endl;
//  cout<<"d_ages_ "<<d_ages_.get().transpose()<<endl;
}

template<>
uint32_t DDPMeansCUDA<float,Euclidean<float> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  ddpLabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(), 
      d_Ns_.data(), d_ages_.data(), d_ws_.data(), this->cl0_.lambda(), 
      this->cl0_.Q(), this->cl0_.tau(), 0, this->K_, i0, this->N_-i0, 
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<>
uint32_t DDPMeansCUDA<double,Euclidean<double> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  ddpLabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(), 
      d_Ns_.data(), d_ages_.data(), d_ws_.data(), this->cl0_.lambda(), 
      this->cl0_.Q(), this->cl0_.tau(), 0, this->K_, i0, this->N_-i0, 
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<>
uint32_t DDPMeansCUDA<float,Spherical<float> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  ddpvMFlabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(),
      d_Ns_.data(), d_ages_.data(), d_ws_.data(), this->cl0_.lambda(),
      this->cl0_.beta(), this->cl0_.Q(), 0, this->K_, i0, this->N_-i0,
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<>
uint32_t DDPMeansCUDA<double,Spherical<double> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  ddpvMFlabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(),
      d_Ns_.data(), d_ages_.data(), d_ws_.data(), this->cl0_.lambda(),
      this->cl0_.beta(), this->cl0_.Q(), 0, this->K_, i0, this->N_-i0,
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

template<class T,class DS>
VectorXu DDPMeansCUDA<T,DS>::initLabels()
{
  cout<<"cuda init labels K="<<this->K_<<endl;
  VectorXu asgnIdces = VectorXu::Ones(this->K_)*UNASSIGNED;
//  return asgnIdces;
  // TODO: seems to slow down the init!
  jsc::GpuMatrix<uint32_t> d_asgnIdces(asgnIdces);

  d_ages_.set(this->ages());
  d_ws_.set(this->weights());

  // TODO not too too sure about this
  Matrix<T,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->isInstantiated())
      ps.col(k) = this->cls_[k]->centroid();
    else if(!this->cls_[k]->isInstantiated() && !this->cls_[k]->isNew())
      ps.col(k) = this->cls_[k]->prevCentroid();
  d_p_.set(ps);

  d_p_.print();
  d_ages_.print();
  d_ws_.print();
//
  ddpLabelsSpecial_gpu(this->cld_->d_x(),  d_p_.data(), d_ages_.data(),
      d_ws_.data(), this->cl0_.lambda(), this->cl0_.Q(),
      this->cl0_.tau(), this->K_, this->N_, d_asgnIdces.data());
  return d_asgnIdces.get();      
}

template<>
VectorXu DDPMeansCUDA<float,Spherical<float> >::initLabels()
{
  cout<<"cuda init labels K="<<this->K_<<endl;
  VectorXu asgnIdces = VectorXu::Ones(this->K_)*UNASSIGNED;
//  return asgnIdces;
  // TODO: seems to slow down the init!
  jsc::GpuMatrix<uint32_t> d_asgnIdces(asgnIdces);

  d_ages_.set(this->ages());
  d_ws_.set(this->weights());

  // TODO not too too sure about this
  Matrix<float,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->isInstantiated())
      ps.col(k) = this->cls_[k]->centroid();
    else if(!this->cls_[k]->isInstantiated() && !this->cls_[k]->isNew())
      ps.col(k) = this->cls_[k]->prevCentroid();
  d_p_.set(ps);

  d_p_.print();
  d_ages_.print();
  d_ws_.print();
//
  ddpvMFLabelsSpecial_gpu(this->cld_->d_x(),  d_p_.data(), d_ages_.data(),
      d_ws_.data(), this->cl0_.lambda(), this->cl0_.Q(),
      this->cl0_.beta(), this->K_, this->N_, d_asgnIdces.data());
  return d_asgnIdces.get();      
}

template<>
VectorXu DDPMeansCUDA<double,Spherical<double> >::initLabels()
{
  cout<<"cuda init labels K="<<this->K_<<endl;
  VectorXu asgnIdces = VectorXu::Ones(this->K_)*UNASSIGNED;
//  return asgnIdces;
  // TODO: seems to slow down the init!
  jsc::GpuMatrix<uint32_t> d_asgnIdces(asgnIdces);

  d_ages_.set(this->ages());
  d_ws_.set(this->weights());

  // TODO not too too sure about this
  Matrix<double,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->isInstantiated())
      ps.col(k) = this->cls_[k]->centroid();
    else if(!this->cls_[k]->isInstantiated() && !this->cls_[k]->isNew())
      ps.col(k) = this->cls_[k]->prevCentroid();
  d_p_.set(ps);

  d_p_.print();
  d_ages_.print();
  d_ws_.print();
//
  ddpvMFLabelsSpecial_gpu(this->cld_->d_x(),  d_p_.data(), d_ages_.data(),
      d_ws_.data(), this->cl0_.lambda(), this->cl0_.Q(),
      this->cl0_.beta(), this->K_, this->N_, d_asgnIdces.data());
  return d_asgnIdces.get();      
}

}
