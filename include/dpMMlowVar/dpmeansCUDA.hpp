/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <dpMMlowVar/kmeans.hpp>

using namespace Eigen;
using std::cout;
using std::endl;

extern void dpLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z,
    double lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
    uint32_t *d_iAction)
extern void dpLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z,
    float lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
    uint32_t *d_iAction)

extern void dpvMFlabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z,
     double lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
     uint32_t *d_iAction)
extern void dpvMFlabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z,
     float lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
     uint32_t *d_iAction)

namespace dplv {

template<class T, class DS>
class DPMeansCUDA : public DPeans<T,DS>
{
public:
  DPMeansCUDA(const shared_ptr<jsc::ClDataGpu<T> >& cld, double lambda);
  virtual ~DPMeansCUDA();

  // TODO: need an add data functionality
//  virtual void nextTimeStepGpu(T* d_x, uint32_t N, uint32_t step, 
//      uint32_t offset, bool reviveOnInit = true);

  void getZfromGpu() {this->cld_->z();};
  uint32_t* d_z(){ return this->cdl_->d_z();};
  
protected:
  jsc::GpuMatrix<uint32_t> d_iAction_;
  jsc::GpuMatrix<T> d_p_;

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);

  void setupComputeLabelsGPU(uint32_t iAction);
  uint32_t computeLabelsGPU(uint32_t i0);

  virtual VectorXu initLabels();
};

// -------------------------------- impl ----------------------------------
template<class T, class DS>
DPMeansCUDA<T,DS>::DPMeansCUDA(const shared_ptr<jsc::ClDataGpu<T> >&
    cld, double lambda)
  : DPMeans<T,DS>(cld, lambda), d_iAction_(1), d_p_(this->D_,1)
{}

template<class T, class DS>
DPMeansCUDA<T,DS>::~DPMeansCUDA()
{}

//template<class T, class DS> void DPMeansCUDA<T,DS>::nextTimeStepGpu(T*
//    d_x, uint32_t N, uint32_t step, uint32_t offset, bool reviveOnInit) 
//{
//
//  this->Kprev_ = this->K_;
//  this->cld_->updateData(d_x,N,step,offset);
//  this->N_ = this->cld_->N();
//
//  if(reviveOnInit) 
//    this->initRevive(); 
//  else if(this->K_ == 0)
//    // revive or add cluster from the first data-point
//    this->createReviveFrom(0);
//};

template<class T, class DS>
void DPMeansCUDA<T,DS>::setupComputeLabelsGPU(uint32_t iAction)
{
  d_iAction_.set(iAction);

  Matrix<T,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k) {
    if(this->cls_[k]->isInstantiated())
      std::cout << "ERROR: found empty cluster" << std::endl;
    else
      ps.col(k) = this->cls_[k]->centroid();
  d_p_.set(ps);

//  cout<<"ddpvMFlabels_gpu K="<<this->K_<<endl;
//  cout<<ps<<endl;
//  d_p_.print();
//  d_Ns_.print();
//  cout<<"d_Ns_ "<<d_Ns_.get().transpose()<<endl;
}

template<>
uint32_t DDPMeansCUDA<float,Euclidean<float> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  dpLabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(), 
      this->cl0_.lambda(), 
      0, this->K_, i0, this->N_-i0, 
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<>
uint32_t DPMeansCUDA<double,Euclidean<double> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  dpLabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(), 
      this->cl0_.lambda(), 
       0, this->K_, i0, this->N_-i0, 
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<>
uint32_t DPMeansCUDA<float,Spherical<float> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  dpvMFlabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(),
      this->cl0_.lambda(),
       0, this->K_, i0, this->N_-i0,
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<>
uint32_t DPMeansCUDA<double,Spherical<double> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  dpvMFlabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(),
      this->cl0_.lambda(),
       0, this->K_, i0, this->N_-i0,
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<class T, class DS>
uint32_t DPMeansCUDA<T,DS>::optimisticLabelsAssign(uint32_t i0)
{
  return computeLabelsGPU(0); // TODO make passing i0 work!
};

}
