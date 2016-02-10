/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <jsCore/global.hpp>
#include <jsCore/clDataGpu.hpp>
#include <jsCore/gpuMatrix.hpp>

#include <dpMMlowVar/dpmeans.hpp>
#include <dpMMlowVar/euclideanData.hpp>
#include <dpMMlowVar/sphericalData.hpp>

using namespace Eigen;
using std::cout;
using std::endl;

extern void dpLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z,
    double lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
    uint32_t *d_iAction);
extern void dpLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z,
    float lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
    uint32_t *d_iAction);

extern void dpvMFlabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z,
     double lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
     uint32_t *d_iAction);
extern void dpvMFlabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z,
     float lambda, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N,
     uint32_t *d_iAction);

namespace dplv {

template<class T, class DS>
class DPMeansCUDA : public DPMeans<T,DS>
{
public:
  DPMeansCUDA(const shared_ptr<jsc::ClDataGpu<T> >& cld, double lambda);
  virtual ~DPMeansCUDA();

  // TODO: need an add data functionality
//  virtual void nextTimeStepGpu(T* d_x, uint32_t N, uint32_t step, 
//      uint32_t offset, bool reviveOnInit = true);
  virtual void updateLabels();
  virtual void updateCenters();

  void getZfromGpu() {this->cld_->z();};
  uint32_t* d_z(){ return this->cdl_->d_z();};
  
protected:
  jsc::GpuMatrix<uint32_t> d_iAction_;
  jsc::GpuMatrix<T> d_p_;

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);

  void setupComputeLabelsGPU(uint32_t iAction);
  uint32_t computeLabelsGPU(uint32_t i0);

//  virtual VectorXu initLabels();
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


template<class T, class DS>
void DPMeansCUDA<T,DS>::updateLabels()
{
  uint32_t i0 = 0;
  uint32_t idAction = UNASSIGNED;
  for (int count = 0; count < this->N_; count++){
    // FIXME: in some edge cases optimisticLabelsAssign can output the same idAction repeatedly when it's supposed to be UNASSIGNED
    idAction = optimisticLabelsAssign(i0);
//    cout<<"[ddpmeans] iter:" << count << " K=" << this->K_ << " i0=" << i0 << " idAction=" << idAction << " N=" << this->N_ << endl;
    if(idAction != UNASSIGNED) {
      T sim = 0.;
      uint32_t z_i = this->indOfClosestCluster(idAction,sim);
//      std::cout << "z_i=" << z_i << std::endl;
      if(z_i == this->K_) {
        this->cls_.push_back(shared_ptr<typename DS::DependentCluster>(new
              typename DS::DependentCluster(this->cld_->x()->col(idAction))));
        this->K_ ++;
//        std::cout << "# " << this->cls_.size() << " K=" << this->K_ << std::endl;
//        std::cout << this->cld_->x()->col(idAction) << std::endl;
      }
      i0 = idAction;
    } else{
      // cout<<"[ddpmeans] break." << endl;
      break;
    }
  }

  // if a cluster runs out of labels reset it to the previous mean!
  for(uint32_t k=0; k<this->K_; ++k)
    if(!this->cls_[k]->isInstantiated())
    {
      std::cout << "ERROR ran  out of data in a cluster" << std::endl;
//      this->cls_[k]->centroid() = this->clsPrev_[k]->centroid();
//      this->cls_[k]->centroid() = this->cls_[k]->prevCentroid();
    }
};


template<class T, class DS>
void DPMeansCUDA<T,DS>::updateCenters()
{
  this->prevNs_.resize(this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    this->prevNs_(k) = this->cls_[k]->N();

  this->cld_->updateK(this->K_);
  this->cld_->computeSS();

  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->N() == 0)
      this->cls_[k]->resetCenter(this->cld_);
    else
      this->cls_[k]->updateCenter(this->cld_,k);

}
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
//    if(this->cls_[k]->isInstantiated())
//      std::cout << "ERROR: found empty cluster" << std::endl;
//    else
      ps.col(k) = this->cls_[k]->centroid();
  }
  d_p_.set(ps);

//  cout<<"ddpvMFlabels_gpu K="<<this->K_<<endl;
//  cout<<ps<<endl;
//  d_p_.print();
//  d_Ns_.print();
//  cout<<"d_Ns_ "<<d_Ns_.get().transpose()<<endl;
}

template<>
uint32_t DPMeansCUDA<float,Euclidean<float> >::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  this->setupComputeLabelsGPU(iAction);
//  cout << "******************BEFORE*******************"<<endl;
  dpLabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(), 
      this->lambda_, 
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
      this->lambda_, 
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
      this->lambda_,
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
      this->lambda_,
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
