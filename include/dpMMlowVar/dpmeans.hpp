#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include "kmeans.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T, class DS>
class DPMeans : public KMeans<T,DS>
{
public:
  DPMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K0,
    double lambda);
  DPMeans(const shared_ptr<ClData<T> >& cld, double lambda);
  virtual ~DPMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);
  virtual void updateLabels();
  virtual void updateCenters();

  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);
  
protected:
  double lambda_;
};

// -------------------------------- impl ----------------------------------
template<class T, class DS>
DPMeans<T,DS>::DPMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
    uint32_t K0, double lambda)
  : KMeans<T,DS>(spx,K0), lambda_(lambda)
{}

template<class T, class DS>
DPMeans<T,DS>::DPMeans(const shared_ptr<ClData<T> >& cld, double lambda)
  : KMeans<T,DS>(cld), lambda_(lambda)
{}

template<class T, class DS>
DPMeans<T,DS>::~DPMeans()
{}

template<class T, class DS>
uint32_t DPMeans<T,DS>::indOfClosestCluster(int32_t i, T& sim_closest)
{
  int z_i = this->K_;
  sim_closest = lambda_;
  for (uint32_t k=0; k<this->K_; ++k)
  {
    T sim_k = this->cls_[k]->dist(this->cld_->x()->col(i));
//      DS::dist(this->ps_.col(k), this->cld_->x()->col(i));
    if(DS::closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

template<class T, class DS>
void DPMeans<T,DS>::updateLabels()
{
//#pragma omp parallel for 
// TODO not sure how to parallelize
  this->prevCost_ = this->cost_;
  this->cost_ = 0.;
  for(uint32_t i=0; i<this->N_; ++i)
  {
    T sim = 0.;
    uint32_t z_i = indOfClosestCluster(i,sim);
    this->cost_ += sim;

    if(z_i == this->K_) 
    {
      this->cls_.push_back(shared_ptr<typename DS::DependentCluster>(new
            typename DS::DependentCluster(this->cld_->x()->col(i))));
      this->K_ ++;
    }
    this->cld_->z(i) = z_i;
  }
};

template<class T, class DS>
void DPMeans<T,DS>::updateCenters()
{
  KMeans<T,DS>::updateCenters();
//  this->ps_ = DS::computeCenters(*this->spx_,this->z_,this->K_,this->Ns_);

  vector<bool> toDelete(this->K_,false);
  for(uint32_t k=0; k<this->K_; ++k)
  {
    if (!this->cls_[k]->isInstantiated()) 
//    if (this->Ns_(k) <= 0) 
      toDelete[k] = true;
  }

  uint32_t kNew = this->K_;
  for(int32_t k=this->K_-1; k>-1; --k)
    if(toDelete[k])
    {
      cout<<"cluster k "<<k<<" empty"<<endl;
#pragma omp parallel for 
      for(uint32_t i=0; i<this->N_; ++i)
      {
        if(static_cast<int32_t>(this->cld_->z(i)) >= k) this->cld_->z(i) -= 1;
      }
      kNew --;
    }

//  Matrix<T,Dynamic,Dynamic> psNew(this->D_,kNew);
//  int32_t offset = 0;
  for(int32_t k=this->K_; k >=0 ; --k)
    if(toDelete[k])
    {
      this->cls_.erase(this->cls_.begin()+k);
    }
//  for(uint32_t k=0; k<this->K_; ++k)
//    if(toDelete[k])
//    {
//      offset ++;
//    }else{
//      psNew.col(k-offset) = this->ps_.col(k);
//    }
//  this->ps_ = psNew;
  this->K_ = kNew;
};
