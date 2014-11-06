#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "kmeans.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T>
class DPMeans : public KMeans<T>
{
public:
  DPMeans(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K0,
    double lambda, boost::mt19937* pRndGen);
  virtual ~DPMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual void updateLabels();
  virtual void updateCenters();

  virtual uint32_t indOfClosestCluster(int32_t i);
  
protected:
  double lambda_;
};

// -------------------------------- impl ----------------------------------
template<class T>
DPMeans<T>::DPMeans(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
    uint32_t K0, double lambda, boost::mt19937* pRndGen)
  : KMeans<T>(spx,K0,pRndGen), lambda_(lambda)
{}

template<class T>
DPMeans<T>::~DPMeans()
{}

template<class T>
uint32_t DPMeans<T>::indOfClosestCluster(int32_t i)
{
  int z_i = this->K_;
  T sim_closest = lambda_;
  for (uint32_t k=0; k<this->K_; ++k)
  {
    T sim_k = dist(this->ps_.col(k), this->spx_->col(i));
    if(closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

template<class T>
void DPMeans<T>::updateLabels()
{
//#pragma omp parallel for 
// TODO not sure how to parallelize
  for(uint32_t i=0; i<this->N_; ++i)
  {
    uint32_t z_i = indOfClosestCluster(i);

    if(z_i == this->K_) 
    {
      this->ps_.conservativeResize(this->D_,this->K_+1);
      this->Ns_.conservativeResize(this->K_+1); 
      this->ps_.col(this->K_) = this->spx_->col(i);
      this->Ns_(this->K_) = 1.;
      this->K_ ++;
    }
    this->z_(i) = z_i;
  }
};

template<class T>
void DPMeans<T>::updateCenters()
{
  vector<bool> toDelete(this->K_,false);
#pragma omp parallel for 
  for(uint32_t k=0; k<this->K_; ++k)
  {
    this->ps_.col(k) = this->computeCenter(k);
    if (this->Ns_(k) <= 0) 
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
        if(static_cast<int32_t>(this->z_(i)) >= k) this->z_(i)--;
      }
      kNew --;
    }

  Matrix<T,Dynamic,Dynamic> psNew(this->D_,kNew);
  int32_t offset = 0;
  for(uint32_t k=0; k<this->K_; ++k)
    if(toDelete[k])
    {
      offset ++;
    }else{
      psNew.col(k-offset) = this->ps_.col(k);
    }
  this->ps_ = psNew;
  this->K_ = kNew;
};
