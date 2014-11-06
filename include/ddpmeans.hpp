#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "sphere.hpp"
#include "dpmeans.hpp"
#include "dir.hpp"
#include "cat.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T>
class DDPMeans : public DPMeans<T>
{
public:
  DDPMeans(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx,
      T lambda, T Q, T tau, boost::mt19937* pRndGen);
  virtual ~DDPMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);
  virtual void updateLabels();
  virtual void updateCenters();
  
  virtual void nextTimeStep(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);
  virtual void updateState(); // after converging for a single time instant

  virtual uint32_t indOfClosestCluster(int32_t i);

  const static uint32_t UNASSIGNED = 4294967295;
protected:

  std::vector<T> ts_; // age of clusters - incremented each iteration
  std::vector<T> ws_; // weights of each cluster 
  T Q_; // Q parameter
  T tau_; // tau parameter 
  T Kprev_; // K before updateLabels()
  Matrix<T,Dynamic,Dynamic> psPrev_; // centroids from last set of datapoints
 
};

// -------------------------------- impl ----------------------------------
template<class T>
DDPMeans<T>::DDPMeans(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
    T lambda, T Q, T tau, boost::mt19937* pRndGen)
  : DPMeans<T>(spx,0,lambda,pRndGen), Q_(Q), tau_(tau)
{
  // compute initial counts for weight initialization
//#pragma omp parallel for
//  for (uint32_t k=0; k<this->K_; ++k)
//    for(uint32_t i=0; i<this->N_; ++i)
//      if(this->z_(i) == k)
//        this->Ns_(k) ++;
//  for (uint32_t k=0; k<this->K_; ++k)
//  {
//    ws_.push_back(this->Ns_(k));
//    ts_.push_back(1);
//  }
  this->Kprev_ = 0; // so that centers are initialized directly from sample mean
  psPrev_ = this->ps_;
}

template<class T>
DDPMeans<T>::~DDPMeans()
{}

template<class T>
uint32_t DDPMeans<T>::indOfClosestCluster(int32_t i)
{
  int z_i = this->K_;
  T sim_closest = this->lambda_;
//  cout<<"K="<<this->K_<<" Ns:"<<this->Ns_.transpose()<<endl;
//  cout<<"cluster dists "<<i<<": "<<this->lambda_;
  for (uint32_t k=0; k<this->K_; ++k)
  {
    T sim_k = dist(this->ps_.col(k), this->spx_->col(i));
    if(this->Ns_(k) == 0) // cluster not instantiated yet in this timestep
    {
      //TODO use gamma
//      T gamma = 1.0/(1.0/ws_[z_i] + ts_[z_i]*tau_);
      sim_k = sim_k/(tau_*ts_[k]+1.) + Q_*ts_[k];
//      sim_k = sim_k/(tau_*ts_[k]+1.) + Q_*ts_[k];
    }
//    cout<<" "<<sim_k;
    if(closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
//  }cout<<endl;
  return z_i;
}

template<class T>
void DDPMeans<T>::updateLabels()
{
  // reset cluster counts -> all uninstantiated
//  for (uint32_t k=0; k<this->K_; ++k)
//    this->Ns_(k) = 0; 
//#pragma omp parallel for 
// TODO not sure how to parallelize
  for(uint32_t i=0; i<this->N_; ++i)
  {
    uint32_t z_i = indOfClosestCluster(i);
    if(z_i == this->K_) 
    { // start a new cluster
      Matrix<T,Dynamic,Dynamic> psNew(this->D_,this->K_+1);
      psNew.leftCols(this->K_) = this->ps_;
      psNew.col(this->K_) = this->spx_->col(i);
      this->ps_ = psNew;
      this->K_ ++;
      this->Ns_.conservativeResize(this->K_); 
      this->Ns_(z_i) = 1.;
    } else {
      if(this->Ns_[z_i] == 0)
      { // instantiated an old cluster
        T gamma = 1.0/(1.0/ws_[z_i] + ts_[z_i]*tau_);
        this->ps_.col(z_i)=(this->ps_.col(z_i)*gamma + this->spx_->col(i))/(gamma+1.);
      }
      this->Ns_(z_i) ++;
    }
    if(this->z_(i) != UNASSIGNED)
    {
      this->Ns_(this->z_(i)) --;
    }
    this->z_(i) = z_i;
  }
};

template<class T>
void DDPMeans<T>::updateCenters()
{
#pragma omp parallel for 
  for(uint32_t k=0; k<this->K_; ++k)
  {
    Matrix<T,Dynamic,1> mean_k = this->computeCenter(k);
    if (this->Ns_(k) > 0) 
    { // have data to update kth cluster
      if(k < this->Kprev_){
        T gamma = 1.0/(1.0/ws_[k] + ts_[k]*tau_);
        this->ps_.col(k) = (this->ps_.col(k)*gamma+mean_k*this->Ns_(k))/
          (gamma+this->Ns_(k));
      }else{
        this->ps_.col(k)=mean_k;
      }
    }
  }
};

template<class T>
void DDPMeans<T>::nextTimeStep(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
{
  assert(this->D_ == spx->rows());
  this->spx_ = spx; // update the data
  this->N_ = spx->cols();
  this->z_.resize(this->N_);
  this->z_.fill(UNASSIGNED);
};

template<class T>
void DDPMeans<T>::updateState()
{
  for(uint32_t k=0; k<this->K_; ++k)
  {
    if (k<ws_.size() && this->Ns_(k) > 0)
    { // instantiated cluster from previous time; 
      ws_[k] = 1./(1./ws_[k] + ts_[k]*tau_) + this->Ns_(k);
      ts_[k] = 0; // re-instantiated -> age is 0
    }else if(k >= ws_.size()){
      // new cluster
      ts_.push_back(0);
      ws_.push_back(this->Ns_(k));
    }
    ts_[k] ++; // increment all ages
    cout<<"cluster "<<k
      <<"\tN="<<this->Ns_(k)
      <<"\tage="<<ts_[k]
      <<"\tweight="<<ws_[k]<<endl;
    cout<<"  center: "<<this->ps_.col(k).transpose()<<endl;
  }
  psPrev_ = this->ps_;
  this->Kprev_ = this->K_;
};
