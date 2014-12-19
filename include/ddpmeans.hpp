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

#ifdef BOOST_OLD
#define mt19937 boost::mt19937
#else
using boost::mt19937;
#endif

#define UNASSIGNED 4294967295

template<class T, class DS>
class DDPMeans : public DPMeans<T,DS>
{
public:
  DDPMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx,
      T lambda, T Q, T tau, mt19937* pRndGen);
  virtual ~DDPMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);
  virtual uint32_t optimisticLabelsAssign(uint32_t i0);
  virtual void updateLabelsSerial();
  virtual void updateLabels();
  virtual void updateCenters();
  
  virtual void nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);
  virtual void updateState(); // after converging for a single time instant

  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);
  Matrix<T,Dynamic,Dynamic> prevCentroids(){ return psPrev_;};

  virtual bool converged(T eps=1e-6) 
  {
    return this->Ns_.size() > 0 && this->Ns_.size() == prevNs_.size() && (prevNs_.array() == this->Ns_.array()).all();
  };


protected:

  std::vector<T> ts_; // age of clusters - incremented each iteration
  std::vector<T> ws_; // weights of each cluster 
  T Q_; // Q parameter
  T tau_; // tau parameter 
  T Kprev_; // K before updateLabels()
  Matrix<T,Dynamic,Dynamic> psPrev_; // centroids from last set of datapoints
  VectorXu prevNs_;

  Matrix<T,Dynamic,Dynamic> xSums_;
  std::vector<uint32_t> globalInd_; // for each non-dead cluster the global id of it;
  uint32_t globalMaxInd_;

  virtual void removeCluster(uint32_t k);
};

// -------------------------------- impl ----------------------------------
template<class T, class DS>
DDPMeans<T,DS>::DDPMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
    T lambda, T Q, T tau, mt19937* pRndGen)
  : DPMeans<T,DS>(spx,0,lambda,pRndGen), Q_(Q), tau_(tau)
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

template<class T, class DS>
DDPMeans<T,DS>::~DDPMeans()
{}

template<class T, class DS>
uint32_t DDPMeans<T,DS>::indOfClosestCluster(int32_t i, T& sim_closest)
{
  int z_i = this->K_;
  sim_closest = this->lambda_;
  T sim_k = 0.;
//  cout<<"K="<<this->K_<<" Ns:"<<this->Ns_.transpose()<<endl;
//  cout<<"cluster dists "<<i<<": "<<this->lambda_;
  for (uint32_t k=0; k<this->K_; ++k)
  {
    if(this->Ns_(k) == 0) 
    {// cluster not instantiated yet in this timestep
      sim_k = DS::distToUninstantiated(this->spx_->col(i), 
          this->ps_.col(k), ts_[k], ws_[k], tau_, Q_);
    }else{ // cluster instantiated
      sim_k = DS::dist(this->ps_.col(k), this->spx_->col(i));
    }
//    T sim_k = DS::dist(this->ps_.col(k), this->spx_->col(i));
//    if(this->Ns_(k) == 0) // cluster not instantiated yet in this timestep
//    {
//      sim_k = sim_k/(tau_*ts_[k]+1.+ 1.0/ws_[k]) + Q_*ts_[k];
//    }
//    cout<<" "<<sim_k;
    if(DS::closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
//  }cout<<endl;
  return z_i;
}

//template<class T, class DS>
//void DDPMeans<T,DS>::updateLabels()
//{
//  this->prevCost_ = this->cost_;
//  this->cost_ = 0.; // TODO:  does this take into account that creating a cluster costs
//  for(uint32_t i=0; i<this->N_; ++i)
//  {
//    T sim = 0.;
//    uint32_t z_i = indOfClosestCluster(i,sim);
//    this->cost_ += sim;
//    if(z_i == this->K_) 
//    { // start a new cluster
//      Matrix<T,Dynamic,Dynamic> psNew(this->D_,this->K_+1);
//      psNew.leftCols(this->K_) = this->ps_;
//      psNew.col(this->K_) = this->spx_->col(i);
//      this->ps_ = psNew;
//      this->K_ ++;
//      this->Ns_.conservativeResize(this->K_); 
//      this->Ns_(z_i) = 1.;
//    } else {
//      if(this->Ns_[z_i] == 0)
//      { // instantiated an old cluster
//        this->ps_.col(z_i) = DS::reInstantiatedOldCluster(this->spx_->col(i),
//            this->ps_.col(z_i), ts_[z_i], ws_[z_i], tau_);
////        T gamma = 1.0/(1.0/ws_[z_i] + ts_[z_i]*tau_);
////        this->ps_.col(z_i)=(this->ps_.col(z_i)*gamma + this->spx_->col(i))/(gamma+1.);
//      }
//      this->Ns_(z_i) ++;
//    }
//    if(this->z_(i) != UNASSIGNED)
//    {
//      this->Ns_(this->z_(i)) --;
//    }
//    this->z_(i) = z_i;
//  }
//};

template<class T,class DS>
uint32_t DDPMeans<T,DS>::optimisticLabelsAssign(uint32_t i0)
{
  uint32_t idAction = UNASSIGNED;
#pragma omp parallel for 
  for(uint32_t i=i0; i<this->N_; ++i)
  {
    T sim = 0.;
    uint32_t z_i = indOfClosestCluster(i,sim);
    if(z_i == this->K_ || this->Ns_[z_i] == 0) 
    { // note this as starting position
#pragma omp critical
      {
        if(idAction > i) idAction = i;
      }
    }
    //    if(this->z_(i) != UNASSIGNED) this->Ns_(this->z_(i)) --;
    this->z_(i) = z_i;
  }
  return idAction;
};

template<class T, class DS>
void DDPMeans<T,DS>::updateLabels()
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
        cout<<"new cluster "<<(this->K_-1)<<endl;
      } 
      else if(this->Ns_[z_i] == 0)
      { // instantiated an old cluster
        this->ps_.col(z_i) = DS::reInstantiatedOldCluster(this->spx_->col(idAction), 1,
            this->ps_.col(z_i), ts_[z_i], ws_[z_i], tau_);
        this->Ns_(z_i) = 1.; // set Ns of revived cluster to 1 tosignal
        // computeLabelsGPU to use the cluster;
        cout<<"revieve cluster "<<z_i<<endl;
      }
      i0 = idAction;
    }
    cout<<" K="<<this->K_<<" Ns="<<this->Ns_.transpose()<<endl;
  }while(idAction != UNASSIGNED);
//  cout<<"ps = "<<this->ps_<<endl;

  // TODO: this cost only works for a single time slice
  T cost =  0.0;
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->Ns_(k) == 1.) cost += this->lambda_;

  //TODO get counts from GPU
  prevNs_ =  this->Ns_;
  this->Ns_.fill(0);
#pragma omp parallel for reduction(+:cost)
  for(uint32_t k=0; k<this->K_; ++k)
    for(uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        this->Ns_(k) ++; 
        T sim_closest = DS::dist(this->ps_.col(k), this->spx_->col(i));
        cost += sim_closest;
      }
  this->prevCost_ = this->cost_;
  this->cost_ = cost;
};

template<class T, class DS>
void DDPMeans<T,DS>::updateLabelsSerial()
{
  for(uint32_t i=0; i<this->N_; ++i)
  {
    T sim = 0.;
    uint32_t z_i = indOfClosestCluster(i,sim);
    if(z_i == this->K_) 
    { // start a new cluster
      this->ps_.conservativeResize(this->D_,this->K_+1);
      this->Ns_.conservativeResize(this->K_+1); 
      this->ps_.col(this->K_) = this->spx_->col(i);
      this->Ns_(z_i) = 1.;
      this->globalInd_.push_back(this->globalMaxInd_++);
      this->K_ ++;
//      cout<<" added new cluster center at "<<this->spx_->col(i).transpose()<<endl;
    } else {
      if(this->Ns_[z_i] == 0)
      { // instantiated an old cluster
        this->ps_.col(z_i) = DS::reInstantiatedOldCluster(this->spx_->col(i), 1,
            this->ps_.col(z_i), ts_[z_i], ws_[z_i], tau_);
      }
      this->Ns_(z_i) ++;
    }
    if(this->z_(i) != UNASSIGNED) this->Ns_(this->z_(i)) --;
    this->z_(i) = z_i;
  }

  prevNs_ =  this->Ns_;
  this->Ns_.fill(0);
#pragma omp parallel for
  for(uint32_t k=0; k<this->K_; ++k)
    for(uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        this->Ns_(k) ++; 
      }
  cout<<" Ns = "<<this->Ns_.transpose()<<endl;
};

template<class T, class DS>
void DDPMeans<T,DS>::updateCenters()
{

  xSums_ = Matrix<T,Dynamic,Dynamic>::Zero(this->D_, this->K_);
#pragma omp parallel for 
  for(uint32_t k=0; k<this->K_; ++k)
  {
     Matrix<T,Dynamic,1> mean_k = DS::computeCenter(*this->spx_,this->z_,k,
       &this->Ns_(k));
     //TODO reuse mean computation
     xSums_.col(k) = DS::computeSum(*this->spx_,this->z_,k, NULL);
//    Matrix<T,Dynamic,1> mean_k = this->computeCenter(k);
    if (this->Ns_(k) > 0) 
    { // have data to update kth cluster
      if(k < this->Kprev_){
        this->ps_.col(k) = DS::reInstantiatedOldCluster(xSums_.col(k), this->Ns_(k),
            this->ps_.col(k), ts_[k], ws_[k], tau_);
//        T gamma = 1.0/(1.0/ws_[k] + ts_[k]*tau_);
//        this->ps_.col(k) = (this->ps_.col(k)*gamma+mean_k*this->Ns_(k))/
//          (gamma+this->Ns_(k));
      }else{
        this->ps_.col(k)=mean_k;
      }
    }
  }
  cout<<"centers"<<endl<<this->ps_<<endl<<this->Ns_.transpose()<<endl;
};

template<class T, class DS>
void DDPMeans<T,DS>::nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
{
  psPrev_ = this->ps_;
  this->Kprev_ = this->K_;
  assert(this->D_ == spx->rows());
  if(this->spx_.get() != spx.get()) this->spx_ = spx; // update the data
  this->N_ = spx->cols();
  this->z_.resize(this->N_);
//  this->z_.fill(0);
  this->z_.fill(UNASSIGNED);
};

template<class T, class DS>
void DDPMeans<T,DS>::updateState()
{
  vector<bool> toRemove(this->K_,false);
  for(uint32_t k=0; k<this->K_; ++k)
  {
    if (k<ws_.size() && this->Ns_(k) > 0)
    { // instantiated cluster from previous time; 
      //TODO reuse mean computation
      Matrix<T,Dynamic,1> xSum = DS::computeCenter(*this->spx_,this->z_,k,
          NULL);
      ws_[k] = DS::updateWeight(xSum, this->Ns_(k), this->ps_.col(k), ts_[k], 
          ws_[k], tau_);
//      ws_[k] = 1./(1./ws_[k] + ts_[k]*tau_) + this->Ns_(k);
      ts_[k] = 0; // re-instantiated -> age is 0
    }else if(k >= ws_.size()){
      // new cluster
      ts_.push_back(0);
      ws_.push_back(this->Ns_(k));
    }
    ts_[k] ++; // increment all ages

    if(DS::clusterIsDead(this->ts_[k],this->lambda_,Q_)) toRemove[k] = true;
//    if(this->ts_[k]*Q_ > this->lambda_) toRemove[k] = true;
//    if(this->ts_[k]*Q_ < this->lambda_) toRemove[k] = true;

    cout<<"cluster "<<k
      <<"\tN="<<this->Ns_(k)
      <<"\tage="<<ts_[k]
      <<"\tweight="<<ws_[k]
      <<"\t dead? "<<DS::clusterIsDead(this->ts_[k],this->lambda_,Q_)<<endl;
    cout<<"  center: "<<this->ps_.col(k).transpose()<<endl;
  }

  vector<int32_t> labelMap(this->K_);
  for(int32_t k=0; k<this->K_; ++k)
    labelMap[k] = k;
  int32_t nRemoved = 0;
  for(int32_t k=this->K_; k>=0; --k)
    if(toRemove[k])
    {
      for(int32_t j=k; j<this->K_; ++j) -- labelMap[j];
      ++ nRemoved;
      removeCluster(k);
    }

  cout<<"labelMap: ";
  for(int32_t k=0; k<this->K_; ++k) cout<<labelMap[k]<<" ";
  cout<<endl;

  this->K_ -= nRemoved;

  // fix labels
#pragma parallel for
  for(uint32_t i=0; i<this->N_; ++i)
    this->z_[i] = labelMap[this->z_[i]];
};

template<class T,class DS>
void DDPMeans<T,DS>::removeCluster(uint32_t k)
{
  cout<<" removeCluster "<<k<<endl;
//  cout<<this->ws_.size()<<endl;
//  cout<<this->ts_.size()<<endl;

  globalInd_.erase(globalInd_.begin()+k);
//  for(uint32_t k=0; k<this->ws_.size(); ++k)
//    cout<<this->ws_[k]<<endl;
//  this->ws_;
  this->ws_.erase(this->ws_.begin()+k);
//  for(uint32_t k=0; k<this->ws_.size(); ++k)
//    cout<<this->ws_[k]<<endl;
//  this->ts_;
  this->ts_.erase(this->ts_.begin()+k);
  //this->Ns_;
//  cout<<"Ns before remove "<<this->Ns_.transpose()<<endl;
  this->Ns_.middleRows(k,this->Ns_.rows()-k-1) = this->Ns_.bottomRows(this->Ns_.rows()-k-1);
  this->Ns_.conservativeResize(this->Ns_.rows()-1);
//  cout<<"Ns after remove "<<this->Ns_.transpose()<<endl;
//  //this->ps_;
//  cout<<this->ps_<<endl;
  this->ps_.middleCols(k,this->ps_.cols()-k-1) = this->ps_.rightCols(this->ps_.cols()-k-1);
  this->ps_.conservativeResize(this->ps_.rows(),this->ps_.cols()-1);

  if(k < this->psPrev_.cols())
  {
//    cout<<this->psPrev_<<endl;
    this->psPrev_.middleCols(k,this->psPrev_.cols()-k-1) = this->psPrev_.rightCols(this->psPrev_.cols()-k-1);
    this->psPrev_.conservativeResize(this->psPrev_.rows(),this->psPrev_.cols()-1);
  }
  //this->xSums_;
//  cout<<this->xSums_<<endl;
  this->xSums_.middleCols(k,this->xSums_.cols()-k-1) = this->xSums_.rightCols(this->xSums_.cols()-k-1);
  this->xSums_.conservativeResize(this->xSums_.rows(),this->xSums_.cols()-1);
//  cout<<this->xSums_<<endl;
//  cout<<"removed "<<k<<endl;
}
