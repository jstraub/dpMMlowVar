#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "dpmeans.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T, class DS>
class DDPMeans : public DPMeans<T,DS>
{
public:
  DDPMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx,
      T lambda, T Q, T tau);
  DDPMeans(const shared_ptr<ClData<T> >& cld,
      T lambda, T Q, T tau);
  virtual ~DDPMeans();

  virtual void updateLabelsSerial();
  virtual void updateLabels();
  virtual void updateCenters();
  
  virtual void nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);
//  virtual void nextTimeStep(const shared_ptr<ClData<T> >& cld);
  virtual void updateState(); // after converging for a single time instant

  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);

  virtual bool converged(T eps=1e-6) 
  {
    return this->counts().size() > 0 && this->counts().size() == prevNs_.size()
      && (prevNs_.array() == this->counts().array()).all();
  };

  VectorXf ages(){
    VectorXf ts(this->K_);
    for(uint32_t k=0; k<this->K_; ++k) ts(k) = this->cls_[k]->t();
    return ts;
  };

  VectorXf weights(){
    VectorXf ws(this->K_);
    for(uint32_t k=0; k<this->K_; ++k) ws(k) = this->cls_[k]->w();
    return ws;
  };

protected:

  T Kprev_; // K before updateLabels()
  VectorXu prevNs_;
  uint32_t globalMaxInd_;

  typename DS::DependentCluster cl0_;
  vector< shared_ptr<typename DS::DependentCluster> > clsPrev_; // prev clusters 

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);
  virtual VectorXu initLabels();
};

// -------------------------------- impl ----------------------------------
template<class T, class DS>
DDPMeans<T,DS>::DDPMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
    T lambda, T Q, T tau)
  : DPMeans<T,DS>(spx,0,lambda), cl0_(tau,lambda,Q)
{
  this->Kprev_ = 0; // so that centers are initialized directly from sample mean
};

template<class T, class DS>
DDPMeans<T,DS>::DDPMeans(const shared_ptr<ClData<T> >& cld, 
    T lambda, T Q, T tau)
  : DPMeans<T,DS>(cld,lambda), cl0_(tau,lambda,Q)
{
  this->Kprev_ = cld->K(); // so that centers are initialized directly from sample mean
};

template<class T, class DS>
DDPMeans<T,DS>::~DDPMeans()
{}

template<class T, class DS>
uint32_t DDPMeans<T,DS>::indOfClosestCluster(int32_t i, T& sim_closest)
{
  int z_i = this->K_;
  sim_closest = this->lambda_;
  T sim_k = 0.;
  for (uint32_t k=0; k<this->K_; ++k)
  {
    sim_k = this->cls_[k]->dist(this->cld_->x()->col(i)); 
    if(DS::closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

template<class T,class DS>
uint32_t DDPMeans<T,DS>::optimisticLabelsAssign(uint32_t i0)
{
  uint32_t idAction = UNASSIGNED;
#pragma omp parallel for 
  for(uint32_t i=i0; i<this->N_; ++i)
  {
    T sim = 0.;
    uint32_t z_i = indOfClosestCluster(i,sim);
    if(z_i == this->K_ || !this->cls_[z_i]->isInstantiated())
    { // note this as starting position
#pragma omp critical
      {
        if(idAction > i) idAction = i;
      }
    }
    this->cld_->z(i) = z_i;
  }
  return idAction;
};

template<class T,class DS>
VectorXu DDPMeans<T,DS>::initLabels()
{
  return VectorXu::Ones(this->K_)*UNASSIGNED;
}

template<class T, class DS>
void DDPMeans<T,DS>::updateLabels()
{
  uint32_t idAction = UNASSIGNED;
  uint32_t i0 = 0;

  do{
    idAction = optimisticLabelsAssign(i0);
    if(idAction != UNASSIGNED)
    {
//      cout<<"idAction "<<idAction<<endl;
//      cout<<" x @ idAction "<<this->cld_->x()->col(idAction).transpose()<<endl;
//      cout<<"K= "<<this->K_<<endl;
      T sim = 0.;
      uint32_t z_i = this->indOfClosestCluster(idAction,sim);
      if(z_i == this->K_) 
      { // start a new cluster
        this->cls_.push_back(shared_ptr<typename DS::DependentCluster>(new
              typename DS::DependentCluster(this->cld_->x()->col(idAction),cl0_)));
        this->cls_[z_i]->globalId = this->globalMaxInd_++;
        this->K_ ++;
        cout<<"new cluster "<<(this->K_-1)<<endl;
      } 
      else if(!this->cls_[z_i]->isInstantiated())
      { // instantiated an old cluster
        this->cls_[z_i]->reInstantiate(this->cld_->x()->col(idAction));
        cout<<"revieve cluster "<<z_i<<endl;
      }
      i0 = idAction;
    }
    cout<<" K="<<this->K_<<" Ns="<<this->counts().transpose()<<endl;
  }while(idAction != UNASSIGNED);
  // if a cluster runs out of labels reset it to the previous mean!
  for(uint32_t k=0; k<this->K_; ++k)
    if(!this->cls_[k]->isInstantiated())
    {
      this->cls_[k]->centroid() = this->clsPrev_[k]->centroid();
    }
};

template<class T, class DS>
void DDPMeans<T,DS>::updateLabelsSerial()
{
  //TODO this one may be broken by now
  for(uint32_t i=0; i<this->N_; ++i)
  {
    T sim = 0.;
    uint32_t z_i = indOfClosestCluster(i,sim);
    if(z_i == this->K_) 
    { // start a new cluster
      this->cls_.push_back(shared_ptr<typename DS::DependentCluster>(new
            typename DS::DependentCluster(this->cld_->x()->col(i),cl0_)));
      this->K_ ++;
    } else {
      if(!this->cls_[z_i]->isInstantiated())
      { // instantiated an old cluster
        this->cls_[z_i]->reInstantiate(this->cld_->x()->col(i));
      }
      ++ this->cls_[z_i]->N();
    }
    if(this->cld_->z(i) != UNASSIGNED) -- this->cls_[z_i]->N();
    this->cld_->z(i) = z_i;
  }
};

template<class T, class DS>
void DDPMeans<T,DS>::updateCenters()
{
  prevNs_.resize(this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    prevNs_(k) = this->cls_[k]->N();

  this->cld_->updateK(this->K_);
  this->cld_->computeSS();

//#pragma omp parallel for 
  for(uint32_t k=0; k<this->K_; ++k)
  {
    this->cls_[k]->updateSS(this->cld_,k);
    if(this->cls_[k]->isInstantiated()) 
    { // have data to update kth cluster
      if(k < this->Kprev_){
        this->cls_[k]->reInstantiate();
      }else{
        this->cls_[k]->updateCenter();
      }
    }
  }
};

template<class T, class DS>
void DDPMeans<T,DS>::nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
{
  this->clsPrev_.clear();
  for (uint32_t k =0; k< this->K_; ++k)
  {
    clsPrev_.push_back(shared_ptr<typename
        DS::DependentCluster>(this->cls_[k]->clone())); 
    this->cls_[k]->N() = 0;
  }

  this->Kprev_ = this->K_;
  this->cld_->updateData(spx);
  this->N_ = this->cld_->N();

  if(false && this->K_ > 0)
  { // seemed to slow down the algorithms confergence
    VectorXu idActions = initLabels();
    for(uint32_t k=0; k<this->K_; ++k)
      if(idActions(k) != UNASSIGNED && !this->cls_[k]->isInstantiated())
      { // instantiated an old cluster
        cout<<"revieve cluster "<<k<<" from point "<<idActions(k)<<endl;
        cout<<(this->cld_->x()->col(idActions(k))).transpose()<<endl;
        this->cls_[k]->reInstantiate(this->cld_->x()->col(idActions(k)));
      }
  }
};

template<class T, class DS>
void DDPMeans<T,DS>::updateState()
{
  vector<bool> toRemove(this->K_,false);
  for(uint32_t k=0; k<this->K_; ++k)
  {
    if(this->cls_[k]->isInstantiated()) this->cls_[k]->updateWeight();

    this->cls_[k]->incAge();

    if(this->cls_[k]->isDead()) toRemove[k] = true;

    this->cls_[k]->print();
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
      this->cls_.erase(this->cls_.begin()+k);
    }
  this->K_ -= nRemoved;

  if(nRemoved > 0)
  {
    cout<<"labelMap: ";
    for(int32_t k=0; k<this->K_+nRemoved; ++k) cout<<labelMap[k]<<" ";
    cout<<endl;
    this->cld_->labelMap(labelMap);
  }
};

