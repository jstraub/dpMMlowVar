#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include "sphere.hpp"
#include "clusterer.hpp"
#include "dir.hpp"
#include "cat.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T, class DS>
class KMeans : public Clusterer<T,DS>
{
public:
  KMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K);
  KMeans(const shared_ptr<ClData<T> >& cld);
  virtual ~KMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual void updateLabels();
  virtual void updateCenters();
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& deviates);
  virtual T avgIntraClusterDeviation();

//  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
//  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
//  virtual bool closer(T a, T b);

  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);
//  virtual Matrix<T,Dynamic,1> computeCenter(uint32_t k);

  virtual bool converged() {return false;};

protected:
  Sphere<T> S_;  // TODO this should not be here - needed for the empty cluster

//  DS space_; // dataspace class has all methods to compute distances and so on.

};

// --------------------------- impl -------------------------------------------

template<class T, class DS>
KMeans<T,DS>::KMeans(
    const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K)
  : Clusterer<T,DS>(spx,K), S_(this->D_) 
{
//  if(K>0){
//    Matrix<T,Dynamic,1> alpha(this->K_);
//    alpha.setOnes(this->K_);
//    Dir<Cat<T>,T> dir(alpha,this->pRndGen_);
//    Cat<T> pi = dir.sample(); 
//    cout<<"init pi="<<pi.pdf().transpose()<<endl;
//    pi.sample(this->z_);
//  }else{
//    this->z_.fill(0);
//  }

//  this->ps_.setZero(); 
//  updateCenters();
//  for(uint32_t k=0; k<this->K_; ++k)
//    this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
//  cout<<"init centers"<<endl<<this->ps_<<endl;
}

template<class T, class DS>
KMeans<T,DS>::KMeans( const shared_ptr<ClData<T> >& cld)
  : Clusterer<T,DS>(cld), S_(this->D_) 
{}

template<class T, class DS>
KMeans<T,DS>::~KMeans()
{}

//
//template<class T, class DS>
//T KMeans<T,DS>::dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
//{
//  return (a-b).squaredNorm();
//};
//
//template<class T, class DS>
//T KMeans<T,DS>::dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
//{
//  return (a-b).squaredNorm();
//};
//
//template<class T, class DS>
//bool KMeans<T,DS>::closer(T a, T b)
//{
//  return a<b; // if dist a is smaller than dist b a is closer than b (Eucledian)
//};


template<class T, class DS>
uint32_t KMeans<T,DS>::indOfClosestCluster(int32_t i, T& sim_closest)
{

  sim_closest = this->cls_[0]->dist(this->cld_->x()->col(i));
//    DS::dist(this->ps_.col(0), this->cld_->x()->col(i));
  uint32_t z_i = 0;
  for(uint32_t k=1; k<this->K_; ++k)
  {
    T sim_k = this->cls_[k]->dist(this->cld_->x()->col(i));
//DS::dist(this->ps_.col(k), this->cld_->x()->col(i));
    if( DS::closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
};

template<class T, class DS>
void KMeans<T,DS>::updateLabels()
{
  T cost = 0.;
#pragma omp parallel for reduction(+:cost)
  for(uint32_t i=0; i<this->N_; ++i)
  {
    T sim = 0;
    this->cld_->z(i) = indOfClosestCluster(i,sim);
    cost += sim;
  }
  this->prevCost_ = this->cost_;
  this->cost_ = cost;
}

//template<class T, class DS>
//Matrix<T,Dynamic,1> KMeans<T,DS>::computeCenter(uint32_t k)
//{
//  this->Ns_(k) = 0.0;
//  Matrix<T,Dynamic,1> mean_k(this->D_);
//  mean_k.setZero(this->D_);
//  for(uint32_t i=0; i<this->N_; ++i)
//    if(this->cld_->z(i) == k)
//    {
//      mean_k += this->cld_->x()->col(i); 
//      this->Ns_(k) ++;
//    }
//  return mean_k/this->Ns_(k);
//}

template<class T, class DS>
void KMeans<T,DS>::updateCenters()
{
  this->cld_->updateLabels(this->K_);
  this->cld_->computeSS();
  for(uint32_t k=0; k<this->K_; ++k)
    this->cls_[k]->updateCenter(this->cld_,k);

//#pragma omp parallel for
//  for(uint32_t k=0; k<this->K_; ++k)
//    this->cls_[k]->computeCenter(*(this->spx_),this->z_,k);

//  this->ps_ = DS::computeCenters(*(this->spx_),this->z_,this->K_,this->Ns_);
//#pragma omp parallel for 
//  for(uint32_t k=0; k<this->K_; ++k)
//  {
//    this->ps_.col(k) = computeCenter(k);
//    if (this->Ns_(k) <= 0) 
//      this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
//  }
}

template<class T, class DS>
MatrixXu KMeans<T,DS>::mostLikelyInds(uint32_t n, 
    Matrix<T,Dynamic,Dynamic>& deviates)
{
  MatrixXu inds = MatrixXu::Zero(n,this->K_);
  deviates = Matrix<T,Dynamic,Dynamic>::Ones(n,this->K_);
  deviates *= 99999.0;
  
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->cld_->z(i) == k)
      {
        T deviate = this->cls_[k]->dist(this->cld_->x()->col(i)); 
//          DS::dist(this->ps_.col(k), this->cld_->x()->col(i));
//        T deviate = (this->ps_.col(k) - this->cld_->x()->col(i)).norm();
        for (uint32_t j=0; j<n; ++j)
          if(DS::closer(deviate, deviates(j,k)))
          {
            for(uint32_t l=n-1; l>j; --l)
            {
              deviates(l,k) = deviates(l-1,k);
              inds(l,k) = inds(l-1,k);
            }
            deviates(j,k) = deviate;
            inds(j,k) = i;
//            cout<<"after update "<<logLike<<endl;
//            Matrix<T,Dynamic,Dynamic> out(n,this->K_*2);
//            out<<logLikes.cast<T>(),inds.cast<T>();
//            cout<<out<<endl;
            break;
          }
      }
  } 
  cout<<"::mostLikelyInds: deviates"<<endl;
  cout<<deviates<<endl;
  cout<<"::mostLikelyInds: inds"<<endl;
  cout<<inds<<endl;
  return inds;
};

template<class T, class DS>
T KMeans<T,DS>::avgIntraClusterDeviation()
{
  Matrix<T,Dynamic,1> deviates(this->K_);
  deviates.setZero(this->K_);
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    this->cls_[k]->N() = 0.0;
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->cld_->z(i) == k)
      {
        deviates(k) += this->cls_[k]->dist( this->cld_->x()->col(i));
//          DS::dist(this->ps_.col(k), this->cld_->x()->col(i));
//        deviates(k) += (this->ps_.col(k) - this->cld_->x()->col(i)).norm();
        this->cls_[k]->N() ++;
      }
//    if(this->Ns_(k) > 0.0) deviates(k) /= this->Ns_(k);
  }
  return deviates.sum()/ this->N_;//static_cast<T>(this->K_);
}
