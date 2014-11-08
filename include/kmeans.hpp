#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "sphere.hpp"
#include "clusterer.hpp"
#include "dir.hpp"
#include "cat.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T>
class KMeans : public Clusterer<T>
{
public:
  KMeans(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen);
  virtual ~KMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual void updateLabels();
  virtual void updateCenters();
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& deviates);
  virtual T avgIntraClusterDeviation();

  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual bool closer(T a, T b);

  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);
  virtual Matrix<T,Dynamic,1> computeCenter(uint32_t k);

  virtual bool converged() {return false;};

protected:
  Sphere<T> S_;  // TODO this should not be here - needed for the empty cluster
};

// --------------------------- impl -------------------------------------------

template<class T>
KMeans<T>::KMeans(
    const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen)
  : Clusterer<T>(spx,K, pRndGen), S_(this->D_) 
{
  if(K>0){
    Matrix<T,Dynamic,1> alpha(this->K_);
    alpha.setOnes(this->K_);
    Dir<Cat<T>,T> dir(alpha,this->pRndGen_);
    Cat<T> pi = dir.sample(); 
    cout<<"init pi="<<pi.pdf().transpose()<<endl;
    pi.sample(this->z_);
  }else{
    this->z_.fill(0);
  }

  this->ps_.setZero(); 
//  updateCenters();
//  for(uint32_t k=0; k<this->K_; ++k)
//    this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
//  cout<<"init centers"<<endl<<this->ps_<<endl;
}

template<class T>
KMeans<T>::~KMeans()
{}


template<class T>
T KMeans<T>::dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
{
  return (a-b).norm();
};

template<class T>
T KMeans<T>::dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
{
  return (a-b).norm();
};

template<class T>
bool KMeans<T>::closer(T a, T b)
{
  return a<b; // if dist a is smaller than dist b a is closer than b (Eucledian)
};


template<class T>
uint32_t KMeans<T>::indOfClosestCluster(int32_t i, T& sim_closest)
{
  sim_closest = dist(this->ps_.col(0), this->spx_->col(i));
  uint32_t z_i = 0;
  for(uint32_t k=1; k<this->K_; ++k)
  {
    T sim_k = dist(this->ps_.col(k), this->spx_->col(i));
    if( closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
};

template<class T>
void KMeans<T>::updateLabels()
{
  T cost = 0.;
#pragma omp parallel for reduction(+:cost)
  for(uint32_t i=0; i<this->N_; ++i)
  {
    T sim = 0;
    this->z_(i) = indOfClosestCluster(i,sim);
    cost += sim;
  }
  this->prevCost_ = this->cost_;
  this->cost_ = cost;
}

template<class T>
Matrix<T,Dynamic,1> KMeans<T>::computeCenter(uint32_t k)
{
  this->Ns_(k) = 0.0;
  Matrix<T,Dynamic,1> mean_k(this->D_);
  mean_k.setZero(this->D_);
  for(uint32_t i=0; i<this->N_; ++i)
    if(this->z_(i) == k)
    {
      mean_k += this->spx_->col(i); 
      this->Ns_(k) ++;
    }
  return mean_k/this->Ns_(k);
}

template<class T>
void KMeans<T>::updateCenters()
{
#pragma omp parallel for 
  for(uint32_t k=0; k<this->K_; ++k)
  {
    this->ps_.col(k) = computeCenter(k);
    if (this->Ns_(k) <= 0) 
      this->ps_.col(k) = S_.sampleUnif(this->pRndGen_);
  }
}

template<class T>
MatrixXu KMeans<T>::mostLikelyInds(uint32_t n, 
    Matrix<T,Dynamic,Dynamic>& deviates)
{
  MatrixXu inds = MatrixXu::Zero(n,this->K_);
  deviates = Matrix<T,Dynamic,Dynamic>::Ones(n,this->K_);
  deviates *= 99999.0;
  
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        T deviate = dist(this->ps_.col(k), this->spx_->col(i));
//        T deviate = (this->ps_.col(k) - this->spx_->col(i)).norm();
        for (uint32_t j=0; j<n; ++j)
          if(closer(deviate, deviates(j,k)))
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

template<class T>
T KMeans<T>::avgIntraClusterDeviation()
{
  Matrix<T,Dynamic,1> deviates(this->K_);
  deviates.setZero(this->K_);
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    this->Ns_(k) = 0.0;
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        deviates(k) += dist(this->ps_.col(k), this->spx_->col(i));
//        deviates(k) += (this->ps_.col(k) - this->spx_->col(i)).norm();
        this->Ns_(k) ++;
      }
//    if(this->Ns_(k) > 0.0) deviates(k) /= this->Ns_(k);
  }
  return deviates.sum()/ this->Ns_.sum();//static_cast<T>(this->K_);
}
