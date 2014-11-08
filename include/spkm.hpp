#pragma once
#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "sphere.hpp"
//#include "karcherMean.hpp"
#include "kmeans.hpp"
#include "clusterer.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

template<class T>
class SphericalKMeans : public KMeans<T>
{
public:
  SphericalKMeans(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen);
  virtual ~SphericalKMeans();

  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual bool closer(T a, T b);
//  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);
  virtual Matrix<T,Dynamic,1> computeCenter(uint32_t k);
  virtual Matrix<T,Dynamic,Dynamic> computeSums();

  virtual T silhouette();
};

//template<class T>
//class SphericalKMeansKarcher : public SphericalKMeans<T>
//{
//public:
//  SphericalKMeansKarcher(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
//      uint32_t K, boost::mt19937* pRndGen);
//  ~SphericalKMeansKarcher();
//
////  void initialize(const Matrix<T,Dynamic,Dynamic>& x);
//
//  virtual Matrix<T,Dynamic,1> computeCenter(uint32_t k);
//};

// --------------------------------- impl -------------------------------------
template<class T>
SphericalKMeans<T>::SphericalKMeans(
    const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen)
  : KMeans<T>(spx,K, pRndGen)
{}

template<class T>
SphericalKMeans<T>::~SphericalKMeans()
{}

template<class T>
T SphericalKMeans<T>::dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
{
//  return acos(min(1.0,max(-1.0,(a.transpose()*b)(0)))); // angular similarity
  return a.transpose()*b; // cosine similarity 
};

template<class T>
T SphericalKMeans<T>::dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
{
  return acos(min(1.0,max(-1.0,(a.transpose()*b)(0)))); // angular similarity
//  return a.transpose()*b; // cosine similarity 
};

template<class T>
bool SphericalKMeans<T>::closer(T a, T b)
{
//  return a<b; // if dist a is greater than dist b a is closer than b (angular dist)
  return a>b; // if dist a is greater than dist b a is closer than b (cosine dist)
};

template<class T>
Matrix<T,Dynamic,1> SphericalKMeans<T>::computeCenter(uint32_t k)
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
  return mean_k/mean_k.norm();
}

template<class T>
Matrix<T,Dynamic,Dynamic> SphericalKMeans<T>::computeSums(void)
{
  Matrix<T,Dynamic,Dynamic> xSums = Matrix<T,Dynamic,Dynamic>::Zero(this->D_, this->K_);
#pragma omp parallel for
  for(uint32_t k=0; k<this->K_; ++k)
    for(uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        xSums.col(k) += this->spx_->col(i); 
      }
  return xSums;
}

template<class T>
T SphericalKMeans<T>::silhouette()
{ 
  if(this->K_<2) return -1.0;
  assert(this->Ns_.sum() == this->N_);
  Matrix<T,Dynamic,Dynamic> xSums = computeSums();
  Matrix<T,Dynamic,1> sil(this->N_);
//#pragma omp parallel for
  for(uint32_t i=0; i<this->N_; ++i)
  {
    Matrix<T,Dynamic,1> b = Matrix<T,Dynamic,1>::Zero(this->K_);
    for(uint32_t k=0; k<this->K_; ++k)
      if (k == this->z_(i))
        b(k) = 1. -(this->spx_->col(i).transpose()*(xSums.col(k) - this->spx_->col(i)))(0)/static_cast<T>(this->Ns_(k));
      else
        b(k) = 1. -(this->spx_->col(i).transpose()*xSums.col(k))(0)/static_cast<T>(this->Ns_(k));
    T a_i = b(this->z_(i)); // average dist to own cluster
    T b_i = this->z_(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<this->K_; ++k)
      if(k != this->z_(i) && b(k) == b(k) && b(k) < b_i && this->Ns_(k) > 0)
      {
        b_i = b(k);
      }
    if(a_i < b_i)
      sil(i) = 1.- a_i/b_i;
    else if(a_i > b_i)
      sil(i) = b_i/a_i - 1.;
    else
      sil(i) = 0.;
    if(sil(i) <-1 || sil(i) > 1)
      cout<<"sil. out of bounds "<<sil(i)<< " a="<<a_i<<" b="<<b_i<<endl;
  }
  return sil.sum()/static_cast<T>(this->N_);
};

//template<class T>
//uint32_t SphericalKMeans<T>::indOfClosestCluster(int32_t i, T& sim_closest)
//{
//  // use cosine similarity because it is faster since acos is not computed
//  sim_closest = this->ps_.col(0).transpose() * this->spx_->col(i);
//  uint32_t z_i = 0;
//  for(uint32_t k=1; k<this->K_; ++k)
//  {
//    T sim_k = this->ps_.col(k).transpose()* this->spx_->col(i);
//    if( sim_k > sim_closest) // because of cosine distance
//    {
//      sim_closest = sim_k;
//      z_i = k;
//    }
//  }
//  return z_i;
//};
  
//template<class T>
//virtual T SphericalKMeans<T>::silhouette()
//{ 
//  Matrix<T,Dynamic,Dynamic> xSum(D_,K_);
//#pragma omp parallel for
//  for(uint32_t k=0; k<K_; ++k)
//  for(uint32_t i=0; i<N_; ++i)
//    if(z_(i) == k)
//  {
//    xSum.col(k) = 
//  }
//}

//template<class T>
//void SphericalKMeans<T>::initialize(const Matrix<T,Dynamic,Dynamic>& x)
//{
//  
//}
//
//template<class T>
//SphericalKMeansKarcher<T>::SphericalKMeansKarcher(
//    const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
//    boost::mt19937* pRndGen)
//  : SphericalKMeans<T>(spx,K,pRndGen)
//{}
//
//template<class T>
//SphericalKMeansKarcher<T>::~SphericalKMeansKarcher()
//{}
//
//template<class T>
//Matrix<T,Dynamic,1> SphericalKMeansKarcher<T>::computeCenter(uint32_t k)
//{
//  Matrix<T,Dynamic,Dynamic> xPs(this->spx_->rows(),this->spx_->cols());
//  Matrix<T,Dynamic,1> mean_k = karcherMean<T>(this->ps_.col(k), *(this->spx_), 
//        xPs, this->z_, k, 100,1);
//  this->Ns_(k) = 0.0;
//  for(uint32_t i=0; i<this->N_; ++i)
//    if(this->z_(i) == k)
//    {
//      this->Ns_(k) ++;
//    }
//  return mean_k;
//}

//template<class T>
//void SphericalKMeansKarcher<T>::updateCenters()
//{
//  Matrix<T,Dynamic,Dynamic> xPs(this->spx_->rows(),this->spx_->cols());
//#pragma omp parallel for
//  for(uint32_t k=0; k<this->K_; ++k)
//  {
////    Matrix<T,Dynamic,1> w(this->N_);
////    for(uint32_t i=0; i<this->N_; ++i)
////      if(this->z_(i) == k) 
////        w(i) = 1.0;
////      else
////        w(i) = 0.0;
////    this->ps_.col(k) = karcherMeanWeighted<T>(this->ps_.col(k), *(this->spx_), w, 100);
//    this->ps_.col(k) = karcherMean<T>(this->ps_.col(k), *(this->spx_), 
//        xPs, this->z_, k, 100,1);
//  }
//}
