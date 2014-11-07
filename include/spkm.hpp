#pragma once
#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include "sphere.hpp"
#include "karcherMean.hpp"
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

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

//  virtual void updateLabels();
//  virtual void updateCenters();
//  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& deviates);
//  virtual T avgIntraClusterDeviation();

  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual bool closer(T a, T b);
  virtual uint32_t indOfClosestCluster(int32_t i);
  virtual Matrix<T,Dynamic,1> computeCenter(uint32_t k);

};

template<class T>
class SphericalKMeansKarcher : public SphericalKMeans<T>
{
public:
  SphericalKMeansKarcher(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
      uint32_t K, boost::mt19937* pRndGen);
  ~SphericalKMeansKarcher();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual Matrix<T,Dynamic,1> computeCenter(uint32_t k);
};

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
  return acos(min(1.0,max(-1.0,(a.transpose()*b)(0)))); // angular similarity
//  return a.transpose()*b; // cosine similarity 
};

template<class T>
bool SphericalKMeans<T>::closer(T a, T b)
{
  return a<b; // if dist a is greater than dist b a is closer than b (angular dist)
//  return a>b; // if dist a is greater than dist b a is closer than b (cosine dist)
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
uint32_t SphericalKMeans<T>::indOfClosestCluster(int32_t i)
{
  // use cosine similarity because it is faster since acos is not computed
  T sim_closest = this->ps_.col(0).transpose() * this->spx_->col(i);
  uint32_t z_i = 0;
  for(uint32_t k=1; k<this->K_; ++k)
  {
    T sim_k = this->ps_.col(k).transpose()* this->spx_->col(i);
    if( sim_k > sim_closest) // because of cosine distance
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
};

//template<class T>
//void SphericalKMeans<T>::initialize(const Matrix<T,Dynamic,Dynamic>& x)
//{
//  
//}
//
template<class T>
SphericalKMeansKarcher<T>::SphericalKMeansKarcher(
    const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen)
  : SphericalKMeans<T>(spx,K,pRndGen)
{}

template<class T>
SphericalKMeansKarcher<T>::~SphericalKMeansKarcher()
{}

template<class T>
Matrix<T,Dynamic,1> SphericalKMeansKarcher<T>::computeCenter(uint32_t k)
{
  Matrix<T,Dynamic,Dynamic> xPs(this->spx_->rows(),this->spx_->cols());
  Matrix<T,Dynamic,1> mean_k = karcherMean<T>(this->ps_.col(k), *(this->spx_), 
        xPs, this->z_, k, 100,1);
  this->Ns_(k) = 0.0;
  for(uint32_t i=0; i<this->N_; ++i)
    if(this->z_(i) == k)
    {
      this->Ns_(k) ++;
    }
  return mean_k;
}

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
