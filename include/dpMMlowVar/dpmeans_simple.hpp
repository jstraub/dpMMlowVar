/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <dpMMlowVar/kmeans.hpp>

using namespace Eigen;
using std::cout;
using std::endl;

namespace dplv {
/// This is a simple version of the DPMeans algorithm in dpmeans.hpp
/// without inheritance or the use of CLData structures which can make
/// it a bit hard to read the other algorithm.
///
/// This implementation is ment as a lightweight alternative for small
/// number of datapoints or if you just want to have a look at how the
/// algorithm works.
template<class T, class DS, int D>
class DPMeansSimple
{
public:
  DPMeansSimple(double lambda);
  virtual ~DPMeansSimple();

  /// Adds an observation (adds obs, computes label, and potentially
  /// adds new cluster depending on label assignment).
  virtual void addObservation(const Eigen::Matrix<T,D,1>& x);
  /// Updates all labels of all data currently stored with the object.
  virtual void updateLabels();
  /// Updates all centers based on the current data and label
  /// assignments.
  virtual void updateCenters();

  
protected:
  double lambda_;
  uint32_t K_;
  std::vector<Eigen::Matrix<T,D,1>> xs_;
  std::vector<Eigen::Matrix<T,D,1>> mus_;
  std::vector<uint32_t> zs_;

  /// Removes all empty clusters.
  void removeEmptyClustes(const std::vector<uint32_t>& Ns);
  /// Computes the index of the closest cluster (may be K_ in which
  /// case a new cluster has to be added).
  uint32_t indOfClosestCluster(int32_t i, T& sim_closest);
};

// -------------------------------- impl ----------------------------------
template<class T, class DS, int D>
DPMeansSimple<T,DS,D>::DPMeansSimple(double lambda)
  : lambda_(lambda), K_(0)
{}
template<class T, class DS, int D>
DPMeansSimple<T,DS,D>::~DPMeansSimple()
{}

template<class T, class DS, int D>
void DPMeansSimple<T,DS,D>::addObservation(const Eigen::Matrix<T,D,1>& x) {
  xs_.push_back(x); 
  T sim_closest = 0;
  uint32_t z = indOfClosestCluster(x, sim_closest);
  if (z == K_) {
    mus_.push_back(x);
    ++K_;
  }
  zs_.push_back(z);
}

template<class T, class DS, int D>
uint32_t DPMeans<T,DS,D>::indOfClosestCluster(const
    Eigen::Matrix<T,D,1>& x, T& sim_closest)
{
  uint32_t z_i = K_;
  sim_closest = lambda_;
  for (uint32_t k=0; k<K_; ++k)
  {
    T sim_k = DS::dist(mus_.col(k), x);
    if(DS::closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
}

template<class T, class DS, int D>
void DPMeans<T,DS,D>::updateLabels()
{
  for(uint32_t i=0; i<xs_.size(); ++i) {
    T sim_closest = 0;
    uint32_t z = indOfClosestCluster(xs_[i], sim_closest);
    if (z == K_) {
      mus_.push_back(x);
      ++K_;
    }
    zs_[i] = z;
  }
}

// General update centers assumes Euclidean
template<class T, class DS, int D>
void DPMeans<T,DS,D>::updateCenters()
{
  std::vector<uint32_t> Ns(K_, 0);
  for(uint32_t k=0; k<K_; ++k)
    mus_[k].fill(0);
  for(uint32_t i=0; i<xs_.size(); ++i) {
    ++Ns[zs_[i]]; 
    mus_[zs_[i]] += xs_[i];
  }
  // Euclidean mean computation
  std::vector<bool> toDelete(K_,false);
  for(uint32_t k=0; k<K_; ++k) {
    mus_[k] /= Ns[k];
  }
  removeEmptyClustes(Ns);
}

// Template specialization to Euclidean data
template<class T, int D>
void DPMeans<T,Euclidean<T>,D>::updateCenters()
{
  std::vector<uint32_t> Ns(K_, 0);
  for(uint32_t k=0; k<K_; ++k)
    mus_[k].fill(0);
  for(uint32_t i=0; i<xs_.size(); ++i) {
    ++Ns[zs_[i]]; 
    mus_[zs_[i]] += xs_[i];
  }
  // Euclidean mean computation
  std::vector<bool> toDelete(K_,false);
  for(uint32_t k=0; k<K_; ++k) {
    mus_[k] /= Ns[k];
  }
  removeEmptyClustes(Ns);
}

// template specialization to spherical data
template<class T, int D>
void DPMeans<T,Spherical<T>,D>::updateCenters()
{
  std::vector<uint32_t> Ns(K_, 0);
  for(uint32_t k=0; k<K_; ++k)
    mus_[k].fill(0);
  for(uint32_t i=0; i<xs_.size(); ++i) {
    ++Ns[zs_[i]]; 
    mus_[zs_[i]] += xs_[i];
  }
  // Spherical mean computation
  std::vector<bool> toDelete(K_,false);
  for(uint32_t k=0; k<K_; ++k) {
    mus_[k] /= mus_[k].norm();
  }
  removeEmptyClustes(Ns);
}

template<class T, class DS, int D>
void DPMeans<T,DS,D>::removeEmptyClustes(const std::vector<uint32_t>& Ns) {

  uint32_t kNew = K_;
  for(int32_t k=K_-1; k>-1; --k)
    if(Ns[k] == 0) {
      cout<<"cluster k "<<k<<" empty"<<endl;
#pragma omp parallel for 
      for(uint32_t i=0; i<xs_.size(); ++i)
        if(static_cast<int32_t>(zs_[i]) >= k) sz_[i] -= 1;
      kNew --;
    }
  for(int32_t k=K_; k>=0; --k)
    if(Ns[k] == 0) mus_.erase(mus_.begin()+k);
  K_ = kNew;
};
