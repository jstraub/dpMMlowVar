/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <dpMMlowVar/sphericalData.hpp>
#include <dpMMlowVar/euclideanData.hpp>

using namespace Eigen;
using std::cout;
using std::endl;

namespace dplv {
/// This is a simple version of the DPMeansSimple algorithm in dpmeans.hpp
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
  /// Constructor
  /// 
  /// lambda = cos(lambda_in_degree * M_PI/180.) - 1.
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

  uint32_t GetK() const {return K_;};
  const std::vector<uint32_t>& GetNs() const {return Ns_;};
  bool GetCenter(uint32_t k, Eigen::Vector3d& mu) const {
    if (k<K_) {mu = mus_[k]; return true; } else { return false; } };
  
protected:
  double lambda_;
  uint32_t K_;
  std::vector<Eigen::Matrix<T,D,1>> xs_;
  std::vector<Eigen::Matrix<T,D,1>> mus_;
  std::vector<uint32_t> zs_;
  std::vector<uint32_t> Ns_;

  /// resets all clusters (mus_ and Ks_) and resizes them to K_
  void resetClusters();
  /// Removes all empty clusters.
  void removeEmptyClusters(const std::vector<uint32_t>& Ns);
  /// Computes the index of the closest cluster (may be K_ in which
  /// case a new cluster has to be added).
  uint32_t indOfClosestCluster(const Eigen::Matrix<T,D,1>& x, T& sim_closest);
};

typedef DPMeansSimple<double,Euclidean<double>,3> DPMeansSimpleE3d;
typedef DPMeansSimple<double,Spherical<double>,3> DPMeansSimpleS3d; 

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
};

template<class T, class DS, int D>
uint32_t DPMeansSimple<T,DS,D>::indOfClosestCluster(const
    Eigen::Matrix<T,D,1>& x, T& sim_closest)
{
  uint32_t z_i = K_;
  sim_closest = lambda_;
  for (uint32_t k=0; k<K_; ++k)
  {
    T sim_k = DS::dist(mus_[k], x);
    if(DS::closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
};

  template<class T, class DS, int D>
void DPMeansSimple<T,DS,D>::updateLabels()
{
  for(uint32_t i=0; i<xs_.size(); ++i) {
    T sim_closest = 0;
    uint32_t z = indOfClosestCluster(xs_[i], sim_closest);
    if (z == K_) {
      mus_.push_back(xs_[i]);
      ++K_;
    }
    zs_[i] = z;
  }
};

// General update centers assumes Euclidean
  template<class T, class DS, int D>
void DPMeansSimple<T,DS,D>::updateCenters()
{
  resetClusters();
  for(uint32_t i=0; i<xs_.size(); ++i) {
    ++Ns_[zs_[i]]; 
//    mus_[zs_[i]] += xs_[i];
  }
  // Euclidean mean computation
//  for(uint32_t k=0; k<K_; ++k) {
//    mus_[k] /= Ns_[k];
//  }
  DS::computeCenters(xs_, zs_, K_, mus_);
  removeEmptyClusters(Ns_);
};

//// Template specialization to Euclidean data
//template<int D>
//void DPMeansSimple<double,Euclidean<double>,D>::updateCenters()
//{
//  resetClusters();
//  for(uint32_t i=0; i<xs_.size(); ++i) {
//    ++Ns_[zs_[i]]; 
////    mus_[zs_[i]] += xs_[i];
//  }
//  // Euclidean mean computation
////  for(uint32_t k=0; k<K_; ++k) {
////    mus_[k] /= Ns_[k];
////  }
//  DS::computeCenters<D>(xs_, zs_, mus_);
//  removeEmptyClusters(Ns_);
//}
//
//// template specialization to spherical data
//template<int D>
//void DPMeansSimple<double,Spherical<double>,D>::updateCenters()
//{
//  resetClusters();
//  for(uint32_t i=0; i<xs_.size(); ++i) {
//    ++Ns_[zs_[i]]; 
//    mus_[zs_[i]] += xs_[i];
//  }
//  // Spherical mean computation
//  for(uint32_t k=0; k<K_; ++k) {
//    mus_[k] /= mus_[k].norm();
//  }
//  removeEmptyClusters(Ns_);
//}

template<class T, class DS, int D>
void DPMeansSimple<T,DS,D>::resetClusters() {
  Ns_.resize(K_, 0);
  for(uint32_t k=0; k<K_; ++k) {
    mus_[k].fill(0);
    Ns_[k] = 0;
  }
}

template<class T, class DS, int D>
void DPMeansSimple<T,DS,D>::removeEmptyClusters(const std::vector<uint32_t>& Ns) {
  uint32_t kNew = K_;
  for(int32_t k=K_-1; k>-1; --k)
    if(Ns[k] == 0) {
      cout<<"cluster k "<<k<<" empty"<<endl;
#pragma omp parallel for 
      for(uint32_t i=0; i<xs_.size(); ++i)
        if(static_cast<int32_t>(zs_[i]) >= k) zs_[i] -= 1;
      kNew --;
    }
  for(int32_t k=K_; k>=0; --k)
    if(Ns[k] == 0) mus_.erase(mus_.begin()+k);
  K_ = kNew;
};

}
