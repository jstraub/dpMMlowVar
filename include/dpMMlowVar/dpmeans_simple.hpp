/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
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
  DPMeansSimple(const DPMeansSimple<T,DS,D>& o);
  virtual ~DPMeansSimple();

  /// Adds an observation (adds obs, computes label, and potentially
  /// adds new cluster depending on label assignment).
  virtual void addObservation(const Eigen::Matrix<T,D,1>& x);
  /// Updates all labels of all data currently stored with the object.
  virtual void updateLabels();
  /// Updates all centers based on the current data and label
  /// assignments.
  virtual void updateCenters();

  /// Iterate updates for centers and labels until cost function
  /// convergence.
  virtual bool iterateToConvergence(uint32_t maxIter);
  /// Compuyte the current cost function value.
  virtual double cost();

  uint32_t GetK() const {return K_;};
  const std::vector<uint32_t>& GetNs() const {return Ns_;};
  bool GetCenter(uint32_t k, Eigen::Matrix<T,D,1>& mu) const {
    if (k<K_) {mu = mus_[k]; return true; } else { return false; } };
  const std::vector<uint32_t>& GetZs() const { return zs_;};
  bool GetX(uint32_t i, Eigen::Matrix<T,D,1>& x) const {
    if (i<xs_.size()) {x=xs_[i]; return true; } else { return false; } };

  DPMeansSimple<T,DS,D>& operator=(const DPMeansSimple<T,DS,D>& o);
  
protected:
  double lambda_;
  uint32_t K_;
  std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1>>> xs_;
  std::vector<Eigen::Matrix<T,D,1>,Eigen::aligned_allocator<Eigen::Matrix<T,D,1>>> mus_;
  std::vector<uint32_t> zs_;
  std::vector<uint32_t> Ns_;

  /// resets all clusters (mus_ and Ks_) and resizes them to K_
  void resetClusters();
  /// Removes all empty clusters.
  void removeEmptyClusters();
  /// Computes the index of the closest cluster (may be K_ in which
  /// case a new cluster has to be added).
  uint32_t indOfClosestCluster(const Eigen::Matrix<T,D,1>& x, T& sim_closest);
};

typedef DPMeansSimple<double,Spherical<double>,1> DPMeansSimpleS1d; 
typedef DPMeansSimple<double,Spherical<double>,2> DPMeansSimpleS2d; 
typedef DPMeansSimple<double,Spherical<double>,3> DPMeansSimpleS3d; 
typedef DPMeansSimple<double,Euclidean<double>,1> DPMeansSimpleE1d;
typedef DPMeansSimple<double,Euclidean<double>,2> DPMeansSimpleE2d;
typedef DPMeansSimple<double,Euclidean<double>,3> DPMeansSimpleE3d;

// -------------------------------- impl ----------------------------------
template<class T, class DS, int D>
DPMeansSimple<T,DS,D>::DPMeansSimple(double lambda)
  : lambda_(lambda), K_(0)
{}
template<class T, class DS, int D>
DPMeansSimple<T,DS,D>::DPMeansSimple(const DPMeansSimple<T,DS,D>& o) 
  : lambda_(o.lambda_), K_(o.K_), xs_(o.xs_), mus_(o.mus_), 
  zs_(o.zs_), Ns_(o.Ns_)
{}
template<class T, class DS, int D>
DPMeansSimple<T,DS,D>::~DPMeansSimple()
{}

template<class T, class DS, int D>
DPMeansSimple<T,DS,D>& DPMeansSimple<T,DS,D>::operator=(const DPMeansSimple<T,DS,D>& o) {
  if (&o == this)
    return *this;
  lambda_ = o.lambda_;
  K_ = o.K_;
  xs_ = o.xs_;
  zs_ = o.zs_;
  Ns_ = o.Ns_;
  if (o.mus_.empty()) 
    mus_.clear();
  else {
    std::cout << " copying: " << o.mus_.size() << " " << mus_.size();
    mus_ = o.mus_;
  }
  return *this;
}

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
  if (xs_.size() == 0) return;
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
  if (xs_.size() == 0) return;
  resetClusters();
  for(uint32_t i=0; i<xs_.size(); ++i) {
    ++Ns_[zs_[i]]; 
//    mus_[zs_[i]] += xs_[i];
  }
  // Euclidean mean computation
//  for(uint32_t k=0; k<K_; ++k) {
//    mus_[k] /= Ns_[k];
//  }
//  for(uint32_t k=0; k<K_; ++k) {
//    std::cout << Ns_[k] << " ";
//  }
//  std::cout << std::endl;
  DS::computeCenters(xs_, zs_, K_, mus_);
  removeEmptyClusters();
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
void DPMeansSimple<T,DS,D>::removeEmptyClusters() {
  if (K_ < 1) return;
  uint32_t kNew = K_;
  std::vector<bool> toDelete(K_,false);
  for(int32_t k=K_-1; k>-1; --k)
    if(Ns_[k] == 0) {
      toDelete[k] = true;
      cout<<"cluster k "<<k<<" empty"<<endl;
//#pragma omp parallel for 
      for(uint32_t i=0; i<xs_.size(); ++i)
        if(static_cast<int32_t>(zs_[i]) >= k) zs_[i] -= 1;
      kNew --;
    }
  uint32_t j=0;
  for(uint32_t k=0; k<K_; ++k) 
    if(toDelete[k]) { 
      mus_[j] = mus_[k];
      Ns_[j] = Ns_[k];
      ++j;
    }
//  std::cout << "K " << K_ << " -> " << kNew << std::endl;
  K_ = kNew;
  Ns_.resize(K_);
  mus_.resize(K_);
//  for(uint32_t k=0; k<K_; ++k) 
//    std::cout << mus_[k].transpose() << std::endl;
}
template<class T, class DS, int D>
double DPMeansSimple<T,DS,D>::cost() {
  double f = lambda_*K_; 
//  std::cout << "f="<<f<< std::endl;
  for(uint32_t i=0; i<xs_.size(); ++i)  {
    f += DS::dist(mus_[zs_[i]],xs_[i]);
//    std::cout << zs_[i] << ", " << xs_[i].transpose() << ", " << mus_[zs_[i]].transpose() << std::endl;
//  std::cout << "f="<<f<< std::endl;
  }
  return f;
}

template<class T, class DS, int D>
bool DPMeansSimple<T,DS,D>::iterateToConvergence(uint32_t maxIter) {
  uint32_t iter = 0;
  double fPrev = 1e99;
  double f = cost();
//  std::cout << "f=" << f << " fPrev=" << fPrev << std::endl;
  while (iter < maxIter && fabs(fPrev - f) > 0.) {
    updateCenters();
    updateLabels();
    fPrev = f;
    f = cost();
    ++iter;
//    std::cout << "f=" << f << " fPrev=" << fPrev << std::endl;
  }
//  std::cout << "iter=" << iter<< std::endl;
  return iter < maxIter;
}
}
