#pragma once

#include <vector>
#include <global.hpp>
#include <clData.hpp>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>


using namespace Eigen;
using std::vector;

template<class T, class DS>
class Clusterer
{
public:
  Clusterer(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K);
  Clusterer(const shared_ptr<ClData<T> >& cld);
  virtual ~Clusterer();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual void updateLabels() = 0;
  virtual void updateCenters() = 0;
  virtual MatrixXu mostLikelyInds(uint32_t n, 
      Matrix<T,Dynamic,Dynamic>& deviates) = 0;
  virtual T avgIntraClusterDeviation() = 0;

  virtual void nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);
  virtual void updateState(){}; // after converging for a single time instant
  
  const VectorXu& z() const {return (this->cld_->z());};
  VectorXu counts() const {
    VectorXu Ns(K_);
    for(uint32_t k=0; k<K_; ++k) Ns(k) = cls_[k]->N();
    return Ns;
  };
  Matrix<T,Dynamic,Dynamic> centroids() const {
    Matrix<T,Dynamic,Dynamic> ps(D_,K_);
    for(uint32_t k=0; k<K_; ++k)                                                
      ps.col(k) = cls_[k]->centroid();
    return ps;
  };

  // natural distance to be used by the algorithm
//  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) = 0;
  // closer in the sense of distance defined above
//  virtual bool closer(T a, T b) = 0;
  // measure of disimilarity between two points (not necessarily the distance)
//  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) = 0;

  virtual uint32_t getK(){return K_;};
  virtual uint32_t K(){return K_;}; 
  virtual bool converged(T eps=1e-6){
//    cout<<cost_<<" "<<prevCost_<<" "<<fabs(cost_-prevCost_)<<endl ; 
    return fabs(cost_-prevCost_)<eps;}; 
  virtual T cost(){return cost_;}; 

  virtual T silhouette();

protected:
  uint32_t K_;
  const uint32_t D_;
  uint32_t N_;
  T cost_, prevCost_;
  shared_ptr<ClData<T> > cld_;
//  shared_ptr<Matrix<T,Dynamic,Dynamic> > spx_; // pointer to data
  vector< shared_ptr<typename DS::DependentCluster> > cls_; // clusters
//  VectorXu z_; // labels
};

// ----------------------------- impl -----------------------------------------
template<class T, class DS>
Clusterer<T,DS>::Clusterer( const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx,
    uint32_t K)
  : K_(K), D_(spx->rows()), N_(spx->cols()), cost_(INFINITY), prevCost_(INFINITY),
  cld_(new ClData<T>(spx,K))
{
  for (uint32_t k=0; k<K_; ++k)
    cls_.push_back(shared_ptr<typename DS::DependentCluster >(new typename DS::DependentCluster()));
};

template<class T, class DS>
Clusterer<T,DS>::Clusterer( const shared_ptr<ClData<T> >& cld)
  : K_(cld->K()), D_(cld->D()), N_(cld->N()), cost_(INFINITY), prevCost_(INFINITY),
  cld_(cld)
{
  for (uint32_t k=0; k<K_; ++k)
    cls_.push_back(shared_ptr<typename DS::DependentCluster >(new typename DS::DependentCluster()));
};

template<class T, class DS>
void Clusterer<T,DS>::nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
{
  // reset cluster centers
  cls_.clear();
  for (uint32_t k=0; k<K_; ++k)
    cls_.push_back(shared_ptr<typename DS::DependentCluster >(new typename DS::DependentCluster()));
  // update the data
  this->cld_->updateData(spx);
  this->N_ = this->cld_->N();

  this->cld_->randomLabels(K_);
  this->cld_->updateLabels(K_);
  this->cld_->computeSS();
  for(uint32_t k=0; k<this->K_; ++k)
    this->cls_[k]->updateCenter(this->cld_,k);
};

template<class T, class DS>
Clusterer<T,DS>::~Clusterer()
{};

template<class T, class DS>
T Clusterer<T,DS>::silhouette()
{ 
  if(K_<2) return -1.0;
//  assert(Ns_.sum() == N_);
  Matrix<T,Dynamic,1> sil(N_);
#pragma omp parallel for
  for(uint32_t i=0; i<N_; ++i)
  {
    Matrix<T,Dynamic,1> b = Matrix<T,Dynamic,1>::Zero(K_);
    for(uint32_t j=0; j<N_; ++j)
      if(j != i)
      {
        b(cld_->z(j)) += DS::dissimilarity(cld_->x()->col(i),cld_->x()->col(j));
      }
    for (uint32_t k=0; k<K_; ++k) b /= cls_[k]->N();
//    b *= Ns_.cast<T>().cwiseInverse(); // Assumes Ns are up to date!
    T a_i = b(cld_->z(i)); // average dist to own cluster
    T b_i = cld_->z(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<K_; ++k)
      if(k != cld_->z(i) && b(k) == b(k) && b(k) < b_i && cls_[k]->N() > 0)
      {
        b_i = b(k);
      }
    if(a_i < b_i)
      sil(i) = 1.- a_i/b_i;
    else if(a_i > b_i)
      sil(i) = b_i/a_i - 1.;
    else
      sil(i) = 0.;
  }
  return sil.sum()/static_cast<T>(N_);
};
