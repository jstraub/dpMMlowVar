#pragma once


#include <global.hpp>
//#include <Eigen/Dense>
//#include <boost/shared_ptr.hpp>

#include <boost/random/mersenne_twister.hpp>

using namespace Eigen;

template<class T>
class Clusterer
{
public:
  Clusterer(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen);
  virtual ~Clusterer();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual void updateLabels() = 0;
  virtual void updateCenters() = 0;
  virtual MatrixXu mostLikelyInds(uint32_t n, 
      Matrix<T,Dynamic,Dynamic>& deviates) = 0;
  virtual T avgIntraClusterDeviation() = 0;
  
  const VectorXu& z() const {return z_;};
  const VectorXu& counts() const {return Ns_;};
  const Matrix<T,Dynamic,Dynamic>& centroids() const {return ps_;};

  // natural distance to be used by the algorithm
  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) = 0;
  // closer in the sense of distance defined above
  virtual bool closer(T a, T b) = 0;
  // measure of disimilarity between two points (not necessarily the distance)
  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) = 0;

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
  boost::shared_ptr<Matrix<T,Dynamic,Dynamic> > spx_; // pointer to data
  Matrix<T,Dynamic,Dynamic> ps_; // centroids on the sphere
  VectorXu Ns_; // counts for each cluster
  VectorXu z_; // labels
  boost::mt19937* pRndGen_;
};

// ----------------------------- impl -----------------------------------------
template<class T>
Clusterer<T>::Clusterer(
    const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K,
    boost::mt19937* pRndGen)
  : K_(K), D_(spx->rows()), N_(spx->cols()), cost_(INFINITY), prevCost_(INFINITY),
  spx_(spx), ps_(D_,K_), Ns_(K_), z_(N_), pRndGen_(pRndGen)
{};

template<class T>
Clusterer<T>::~Clusterer()
{};

template<class T>
T Clusterer<T>::silhouette()
{ 
  if(K_<2) return -1.0;
  assert(Ns_.sum() == N_);
  Matrix<T,Dynamic,1> sil(N_);
#pragma omp parallel for
  for(uint32_t i=0; i<N_; ++i)
  {
    Matrix<T,Dynamic,1> b = Matrix<T,Dynamic,1>::Zero(K_);
    for(uint32_t j=0; j<N_; ++j)
      if(j != i)
      {
        b(z_(j)) += dissimilarity(spx_->col(i),spx_->col(j));
      }
    b *= Ns_.cast<T>().cwiseInverse(); // Assumes Ns are up to date!
    T a_i = b(z_(i)); // average dist to own cluster
    T b_i = z_(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<K_; ++k)
      if(k != z_(i) && b(k) == b(k) && b(k) < b_i && Ns_(k) > 0)
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
