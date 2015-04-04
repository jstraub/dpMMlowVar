/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <vector>
#include <algorithm>
#include <Eigen/Dense>

#include <jsCore/global.hpp>

using namespace Eigen;
using std::vector;
using std::cout;
using std::endl;

#define UNASSIGNED 4294967294

namespace jsc {

/* clustered data */
template <class T>
class ClData
{
protected:
  spVectorXu z_; // labels
  shared_ptr<Matrix<T,Dynamic,Dynamic> > x_; // data

  uint32_t K_; // number of classes
  uint32_t N_;
  uint32_t D_;

  Matrix<T,Dynamic,1> Ns_; // counts
  Matrix<T,Dynamic,Dynamic> xSums_; // xSums

public:
  ClData(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
      const spVectorXu& z, uint32_t K);
  ClData(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, uint32_t K);
  ClData(uint32_t D, uint32_t K);
  virtual ~ClData();

  /* after changing z_ outside - we can use update to get new statistics */
  virtual void updateLabels(uint32_t K);
  virtual void randomLabels(uint32_t K);
  virtual void computeSS();

  virtual void labelMap(const vector<int32_t>& map);

  virtual void updateK(uint32_t K){ K_ = K;};
  virtual void updateData(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x);
  virtual void updateData(T* d_x, uint32_t N, uint32_t step, uint32_t offset);

  virtual VectorXu& z() {return *z_;};
  virtual uint32_t& z(uint32_t i) const {return (*z_)(i);};
  virtual const spVectorXu& labels() const {return z_;};
  virtual const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x() const {return x_;};

 virtual T* d_x(){ return x_->data();};
 virtual uint32_t* d_z(){ return z_->data();};

  virtual const Matrix<T,Dynamic,Dynamic>& xMat() const {return (*x_);};
  virtual uint32_t N() const {return N_;};
  virtual uint32_t K() const {return K_;};
  virtual uint32_t D() const {return D_;};

  virtual const Matrix<T,Dynamic,1>& counts() const {return Ns_;};
  virtual T count(uint32_t k) const {return Ns_(k);};

  virtual const Matrix<T,Dynamic,Dynamic>& xSums() const {return xSums_;};
  virtual Matrix<T,Dynamic,1> xSum(uint32_t k) const {return xSums_.col(k);};


};

typedef ClData<float> ClDataf;
typedef ClData<double> ClDatad;

template<class T, class DS>
T silhouetteClD(const ClData<T>& cld)
{ 
  if(cld.K()<2) return -1.0;
//  assert(Ns_.sum() == N_);
  Matrix<T,Dynamic,1> sil(cld.N());
#pragma omp parallel for
  for(uint32_t i=0; i<cld.N(); ++i)
  {
    Matrix<T,Dynamic,1> b = Matrix<T,Dynamic,1>::Zero(cld.K());
    for(uint32_t j=0; j<cld.N(); ++j)
      if(j != i)
      {
        b(cld.z(j)) += DS::dissimilarity(cld.x()->col(i),cld.x()->col(j));
      }
    for (uint32_t k=0; k<cld.K(); ++k) b(k) /= cld.count(k);
//    b *= Ns_.cast<T>().cwiseInverse(); // Assumes Ns are up to date!
    T a_i = b(cld.z(i)); // average dist to own cluster
    T b_i = cld.z(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<cld.K(); ++k)
      if(k != cld.z(i) && b(k) == b(k) && b(k) < b_i && cld.count(k) > 0)
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
  return sil.sum()/static_cast<T>(cld.N());
};

// -------------------------- impl --------------------------------------------
template<class T>
ClData<T>::ClData(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    const spVectorXu& z, uint32_t K)
 : z_(z), x_(x), K_(z->maxCoeff()+1), N_(x->cols()), D_(x->rows())
{
  cout<<"D="<<D_<<" N="<<N_<<" K="<<K_<<endl;
};

template<class T>
ClData<T>::ClData(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    uint32_t K)
 : z_(new VectorXu(VectorXu::Zero(x->cols()))), x_(x),
  K_(K), N_(x->cols()), D_(x->rows())
{
  cout<<"D="<<D_<<" N="<<N_<<" K="<<K_<<endl;
  // randomly init z
  if(K_ > 1)
  {
    this->randomLabels(K_);
    cout<<"init z: "<<(*z_).transpose()<<endl;
  }else if(K==1)
    z_->fill(0.);
  else
    z_->fill(UNASSIGNED);
};

template<class T>
ClData<T>::ClData(uint32_t D, uint32_t K)
 : z_(new VectorXu(VectorXu::Zero(0))), x_(new Matrix<T,Dynamic,Dynamic>(D,0)), K_(K), N_(0),
  D_(D)
{
  cout<<"D="<<D_<<" N="<<N_<<" K="<<K_<<endl;
};

template<class T>
ClData<T>::~ClData()
{};

template<class T>
void ClData<T>::randomLabels(uint32_t K)
{
  std::vector<uint32_t> z(N_);
  for(uint32_t i=0; i<N_; ++i) z[i] = i%K;
  // to destreuy symmetry
  for(uint32_t i=0; i<K_; ++i) z[i] = 0;
  std::random_shuffle(z.begin(),z.end());
  for(uint32_t i=0; i<N_; ++i) (*z_)(i) = z[i];
}

template<class T>
void ClData<T>::updateLabels(uint32_t K)
{
  K_ = K;
}

template<class T>
void ClData<T>::updateData(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& x)
{
  x_ = x;
  assert(D_ == x_->rows());
  if(N_ != x->cols())
  {
    N_ = x->cols();
    z_->resize(N_);
    z_->fill(UNASSIGNED);
  }
};

template<class T>
void ClData<T>::updateData(T* d_x, uint32_t N, uint32_t step, uint32_t
    offset)
{
  assert(false);
  //TODO: implement this
};

template<class T>
void ClData<T>::computeSS()
{
  Ns_.setZero(K_);
  xSums_.setZero(D_,K_);

#pragma omp parallel for
  for(uint32_t k=0; k<K_; ++k)
    for(uint32_t i=0; i<N_; ++i)
      if(k == (*z_)(i))
    {
      xSums_.col(k) += x_->col(i);
      Ns_(k)++;
    }
}

template<class T>
void ClData<T>::labelMap(const vector<int32_t>& map)
{
    // fix labels
#pragma omp parallel for
    for(uint32_t i=0; i<this->N_; ++i)
      (*this->z_)(i) = map[ (*this->z_)(i)];
};

}
