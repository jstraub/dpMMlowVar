#pragma once

#include <vector>
#include <Eigen/Dense>

using std::vector;
using std::cout;
using std::endl;
using boost::shared_ptr;

/* clustered data */
template <class T>
class ClData
{
protected:
  spVectorXu z_; // labels
  boost::shared_ptr<Matrix<T,Dynamic,Dynamic> > x_; // data

  uint32_t K_; // number of classes
  uint32_t N_;
  uint32_t D_;

  Matrix<T,Dynamic,1> Ns_; // counts
  Matrix<T,Dynamic,Dynamic> xSums_; // xSums
//  vector<Matrix<T,Dynamic,Dynamic> > Ss_; //scatter matrices

public:
  ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
      const spVectorXu& z, uint32_t K);
  virtual ~ClData();

  /* after changing z_ outside - we can use update to get new statistics */
  virtual void updateLabels(uint32_t K);
  virtual void computeSS();

//  virtual const spVectorXu& z() const {return z_;};
  virtual VectorXu& z() {return *z_;};
  virtual const spVectorXu& labels() const {return z_;};
  virtual const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x() const {return x_;};
  virtual const Matrix<T,Dynamic,Dynamic>& xMat() const {return (*x_);};
  virtual uint32_t N() const {return N_;};
  virtual uint32_t K() const {return K_;};
  virtual uint32_t D() const {return D_;};

  virtual const Matrix<T,Dynamic,1>& counts() const {return Ns_;};
  virtual T count(uint32_t k) const {return Ns_(k);};

  virtual const Matrix<T,Dynamic,Dynamic>& xSums() const {return xSums_;};
  virtual Matrix<T,Dynamic,1> xSum(uint32_t k) const {return xSums_.col(k);};

//  virtual const vector<Matrix<T,Dynamic,Dynamic> >& scatters() const 
//  {return Ss_;};
//  virtual const Matrix<T,Dynamic,Dynamic>& S(uint32_t k) const 
//  {return Ss_[k];};
};

typedef ClData<float> ClDataf;
typedef ClData<double> ClDatad;


//class ClDataGpu : public ClData
//{
//  
//};

// -------------------------- impl --------------------------------------------
template<class T>
ClData<T>::ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    const spVectorXu& z, uint32_t K)
 : z_(z), x_(x), K_(K>0?K:z->maxCoeff()+1), N_(x->cols()), D_(x->rows())
//   Ss_(K_,Matrix<T,Dynamic,Dynamic>::Zero(D_,D_))
{
  cout<<"D="<<D_<<" N="<<N_<<" K="<<K_<<endl;
};

template<class T>
ClData<T>::~ClData()
{};

template<class T>
void ClData<T>::updateLabels(uint32_t K)
{
  K_ = K;
}

template<class T>
void ClData<T>::computeSS()
{
  Ns_.setZero(K_);
  xSums_.setZero(D_,K_);

  for (uint32_t i=0; i<N_; ++i)
//  {
//    cout<<"N="<<N_<<" z_i="<<(*z_)(i)<<" i="<<i<<" K="<<K_<<endl;
    Ns_((*z_)(i))++;
//  }

  for(uint32_t i=0; i<N_; ++i)
    xSums_.col((*z_)(i)) += x_->col(i);

//  for(uint32_t k=0; k<K_; ++k)
//  {
//    xSums_.col(k) /= Ns_(k);
//    Ss_[k].setZero(D_,D_);
//  }

//  for(uint32_t i=0; i<N_; ++i)
//    Ss_[(*z_)(i)] += (x_->col(i) - xSums_.col((*z_)(i)))*
//      (x_->col(i) - xSums_.col((*z_)(i))).transpose();
}

