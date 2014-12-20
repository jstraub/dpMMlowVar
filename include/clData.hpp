#pragma once

#include <vector>
#include <algorithm>
#include <Eigen/Dense>

using std::vector;
using std::cout;
using std::endl;
using boost::shared_ptr;

#define UNASSIGNED 4294967295

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

public:
  ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
      const spVectorXu& z, uint32_t K);
  ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, uint32_t K);
  virtual ~ClData();

  /* after changing z_ outside - we can use update to get new statistics */
  virtual void updateLabels(uint32_t K);
  virtual void computeSS();

  virtual void labelMap(const vector<int32_t>& map);

  virtual void updateK(uint32_t K){ K_ = K;};
  virtual void updateData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x);

  virtual VectorXu& z() {return *z_;};
  virtual uint32_t& z(uint32_t i) {return (*z_)(i);};
  virtual const spVectorXu& labels() const {return z_;};
  virtual const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x() const {return x_;};

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

// -------------------------- impl --------------------------------------------
template<class T>
ClData<T>::ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    const spVectorXu& z, uint32_t K)
 : z_(z), x_(x), K_(z->maxCoeff()+1), N_(x->cols()), D_(x->rows())
{
  cout<<"D="<<D_<<" N="<<N_<<" K="<<K_<<endl;
};

template<class T>
ClData<T>::ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    uint32_t K)
 : z_(new VectorXu(VectorXu::Zero(x->cols()))), x_(x),
  K_(K), N_(x->cols()), D_(x->rows())
{
  cout<<"D="<<D_<<" N="<<N_<<" K="<<K_<<endl;
  // randomly init z
  if(K_ > 1)
  {
    std::vector<uint32_t> z(N_);
    for(uint32_t i=0; i<N_; ++i) z[i] = i%K_;
     // to destrey symmetry
    for(uint32_t i=0; i<K_; ++i) z[i] = 0;
    std::random_shuffle(z.begin(),z.end());
    for(uint32_t i=0; i<N_; ++i) (*z_)(i) = z[i];
    cout<<"init z: "<<(*z_).transpose()<<endl;
  }else if(K==1)
    z_->fill(0.);
  else
    z_->fill(UNASSIGNED);
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
void ClData<T>::updateData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x)
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
void ClData<T>::computeSS()
{
  Ns_.setZero(K_);
  xSums_.setZero(D_,K_);

#pragma omp prallel for
  for(uint32_t i=0; i<N_; ++i)
  {
    xSums_.col((*z_)(i)) += x_->col(i);
    Ns_((*z_)(i))++;
  }
}

template<class T>
void ClData<T>::labelMap(const vector<int32_t>& map)
{
    // fix labels
#pragma parallel for
    for(uint32_t i=0; i<this->N_; ++i)
      (*this->z_)(i) = map[ (*this->z_)(i)];
};
