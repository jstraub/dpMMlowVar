#pragma once

#include <vector>
#include <algorithm>
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
  ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, uint32_t K);
  virtual ~ClData();

  /* after changing z_ outside - we can use update to get new statistics */
  virtual void updateLabels(uint32_t K);
  virtual void computeSS();

  virtual void labelMap(const vector<int32_t>& map);

//  virtual const spVectorXu& z() const {return z_;};
  virtual VectorXu& z() {return *z_;};
  virtual uint32_t& z(uint32_t i) {return (*z_)(i);};
  virtual const spVectorXu& labels() const {return z_;};
  virtual const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x() const {return x_;};
//  virtual const Matrix<T,Dynamic,1>& x_c(uint32_t i) const {return x_->col(i);};

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
ClData<T>::ClData(const boost::shared_ptr<Matrix<T,Dynamic,Dynamic> >& x, 
    uint32_t K)
 : z_(new VectorXu(VectorXu::Zero(x->cols()))), x_(x),
  K_(K>0?K:1), N_(x->cols()), D_(x->rows())
//   Ss_(K_,Matrix<T,Dynamic,Dynamic>::Zero(D_,D_))
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
  }
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

//  cout<<xSums_<<endl;
//  cout<<Ns_.transpose()<<endl;

//  for(uint32_t k=0; k<K_; ++k)
//  {
//    xSums_.col(k) /= Ns_(k);
//    Ss_[k].setZero(D_,D_);
//  }

//  for(uint32_t i=0; i<N_; ++i)
//    Ss_[(*z_)(i)] += (x_->col(i) - xSums_.col((*z_)(i)))*
//      (x_->col(i) - xSums_.col((*z_)(i))).transpose();
}

template<class T>
void ClData<T>::labelMap(const vector<int32_t>& map)
{
    // fix labels
#pragma parallel for
    for(uint32_t i=0; i<this->N_; ++i)
      (*this->z_)(i) = map[ (*this->z_)(i)];
};
