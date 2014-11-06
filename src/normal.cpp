#include "normal.hpp"


// ---------------------------------------------------------------------------

template<typename T>
Normal<T>::Normal(const Matrix<T,Dynamic,1>& mu, const Matrix<T,Dynamic,Dynamic>& Sigma,boost::mt19937 *pRndGen)
  : Distribution<T>(pRndGen), D_(mu.size()), mu_(mu), Sigma_(Sigma), 
    SigmaLDLT_(Sigma_)
{
  assert(mu_.size() == Sigma_.cols());
  assert(mu_.size() == Sigma_.rows());

  // equivalent to log(det(Sigma)) but more stable for small values
  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
};

template<typename T>
Normal<T>::Normal(const Matrix<T,Dynamic,Dynamic>& Sigma, boost::mt19937 *pRndGen)
  : Distribution<T>(pRndGen),D_(Sigma.cols()), Sigma_(Sigma),  
    SigmaLDLT_(Sigma_)
{
  mu_.setZero(D_);

  assert(mu_.size() == Sigma_.cols());
  assert(mu_.size() == Sigma_.rows());

  // equivalent to log(det(Sigma)) but more stable for small values
  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
};

template<typename T>
Normal<T>::Normal(uint32_t D, boost::mt19937 *pRndGen)
  : Distribution<T>(pRndGen), D_(D)
{

  Sigma_ = Matrix<T,Dynamic,Dynamic>::Identity(D_,D_);
  mu_.setZero(D_);

  assert(mu_.size() == Sigma_.cols());
  assert(mu_.size() == Sigma_.rows());

  setSigma(Sigma_);
//  SigmaLDLT_.compute(Sigma_);
//  // equivalent to log(det(Sigma)) but more stable for small values
//  logDetSigma_ = ((Sigma_.eigenvalues()).array().log().sum()).real();
};

template<typename T>
Normal<T>::Normal(const Normal<T>& other)
  : Distribution<T>(other.pRndGen_),D_(mu_.size()), mu_(other.mu_), 
  Sigma_(other.Sigma_), logDetSigma_(other.logDetSigma_), SigmaLDLT_(Sigma_)
{};

template<typename T>
Normal<T>::~Normal()
{}

template<typename T>
T Normal<T>::logPdf(const Matrix<T,Dynamic,Dynamic>& x) const
{
#ifndef NDEBUG
  assert(logDetSigma_ == ((Sigma_.eigenvalues()).array().log().sum()).real());
  Matrix<T,Dynamic,1> b = SigmaLDLT_.solve(x-mu_);
  Matrix<T,Dynamic,1> c = Sigma_.fullPivHouseholderQr().solve(x-mu_);
//  cout<<(b-Sigma_.fullPivHouseholderQr().solve(x-mu_)).transpose()<<endl;
//  ASSERT(((b-Sigma_.fullPivHouseholderQr().solve(x-mu_)).array().abs() < 1.e-6).all() , (b-Sigma_.fullPivHouseholderQr().solve(x-mu_)).transpose() 
  ASSERT(( fabs( (b.transpose()*b)(0) - (c.transpose()*c)(0)) / (b.transpose()*b)(0)  < 1.e-6), 
      (b-Sigma_.fullPivHouseholderQr().solve(x-mu_)).transpose() 
      << endl << Sigma_<< endl << SigmaLDLT_.reconstructedMatrix()<<endl
      << Sigma_ - SigmaLDLT_.reconstructedMatrix()<< endl
      << b.transpose()<< " vs "<< Sigma_.fullPivHouseholderQr().solve(x-mu_).transpose()<<endl
      << "x="<<x.transpose()<< "mu_="<<mu_.transpose());

#endif
  return -0.5*(LOG_2PI*D_ + logDetSigma_
//      +((x-mu_).transpose()*Sigma_.inverse()*(x-mu_)).sum() );
  +((x-mu_).transpose()*SigmaLDLT_.solve(x-mu_)).sum() );
}

template<typename T>
T Normal<T>::logPdfSlower(const Matrix<T,Dynamic,Dynamic>& x) const
{
  return -0.5*(LOG_2PI*D_ + logDetSigma_
//      +((x-mu_).transpose()*Sigma_.inverse()*(x-mu_)).sum() );
  +((x-mu_).transpose()*Sigma_.fullPivHouseholderQr().solve(x-mu_)).sum() );
}

template<typename T>
T Normal<T>::logPdf(const Matrix<T,Dynamic,Dynamic>& scatter, 
      const Matrix<T,Dynamic,1>& mean, T count) const
{
  return -0.5*((LOG_2PI*D_ + logDetSigma_)*count
      + count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum() 
      -2.*count*(mean.transpose()*SigmaLDLT_.solve(mu_)).sum()
      +(SigmaLDLT_.solve(scatter + mean*mean.transpose()*count )).trace());
}

template<typename T>
T Normal<T>::logPdf(const Matrix<T,Dynamic,Dynamic>& scatter, T count) const
{
  assert(false);
//  cout<<count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum()<<endl;
//  cout<<(SigmaLDLT_.solve(scatter)).trace()<<endl;
  return -0.5*((LOG_2PI*D_ + logDetSigma_)*count
      + count*(mu_.transpose()*SigmaLDLT_.solve(mu_)).sum() 
      +(SigmaLDLT_.solve(scatter)).trace());
}

template<typename T>
Matrix<T,Dynamic,1> Normal<T>::sample()
{
  // populate the mean
  Matrix<T,Dynamic,1> x(D_);
  for (uint32_t d=0; d<D_; d++)
    x[d] = gauss_(*this->pRndGen_); //gsl_ran_gaussian(r,1);
  return Sigma_*x + mu_;
};

template<typename T>
void Normal<T>::print() const
{
  cout<<"mu="<<mu_.transpose()<<endl;
  cout<<Sigma_<<endl;
}

template class Normal<double>;
template class Normal<float>;
