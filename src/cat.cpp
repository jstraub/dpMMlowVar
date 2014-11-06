#include "cat.hpp"

// ---------------------------------------------------------------------------

template<typename T>
Cat<T>::Cat(const Matrix<T,Dynamic,1>& pdf, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), K_(pdf.size()), pdf_(pdf)
{
  updateCdf();
};

template<typename T>
Cat<T>::Cat(const VectorXu& z, boost::mt19937 *pRndGen)
: Distribution<T>(pRndGen), K_(z.maxCoeff()+1)
{
  pdf_ = counts<T,uint32_t>(z,K_);
  pdf_ /= pdf_.sum();

  updateCdf();
};

template<typename T>
Cat<T>::Cat(const Cat<T>& other)
  : Distribution<T>(other.pRndGen_), K_(other.K_), pdf_(other.pdf_), 
    cdf_(other.cdf_)
{};

template<typename T>
Cat<T>::~Cat()
{};

template<typename T>
uint32_t Cat<T>::sample()
{
  T r=unif_(*this->pRndGen_);
  //cout<<cdf_.transpose()<<" -> "<<r<<endl;
  for (uint32_t k=0; k<K_; ++k)
    if (r<cdf_(k)){return k-1;}
  return K_-1;
};

template<typename T>
void Cat<T>::sample(VectorXu& z)
{
  for(uint32_t i=0; i<z.size(); ++i)
    z(i) = this->sample();
};

template<typename T>
void Cat<T>::updateCdf()
{
  cdf_.setZero(pdf_.size()+1);
  for (uint32_t k=0; k<K_; ++k)
    cdf_(k+1) = cdf_(k)+pdf_(k);
}

template<typename T>
void Cat<T>::print() const
{
  cout<<"pi="<<pdf_.transpose()<<endl;
}

template class Cat<double>;
template class Cat<float>;
