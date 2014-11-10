//#include "dir.hpp"

// ----------------------------------------------------------------------------
template<class Disc, typename T>
Dir<Disc,T>::Dir(const Matrix<T,Dynamic,1>& alpha, mt19937* pRndGen) 
  : Distribution<T>(pRndGen), K_(alpha.size()), alpha_(alpha), 
  counts_(Matrix<T,Dynamic,1>::Zero(K_))
{
  for(uint32_t k=0; k<K_; ++k)
    gammas_.push_back(gamma_distribution<>(alpha_(k)));
};

template<class Disc, typename T>
Dir<Disc,T>::Dir(const Dir& other)
  : Distribution<T>(other.pRndGen_), K_(other.K_), alpha_(other.alpha_), 
  counts_(other.counts())
{
  for(uint32_t k=0; k<K_; ++k)
    gammas_.push_back(gamma_distribution<>(alpha_(k)));
};

template<class Disc, typename T>
Dir<Disc,T>* Dir<Disc,T>::copy()
{
  Dir<Disc,T>* cp = new Dir<Disc,T>(*this);
  return cp;
};


template<class Disc, typename T>
Dir<Disc,T>::~Dir()
{};

template<class Disc, typename T>
Matrix<T,Dynamic,1> Dir<Disc,T>::samplePdf()
{
  // sampling from Dir via gamma distribution 
  // http://en.wikipedia.org/wiki/Dirichlet_distribution
  Matrix<T,Dynamic,1> pi(K_);
  for (uint32_t k=0; k<K_; ++k)
    pi(k) = gammas_[k](*this->pRndGen_);
  return pi/pi.sum();
};

template<class Disc, typename T>
Disc Dir<Disc,T>::sample()
{
  return Disc(this->samplePdf(),this->pRndGen_);
};

template<class Disc, typename T>
Dir<Disc,T> Dir<Disc,T>::posterior(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, uint32_t k)
{
//  cout << "posterior alpha: "<<(alpha_+counts(z,K_)).transpose()<<endl;
  counts_.setZero(K_);
  for (uint32_t i=0; i<z.size(); ++i)
    if(z(i) == k)
      counts_ += x.col(i); // TODO: for Mult<T>
  return posterior();   
};


template<class Disc, typename T>
Dir<Disc,T> Dir<Disc,T>::posterior(const VectorXu& z)
{
//  cout << "posterior alpha: "<<(alpha_+counts(z,K_)).transpose()<<endl;
  counts_.setZero(K_);
  for (uint32_t i=0; i<z.size(); ++i)
    counts_(z(i))++; 
  return posterior();   
};

template<class Disc, typename T>
Dir<Disc,T> Dir<Disc,T>::posterior() const
{
//  cout << "posterior alpha: "<<(alpha_+counts(z,K_)).transpose()<<endl;
  return Dir<Disc,T>(alpha_+counts_,this->pRndGen_);   
};

template<class Disc, typename T>
Dir<Disc,T> Dir<Disc,T>::posteriorFromCounts(const Matrix<T,Dynamic,1>& counts)
{
  counts_ = counts;
  return posterior();
};

template<class Disc, typename T>
Dir<Disc,T> Dir<Disc,T>::posteriorFromCounts(const VectorXu& counts)
{
  counts_ = counts.cast<T>();
  return posterior();
};

template<class Disc, typename T>
T Dir<Disc,T>::logPdf(const Disc& disc)
{
  //gammaln(np.sum(s.alpha)) - np.sum(gammaln(s.alpha))
  //+ np.sum((s.alpha-1)*np.log(pi)) 
  T logPdf = boost::math::lgamma(alpha_.sum());
  for(uint32_t k=0; k<K_; ++k)
    logPdf += -boost::math::lgamma(alpha_[k]) + (alpha_[k]-1.)*log(disc.pdf()[k]);
  return logPdf;
};

template<class Disc, typename T>
T Dir<Disc,T>::logPdf(const Disc& disc) const
{
  T logPdf = lgamma(alpha_.sum());
  for(uint32_t k=0; k<K_; ++k)
    logPdf +=  (alpha_(k)-1.0)*log(disc.pdf()(k)) - lgamma(alpha_(k));
  return logPdf;
};

template<class Disc, typename T>
T Dir<Disc,T>::logLikelihoodMarginalized(const Matrix<T,Dynamic,1>& counts) const
{
  T logPdf = lgamma(alpha_.sum()) - lgamma(counts.sum()+alpha_.sum());
  for(uint32_t k=0; k<K_; ++k)
    logPdf += lgamma(alpha_(k)+counts(k)) - lgamma(alpha_(k));
  return logPdf;
};

template<class Disc, typename T>
T Dir<Disc,T>::logPdfMarginalized() const
{
    return logLikelihoodMarginalized(counts_);
};

template<class Disc, typename T>
T Dir<Disc,T>::logPdfUnderPriorMarginalizedMerged(const Dir<Disc,T>& other) const
{
  Matrix<T,Dynamic,1> counts;
  computeMergedSS(*this,other,counts);
  return logLikelihoodMarginalized(counts);
}


template<class Disc, typename T>
void Dir<Disc,T>::print() const
{
  cout<<"alpha="<<alpha_.transpose()<<endl; 
};

template<class Disc, typename T>
Dir<Disc,T>* Dir<Disc,T>::merge(const Dir<Disc,T>& other)
{
  Dir<Disc,T>* merged = this->copy();
  merged->fromMerge(*this,other);
  return merged;
};

template<class Disc, typename T>
void Dir<Disc,T>::fromMerge(const Dir<Disc,T>& dirA, const Dir<Disc,T>& dirB)
{
  computeMergedSS(dirA,dirB,counts_); 
};

template<class Disc, typename T>
void Dir<Disc,T>::computeMergedSS( const Dir<Disc,T>& dirA, 
    const Dir<Disc,T>& dirB, Matrix<T,Dynamic,1>& NsM) const
{
  NsM = dirA.counts() + dirB.counts();
};

//template class Dir<Cat<double>, double>;
//template class Dir<Cat<float>, float>;
//template class Dir<Mult<double>, double>;
//template class Dir<Mult<float>, float>;
