#pragma once

#include <Eigen/Dense>
#include <time.h>
#include <vector>

#include <boost/random/gamma_distribution.hpp> // for gamma_distribution.
#include <boost/math/special_functions/gamma.hpp>

#include "distribution.hpp"
#include "cat.hpp"
//#include "mult.hpp"

using namespace Eigen;
using std::cout;
using std::endl;
using std::vector;

#ifdef BOOST_OLD
using boost::gamma_distribution;
#define mt19937 boost::mt19937
#else
using boost::mt19937;
using boost::random::gamma_distribution;
#endif

template<class Disc, typename T>
class Dir : public Distribution<T>
{
public:
  uint32_t K_;
  Matrix<T,Dynamic,1> alpha_;

  Dir(const Matrix<T,Dynamic,1>& alpha, mt19937 *pRndGen);
  Dir(const Dir& other);
  ~Dir();

  Dir<Disc,T>* copy();

  Disc sample();
  Dir<Disc,T> posterior() const;
  Dir<Disc,T> posterior(const VectorXu& z);
  Dir<Disc,T> posterior(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, uint32_t k);
  Dir<Disc,T> posteriorFromCounts(const Matrix<T,Dynamic,1>& counts);
  Dir<Disc,T> posteriorFromCounts(const VectorXu& counts);

  T logPdf(const Disc& cat);

  uint32_t K(){return K_;}

  T logPdf(const Disc& cat) const;
  T logPdfMarginalized() const; // log pdf of SS under NIW prior
  T logPdfUnderPriorMarginalizedMerged(const Dir<Disc,T>& other) const;

  T logLikelihoodMarginalized(const Matrix<T,Dynamic,1>& counts) const;
  void print() const;

  virtual Dir<Disc,T>* merge(const Dir<Disc,T>& other);
  void fromMerge(const Dir<Disc,T>& niwA, const Dir<Disc,T>& niwB);

//  const Matrix<T,Dynamic,Dynamic>& scatter() const {return scatter_;};
//  Matrix<T,Dynamic,Dynamic>& scatter() {return scatter_;};
//  const Matrix<T,Dynamic,1>& mean() const {return mean_;};
//  Matrix<T,Dynamic,1>& mean() {return mean_;};
//  T count() const {return count_;};
//  T& count() {return count_;};
//
  const Matrix<T,Dynamic,1>& counts() const {return counts_;};
  Matrix<T,Dynamic,1>& counts() {return counts_;};
  T count() const {return counts_.sum();};

  void computeMergedSS( const Dir<Disc,T>& dirA, 
      const Dir<Disc,T>& dirB, Matrix<T,Dynamic,1>& NsM) const;

private:

  Matrix<T,Dynamic,1> counts_; // counts for the different classes -> SS
  vector<gamma_distribution<> > gammas_;

  Matrix<T,Dynamic,1> samplePdf();
};

typedef Dir<Cat<double>, double> DirCatd;
typedef Dir<Cat<float>, float> DirCatf;
//typedef Dir<Mult<double>, double> DirMultd;
//typedef Dir<Mult<float>, float> DirMultf;
#include <dir.cpp>
