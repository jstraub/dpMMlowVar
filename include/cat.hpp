#pragma once

#include <Eigen/Dense>
#include <time.h>
#include <memory>

//#include <boost/random/uniform_int_distribution.hpp>
//#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>

#include "distribution.hpp"
#include "sampler.hpp"

using namespace Eigen;
using std::cout;
using std::endl;
 
#ifdef BOOST_OLD
//using boost::mt19937;
#define mt19937 boost::mt19937
#else
using boost::mt19937;
#endif

template<typename T>
class Cat : public Distribution<T>
{
public:
  uint32_t K_;
  Matrix<T,Dynamic,1> pdf_;
  Matrix<T,Dynamic,1> cdf_;

  /* constructor from pdf */
  Cat(const Matrix<T,Dynamic,1>& pdf, mt19937 *pRndGen);
  /* constructor from indicators - estimates from counts */
  Cat(const VectorXu& z, mt19937 *pRndGen);
  /* copy constructor */
  Cat(const Cat& other);
  virtual ~Cat();

  uint32_t sample();
  void sample(VectorXu& z);

  T logPdf(const Matrix<T,Dynamic,1>& x) const {
    assert(false); //TODO
    return log(pdf_(0));
  };

  const Matrix<T,Dynamic,1>& pdf() const {return pdf_;};
  void pdf(const Matrix<T,Dynamic,1>& pdf){
    pdf_ = pdf;
    updateCdf();
  };
  const Matrix<T,Dynamic,1>& cdf() const {return cdf_;};

  void print() const;

private:
  boost::uniform_01<T> unif_;
  void updateCdf();
};

typedef Cat<float> Catf;
typedef Cat<double> Catd;

