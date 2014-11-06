#pragma once

#include <stdint.h>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>

#include <boost/random/mersenne_twister.hpp>

#include "global.hpp"
#include "normal.hpp"

using  namespace Eigen;
using std::min;
using std::max;
using std::cout;
using std::endl;

#ifdef BOOST_OLD
using boost::mt19937;
#else
using boost::mt19937;
#endif


// TODO Template needs fixing - probably just remove
template <typename T>
class Sphere
{
public:
  Sphere(uint32_t D) : D_(D), north_(Matrix<T,Dynamic,1>::Zero(D))
  {north_(D_-1) = 1.0;};
  virtual ~Sphere() {};

  /* 
   * normal in tangent space around p rotate to the north pole
   * -> the last dimension will always be 0
   * -> return only first 2 dims
   */
  virtual Matrix<T,Dynamic,Dynamic> Log_p_north(const Matrix<T,Dynamic,1>& p, 
      const Matrix<T,Dynamic,Dynamic>& q) const;
  /* 
   * rotate points x in tangent plane around north pole down to p
   */
  Matrix<T,Dynamic,Dynamic> rotate_north2p(const Matrix<T,Dynamic,1>& p, 
      const Matrix<T,Dynamic,Dynamic>& xNorth) const;

  /* rotate points x in tangent plane around p to north pole and
   * return 2D coordinates */
  Matrix<T,Dynamic,Dynamic> rotate_p2north(const Matrix<T,Dynamic,1>& p, 
      const Matrix<T,Dynamic,Dynamic>& x) const;
  void rotate_p2north( const Matrix<T,Dynamic,Dynamic>& ps, 
    const Matrix<T,Dynamic,Dynamic>& x, Matrix<T,Dynamic,Dynamic>& xNorth,
    const VectorXu& z, uint32_t K) const;
  void rotate_p2north(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& x_p, Matrix<T,Dynamic,Dynamic>& xNorth,
    const VectorXu& z, uint32_t k) const;


  /* compute rotation from TpS^2 to north pole on sphere */
  Matrix<T,Dynamic,Dynamic> north_R_TpS2(const Matrix<T,Dynamic,1>& p) const;

  /* compute logarithm map from sphere to T_pS */
  Matrix<T,Dynamic,Dynamic> Log_p(const Matrix<T,Dynamic,1>& p, 
      const Matrix<T,Dynamic,Dynamic>& q) const;
  Matrix<T,Dynamic,1> Log_p_single(const Matrix<T,Dynamic,1>& p, 
      const Matrix<T,Dynamic,1>& q) const;
  /* compute logarithm map from sphere to T_pS skips points where w_i==0 */
  Matrix<T,Dynamic,Dynamic> Log_p(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& q, const Matrix<T,Dynamic,1>& w) const;
  /* compute logmap for q where z_i ==k; leaves other x_i untouched */
  void Log_p(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& q, const VectorXu& z, uint32_t k,
    Matrix<T,Dynamic,Dynamic>& x, uint32_t zDivider=1) const;
  /* compute logarithm map from sphere to T_pS around several points p 
   * indicated by labels z; fill in matrix x */
  void Log_ps(const Matrix<T,Dynamic,Dynamic>& p, 
    const Matrix<T,Dynamic,Dynamic>& q, const VectorXu& z, 
    Matrix<T,Dynamic,Dynamic>& x) const;

  /* compute exponential map from T_pS to sphere */
  Matrix<T,Dynamic,Dynamic> Exp_p(const Matrix<T,Dynamic,1>& p, 
      const Matrix<T,Dynamic,Dynamic>& x) const;
  Matrix<T,Dynamic,1> Exp_p_single(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,1>& x) const;

  Matrix<T,Dynamic,1> sampleUnif(mt19937* pRndGen);
 
  // http://en.wikipedia.org/wiki/N-sphere
  T logSurfaceArea() const {
    return (LOG_2+0.5*D_*LOG_PI - boost::math::lgamma(0.5*D_));};

  uint32_t D() const {return D_;};
  const Matrix<T,Dynamic,1>& north() const {return north_;};
protected:

  T invSincDot(T dot) const;

  static const double MIN_DOT = -0.95;
  static const double MAX_DOT = 0.95;
  uint32_t D_; // dimension of ambient space; sphere is D-1 dimensional.
  Matrix<T,Dynamic,1> north_;
};

/* rotation from point A to B; percentage specifies how far the rotation will 
 * bring us towards B [0,1] */
template<typename T>
inline Matrix<T,Dynamic,Dynamic> rotationFromAtoB(const Matrix<T,Dynamic,1>& a,
    const Matrix<T,Dynamic,1>& b, T percentage=1.0)
{
  assert(b.size() == a.size());

  uint32_t D_ = b.size();
  Matrix<T,Dynamic,Dynamic> bRa(D_,D_);
   
  T dot = b.transpose()*a;
  ASSERT(fabs(dot) <=1.0, "a="<<a.transpose()<<" |.| "<<a.norm()
      <<" b="<<b.transpose()<<" |.| "<<b.norm()
      <<" -> "<<dot);
  dot = max(static_cast<T>(-1.0),min(static_cast<T>(1.0),dot));
//  cout << "dot="<<dot<<" | |"<<fabs(dot+1.)<<endl;
  if(fabs(dot -1.) < 1e-6)
  {
    // points are almost the same -> just put identity
    bRa =  Matrix<T,Dynamic,Dynamic>::Identity(D_,D_);
//    bRa(0,0) = cos(percentage*PI);
//    bRa(1,1) = cos(percentage*PI);
//    bRa(0,1) = -sin(percentage*PI);
//    bRa(1,0) = sin(percentage*PI);
  }else if(fabs(dot +1.) <1e-6) 
  {
    // direction does not matter since points are on opposing sides of sphere
    // -> pick one and rotate by percentage;
    bRa = -Matrix<T,Dynamic,Dynamic>::Identity(D_,D_);
    bRa(0,0) = cos(percentage*PI*0.5);
    bRa(1,1) = cos(percentage*PI*0.5);
    bRa(0,1) = -sin(percentage*PI*0.5);
    bRa(1,0) = sin(percentage*PI*0.5);
  }else{
    T alpha = acos(dot) * percentage;
//    cout << "alpha="<<alpha<<endl;

    Matrix<T,Dynamic,1> c(D_);
    c = a - b*dot;
    ASSERT(c.norm() >1e-5, "c="<<c.transpose()<<" |.| "<<c.norm());
    c /= c.norm();
    Matrix<T,Dynamic,Dynamic> A = b*c.transpose() - c*b.transpose();

    bRa = Matrix<T,Dynamic,Dynamic>::Identity(D_,D_) + sin(alpha)*A + 
      (cos(alpha)-1.)*(b*b.transpose() + c*c.transpose());
  }
  return bRa;
}

