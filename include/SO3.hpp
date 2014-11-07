#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using namespace Eigen;
using std::vector;
using std::cout;
using std::endl;

template<typename T>
class SO3
{
  public:
    static inline Matrix<T,Dynamic,1> vee(const Matrix<T,Dynamic,Dynamic>& W);
    static inline Matrix<T,Dynamic,Dynamic> invVee(const Matrix<T,Dynamic,1>& w);
    static inline Matrix<T,Dynamic,Dynamic> logMapW(const Matrix<T,Dynamic,Dynamic>& R);
    static inline Matrix<T,Dynamic,1> logMap(const Matrix<T,Dynamic,Dynamic>& R);
    static inline Matrix<T,Dynamic,Dynamic> expMap(const Matrix<T,Dynamic,1>& w);
    static inline Matrix<T,Dynamic,Dynamic> expMap(const Matrix<T,Dynamic,Dynamic>& W);
    static inline Matrix<T,Dynamic,Dynamic> meanRotation(const
        vector<Matrix<T,Dynamic,Dynamic> >& Rs, uint32_t Tmax=100);
    static inline Matrix<T,Dynamic,Dynamic> meanRotation(const
        vector<Matrix<T,Dynamic,Dynamic> >& Rs, Matrix<T,Dynamic,1> w, uint32_t
        Tmax=100);
};

// ----------------------------- impl ----------------------------------

template<typename T>
inline Matrix<T,Dynamic,1> SO3<T>::vee(const Matrix<T,Dynamic,Dynamic>& W)
{
  const Matrix<T,Dynamic,Dynamic> A = 0.5*(W - W.transpose());
  Matrix<T,Dynamic,1> w(3);
  w << A(2,1), A(0,2), A(1,0);
  return w;
};

template<typename T>
inline Matrix<T,Dynamic,Dynamic> SO3<T>::invVee(const Matrix<T,Dynamic,1>& w)
{
  Matrix<T,Dynamic,Dynamic> W = MatrixXf::Zero(3,3);
  W(2,1) = w(0);
  W(0,2) = w(1);
  W(1,0) = w(2);

  W(1,2) = -w(0);
  W(2,0) = -w(1);
  W(0,1) = -w(2);
  return W;
};

template<typename T>
inline Matrix<T,Dynamic,Dynamic> SO3<T>::logMapW(const Matrix<T,Dynamic,Dynamic>& R)
{
  assert(R.rows() ==3);
  assert(R.cols() ==3);
  const T theta = acos((R.trace()-1.)*0.5);
//  cout<<"theta="<<theta<<endl;
  T a = theta/(2.*sin(theta));
  if(a!=a) a = 0.0;
  Matrix<T,Dynamic,Dynamic> W = a*(R-R.transpose());
  return W;
};

template<typename T>
inline Matrix<T,Dynamic,1> SO3<T>::logMap(const Matrix<T,Dynamic,Dynamic>& R)
{
  return vee(logMapW(R));
};

template<typename T>
inline Matrix<T,Dynamic,Dynamic> SO3<T>::expMap(const Matrix<T,Dynamic,1>& w)
{
  assert(w.rows() ==w);
  const T theta = sqrt(w.array().square().matrix().sum());
//  cout<<"theta="<<theta<<endl;
  const Matrix<T,Dynamic,Dynamic> W = invVee(w);
  T a = sin(theta)/theta;
  if(a!=a) a = 0.0;
  T b = (1.-cos(theta))/(theta*theta);
  if(b!=b) b = 0.0;
  const Matrix<T,Dynamic,Dynamic> R = MatrixXf::Identity(3,3) + a * W + b * W*W;
//  cout<<"W"<<endl<<W<<endl;
//  cout<<"W*W"<<endl<<W*W<<endl;
//  cout<<"Rdet="<<R.determinant()<<endl;
  return R;
};

template<typename T>
inline Matrix<T,Dynamic,Dynamic> SO3<T>::expMap(const Matrix<T,Dynamic,Dynamic>& W)
{
  return expMap(invVee(W));
};

/* compute the mean rotation using karcher mean on SO3 */
template<typename T>
inline Matrix<T,Dynamic,Dynamic> SO3<T>::meanRotation(const vector<Matrix<T,Dynamic,Dynamic> >& Rs, uint32_t Tmax)
{
  Matrix<T,Dynamic,Dynamic> x(3,Rs.size());
  Matrix<T,Dynamic,Dynamic> muR = Rs[0];// arbitrarily 
  Matrix<T,Dynamic,1> xMean;
  for(uint32_t t=0; t<Tmax; ++t)
  {
    muR = 0.5*(muR-muR.transpose()); // symmetrize here 
    for(uint32_t i=0; i<Rs.size(); ++i)
      x.col(i) = logMap(muR.transpose()*Rs[i]);
    xMean = x.rowwise().sum()/x.cols();
    muR = expMap(xMean)*muR;
//    cout<<"@t"<<t<<": "<<xMean.transpose()<<endl;
//    cout<<x<<endl;
    if((xMean.array().abs()<1e-6).all()) break;
  }
  return muR;
}

/* compute the weighted mean rotation using karcher mean on SO3 */
template<typename T>
inline Matrix<T,Dynamic,Dynamic> SO3<T>::meanRotation(const vector<Matrix<T,Dynamic,Dynamic> >& Rs, Matrix<T,Dynamic,1> w, uint32_t Tmax)
{
  Matrix<T,Dynamic,Dynamic> x(3,Rs.size());
  Matrix<T,Dynamic,Dynamic> muR = Rs[0];// arbitrarily 
  Matrix<T,Dynamic,1> xMean;
  for(uint32_t t=0; t<Tmax; ++t)
  {
    muR = 0.5*(muR-muR.transpose()); // symmetrize here 
    for(uint32_t i=0; i<Rs.size(); ++i)
      x.col(i) = logMap(muR.transpose()*Rs[i])*w(i);
    xMean = x.rowwise().sum()/w.sum();
    muR = expMap(xMean)*muR;
//    cout<<"@t"<<t<<": "<<xMean.transpose()<<endl;
//    cout<<x<<endl;
    if((xMean.array().abs()<1e-6).all()) break;
  }
  return muR;
}

