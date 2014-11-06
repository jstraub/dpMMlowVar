#include "sphere.hpp"


// ---------------------------------- impl ------------------------------------

/* normal in tangent space around p rotate to the north pole
 * -> the third dimension will always be 0
 * -> return only first 2 dims
 */
template <typename T>
Matrix<T,Dynamic,Dynamic> Sphere<T>::Log_p_north(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& q) const
{
  return rotate_p2north(p,Log_p(p,q));
};

/* rotate points x in tangent plane around north pole down to p
*/
template <typename T>
Matrix<T,Dynamic,Dynamic> Sphere<T>::rotate_north2p(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& xNorth) const
{
  Matrix<T,Dynamic,Dynamic> northR = north_R_TpS2(p);
//  cout<<"northR"<<endl<<northR<<endl;
//  if(xNorth.cols() == D_-1)
//  {
//    Matrix<T,Dynamic,Dynamic> x(xNorth.rows(),D_);
//    x = xNorth * northR.transpose().topRows(D_-1);
//    return x;
//  }else 
  if (xNorth.rows() == D_-1){
    Matrix<T,Dynamic,Dynamic> x(D_,xNorth.cols());
    //augment the vector
    x.topRows(D_-1) = xNorth; 
    // push it up to north pole
    x(D_-1) = 0.;
    // rotate it 
    return northR.transpose() * x;
  }else{
    assert(false);
    return Matrix<T,Dynamic,Dynamic>::Zero(D_,1);
  }
}
/* rotate points x in tangent plane around p to north pole and
 * return 2D coordinates
 */
template <typename T>
Matrix<T,Dynamic,Dynamic> Sphere<T>::rotate_p2north(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& x) const
{
  assert(D_ == p.rows());
  Matrix<T,Dynamic,Dynamic> northR = north_R_TpS2(p);


//  cout<< northR<<endl;
//  if(x.cols() == D_)
//  {
//    //Matrix<T,Dynamic,Dynamic> xNorth(x.rows(),2);
//    assert(fabs((x.row(0)*northR.transpose()).bottomRows<1>().sum()) < 1e-6);
//    return (x * northR.transpose()).leftCols(D_-1);
  if (x.rows() == D_){
    Matrix<T,Dynamic,Dynamic> xhat = (northR * x);
    for(uint32_t i=0; i<xhat.cols(); ++i)
      if(fabs(xhat.col(i)(D_-1))>1e-5) 
      {
        // projection to zero last entry
        xhat.col(i) -= (xhat.col(i).transpose()*north_).sum()*north_;
      }
//    cout<<"x           : "<< x.col(0).transpose()
//      <<" | | "<< x.col(0).norm()<<endl;
//    cout<<northR<<endl;
//    cout<<"x@ northPole: "<<xhat.col(0).transpose()<<endl;
    assert((xhat.bottomRows(1).array() < 1e-3).all());
    return xhat.topRows(D_-1);
  }else{
    assert(false);
    return Matrix<T,Dynamic,Dynamic>::Zero(D_,1);
  }
}

template <typename T>
void Sphere<T>::rotate_p2north(const Matrix<T,Dynamic,1>& p, 
   const Matrix<T,Dynamic,Dynamic>& x_p, Matrix<T,Dynamic,Dynamic>& xNorth,
   const VectorXu& z, uint32_t k) const
{
  assert(D_ == p.rows());
  Matrix<T,Dynamic,Dynamic> northR = north_R_TpS2(p);
//  cout<<northR<<endl;

  if (x_p.rows() == D_){
#pragma omp parallel for
    for(uint32_t i=0; i<x_p.cols(); ++i)
      if(z(i) == k)
      {
        ASSERT(fabs(x_p.col(i).transpose()*p)<1e-5,
          x_p.col(i).transpose()*p);

         Matrix<T,Dynamic,1>  xNorth_i = northR*x_p.col(i);
//        cout<< x_p.col(i).transpose()<<endl;
//        cout<< xNorth_i.transpose()<<endl;
//        cout<< xNorth_i.topRows(D_-1).transpose()<<endl;
//        cout<< xNorth.col(i).transpose()<<endl;
        ASSERT(fabs(xNorth_i(D_-1))<1e-5, xNorth_i.transpose());
        ASSERT(fabs(xNorth_i(D_-1))<1e-5, xNorth_i.transpose() 
          << " dot prod of x and p: "<<x_p.col(i).transpose()*p);
//        if(fabs(xNorth_i(D_-1))>1e-5) 
//        {
//          // projection to zero last entry
//          xNorth_i -= (xNorth_i.transpose()*north_).sum()*north_;
//        }
//        cout<< xNorth_i.transpose()<<endl;
        xNorth.col(i) = xNorth_i.topRows(D_-1);
      }
//    assert((xhat.bottomRows(1).array() < 1e-3).all());
//    return xhat.topRows(D_-1);
  }else{
    assert(false);
//    return Matrix<T,Dynamic,Dynamic>::Zero(D_,1);
  }
}

template <typename T>
void Sphere<T>::rotate_p2north( const Matrix<T,Dynamic,Dynamic>& ps, 
    const Matrix<T,Dynamic,Dynamic>& x, Matrix<T,Dynamic,Dynamic>& xNorth,
    const VectorXu& z, uint32_t K) const
{
  assert(D_ == ps.rows());
  assert(D_ == x.rows());
  assert(D_-1 == xNorth.rows());
  assert(K == ps.cols());

  uint32_t N = x.cols();

  std::vector<Matrix<T,Dynamic,Dynamic> > northRs(K);
  for(uint32_t k=0; k<K; ++k)
  {
    ASSERT((fabs(ps.col(k).norm()-1.)<1e-7),
        ps.col(k).transpose()<<" |.| "<<ps.col(k).norm());
    northRs[k] = this->north_R_TpS2(ps.col(k));
  }
//  cout<< northR<<endl;
//  if(x.cols() == D_)
//  {
//    //Matrix<T,Dynamic,Dynamic> xNorth(x.rows(),2);
//    assert(fabs((x.row(0)*northR.transpose()).bottomRows<1>().sum()) < 1e-6);
//    return (x * northR.transpose()).leftCols(D_-1);
  if (x.rows() == D_){
#pragma omp parallel for
    for(uint32_t i=0; i<N; ++i)
    {
      ASSERT(fabs(x.col(i).transpose()*ps.col(z(i)))<1e-6,
          x.col(i).transpose()*ps.col(z(i)));

      Matrix<T,Dynamic,1>  xNorth_i = northRs[z(i)] * x.col(i); 

      ASSERT(fabs(xNorth_i(D_-1))<1e-6, xNorth_i.transpose() 
          << " dot prod of x and p: "<<x.col(i).transpose()*ps.col(z(i)));

//      if(fabs(xNorth_i(D_-1))>1e-5) 
//      {
////        cout<<xNorth_i.transpose()<<endl;
////        assert(xNorth_i(D_-1) < 1e-5);
//        // projection to zero last entry
//        xNorth_i -= (xNorth_i.transpose()*north_).sum()*north_;
//      }
//      assert(xNorth_i(D_-1) < 1e-5);
      xNorth.col(i) = xNorth_i.topRows(D_-1);
    }
//    cout<<"x           : "<< x.col(0).transpose()
//      <<" | | "<< x.col(0).norm()<<endl;
//    cout<<northR<<endl;
//    cout<<"x@ northPole: "<<xhat.col(0).transpose()<<endl;
  }else{
    assert(false);
  }
}

/* compute rotation from TpS^2 to north pole on sphere
*/
template <typename T>
Matrix<T,Dynamic,Dynamic> Sphere<T>::north_R_TpS2(const Matrix<T,Dynamic,1>& p)
  const
{
  assert(D_ == p.rows()); 
  Matrix<T,Dynamic,Dynamic> northRtpS(D_,D_);  
//  cout<<"north="<<north_.transpose()<<endl;
  return rotationFromAtoB<T>(p,north_);
}

template <typename T>
T Sphere<T>::invSincDot(T dot) const
{
  // 2nd order taylor expansions for the limit cases obtained via mathematica
  if(static_cast<T>(MIN_DOT) < dot && dot < static_cast<T>(MAX_DOT))
    return acos(dot)/sqrt(1.-dot*dot);
  else if(dot <= static_cast<T>(MIN_DOT))
    return PI/(sqrt(2.)*sqrt(dot+1.)) -1. + PI*sqrt(dot+1.)/(4.*sqrt(2.))
      -(dot+1.)/3. + 3.*PI*(dot+1.)*sqrt(dot+1.)/(32.*sqrt(2.)) 
      - 2./15.*(dot+1.)*(dot+1.);
  else //if(dot >= static_cast<T>(MAX_DOT))
    return 1. - (dot-1)/3. + 2./5.*(dot-1.)*(dot-1.);
}

template <typename T>
Matrix<T,Dynamic,Dynamic> Sphere<T>::Log_p(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& q) const
{
  assert(D_ == p.rows());
  assert(q.rows() == D_);
  Matrix<T,Dynamic,Dynamic> x(q.rows(),q.cols());
#pragma omp parallel for
  for (uint32_t i=0; i<q.cols(); ++i)
  {
    x.col(i) =  Log_p_single(p,q.col(i));
//    Log_p(p,q.col(i),x.col(i));
//    T dot = max(static_cast<T>(-1.0),min(static_cast<T>(1.0),p.dot(q.col(i))));
//    x.col(i) = (q.col(i)-p*dot)*invSincDot(dot);
//    assert(x.col(i).norm() < PI);
//#ifndef NDEBUG
//    if(x.col(i).norm() >= PI && x.col(i).norm() < PI+1.)
//    {
//      cout<<"renormalizing"<<endl;
//      x.col(i) = (x.col(i)/x.col(i).norm())*(PI-1e-6);
//      cout<<x.col(i).transpose()<<" | |"<<x.col(i).norm()<<endl;
//    }
//    if(x.col(i).norm() >= PI)
//    {
//      cout<<p.transpose()<<" | |"<<p.norm()<<endl;
//      cout<<q.col(i).transpose()<<" | |"<<q.col(i).norm()<<endl;
//      cout<<x.col(i).transpose()<<" | |"<<x.col(i).norm()<<endl;
//      cout<<invSincDot(dot)<<" "<<dot<<endl;
//    }
//#endif
  }
  return x;
}

template <typename T>
Matrix<T,Dynamic,1> Sphere<T>::Log_p_single(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,1>& q) const
{
  assert(D_ == p.rows());
  assert(q.rows() == D_);

  ASSERT((fabs(p.norm()-1.)<1e-6),p.transpose()<<" |.| "<<p.norm()
      <<" "<<fabs(p.norm()-1.));
  ASSERT((fabs(q.norm()-1.)<1e-6),q.transpose()<<" |.| "<<q.norm()
      <<" "<<fabs(q.norm()-1.));

  T dot = max(static_cast<T>(-1.0),min(static_cast<T>(1.0),p.dot(q)));
#ifdef NDEBUG
  return (q-p*dot)*invSincDot(dot);
#else
  Matrix<T,Dynamic,1> x = (q-p*dot)*invSincDot(dot);
//  ASSERT((x.norm()<PI+1e-6),x.transpose()<<" |.| "<<x.norm());
  if(x.norm() >= PI && x.norm() < PI+1.)
  {
    cout<<"renormalizing"<<endl;
    x = (x/x.norm())*(PI-1e-6);
    cout<<x.transpose()<<" | |"<<x.norm()<<endl;
  }
  if(x.norm() >= PI)
  {
    cout<<p.transpose()<<" | |"<<p.norm()<<endl;
    cout<<q.transpose()<<" | |"<<q.norm()<<endl;
    cout<<x.transpose()<<" | |"<<x.norm()<<endl;
    cout<<invSincDot(dot)<<" "<<dot<<endl;
  }
  return x;
#endif
}

template <typename T>
Matrix<T,Dynamic,Dynamic> Sphere<T>::Log_p(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& q, const Matrix<T,Dynamic,1>& w) const
{
  assert(D_ == p.rows());
  assert(q.rows() == D_);
  Matrix<T,Dynamic,Dynamic> x(q.rows(),q.cols());
#pragma omp parallel for
  for (uint32_t i=0; i<q.cols(); ++i)
    if(w(i) >0.0)
    {
     x.col(i) =  Log_p_single(p,q.col(i));
//      T dot = max(static_cast<T>(-1.0),min(static_cast<T>(1.0),p.dot(q.col(i))));
//      x.col(i) = (q.col(i)-p*dot)*invSincDot(dot);
//      assert(x.col(i).norm() < PI);
//#ifndef NDEBUG
//      if(x.col(i).norm() >= PI && x.col(i).norm() < PI+1.)
//      {
//        cout<<"renormalizing"<<endl;
//        x.col(i) = (x.col(i)/x.col(i).norm())*(PI-1e-6);
//        cout<<x.col(i).transpose()<<" | |"<<x.col(i).norm()<<endl;
//      }
//      if(x.col(i).norm() >= PI)
//      {
//        cout<<p.transpose()<<" | |"<<p.norm()<<endl;
//        cout<<q.col(i).transpose()<<" | |"<<q.col(i).norm()<<endl;
//        cout<<x.col(i).transpose()<<" | |"<<x.col(i).norm()<<endl;
//        cout<<invSincDot(dot)<<" "<<dot<<endl;
//      }
//#endif
    }
  return x;
}

template <typename T>
void Sphere<T>::Log_p(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& q, const VectorXu& z, uint32_t k,
    Matrix<T,Dynamic,Dynamic>& x, uint32_t zDivider) const
{
  assert(D_ == p.rows());
  assert(q.rows() == D_);
  assert(x.rows() == q.rows());
  assert(x.cols() == q.cols());

#pragma omp parallel for
  for (uint32_t i=0; i<q.cols(); ++i)
      if(z(i)/zDivider ==k)
    {
      x.col(i) =  Log_p_single(p,q.col(i));
    }
}

template <typename T>
void Sphere<T>::Log_ps(const Matrix<T,Dynamic,Dynamic>& p, 
    const Matrix<T,Dynamic,Dynamic>& q, const VectorXu& z,
    Matrix<T,Dynamic,Dynamic>& x) const
{
  assert(D_ == p.rows());
  assert(q.rows() == D_);
  assert(x.rows() == q.rows());
  assert(x.cols() == q.cols());

#pragma omp parallel for
  for (uint32_t i=0; i<q.cols(); ++i)
    {
     x.col(i) =  Log_p_single(p.col(z(i)),q.col(i));
//     Log_p(p.col(z(i)),q.col(i), x.col(i));
//      cout<<x.col(i)<<endl;
//      uint32_t k = z(i);
//      T dot = max(static_cast<T>(-1.0),min(static_cast<T>(1.0),
//            p.col(k).dot(q.col(i))));
//      x.col(i) = (q.col(i) - p.col(k)*dot)*invSincDot(dot);
//      assert(x.col(i).norm() < PI);
//#ifndef NDEBUG
//      if(x.col(i).norm() >= PI && x.col(i).norm() < PI+1.)
//      {
//        cout<<"renormalizing"<<endl;
//        x.col(i) = (x.col(i)/x.col(i).norm())*(PI-1e-6);
//        cout<<x.col(i).transpose()<<" | |"<<x.col(i).norm()<<endl;
//      }
//      if(x.col(i).norm() >= PI)
//      {
//        cout<<p.col(k).transpose()<<" | |"<<p.col(k).norm()<<endl;
//        cout<<q.col(i).transpose()<<" | |"<<q.col(i).norm()<<endl;
//        cout<<x.col(i).transpose()<<" | |"<<x.col(i).norm()<<endl;
//        cout<<invSincDot(dot)<<" "<<dot<<endl;
//      }
//#endif
    }
}

template <typename T>
Matrix<T,Dynamic,1> Sphere<T>::Exp_p_single(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,1>& x) const
{
  assert(p.cols() ==1);
  assert(p.rows() == x.rows());
  Matrix<T,Dynamic,1> q(x.rows());

  ASSERT((fabs(p.norm()-1.)<1e-6),p.transpose()<<" |.| "<<p.norm());
  ASSERT((x.norm() < PI+1e-6),x.transpose()<<" |.| "<<x.norm());

  T theta_i = x.norm();
  //cout<<"theta "<<theta_i<<endl;
  if (fabs(theta_i) < 0.05)
  {
    q = p*cos(theta_i) + x*(1.-theta_i*theta_i/6.); // + O(x^4)
    //cout<<(1.-theta_i*theta_i/6.)<<" vs "<<(sin(theta_i)/theta_i)<<endl;
  }else{
    q = p*cos(theta_i) + x*(sin(theta_i)/theta_i);
  }
#ifndef NDEBUG
  ASSERT(fabs(q.norm()-1.0)<1e-6,q.transpose()<<" |.| "<<q.norm());
  //cout<<q.col(i).transpose()<<endl;
//  if (fabs(q.norm()-1.0)>1e-5)
//  {
//    cout<<"p "<<p.transpose()<<" ||="<<p.norm()<<endl;
//    cout<<"x "<<x.transpose()<<" ||="<<x.norm()<<endl;
//    cout<<"q "<<q.transpose()<<" ||="<<q.norm()<<endl;
//    cout<<"theta "<<theta_i<<endl;
//  }
//  assert(fabs(q.norm()-1.0)<1e-1);
//  assert(((x.transpose()*p).array().abs()-1. < 1.e-5).all());
#endif
  q /= q.norm();

  return q;
}

template <typename T>
Matrix<T,Dynamic,Dynamic> Sphere<T>::Exp_p(const Matrix<T,Dynamic,1>& p, 
    const Matrix<T,Dynamic,Dynamic>& x) const
{
  assert(p.cols() ==1);
  Matrix<T,Dynamic,Dynamic> q(x.rows(),x.cols());

#pragma omp parallel for
  for (uint32_t i=0; i<x.cols(); ++i)
  {
    q.col(i) = Exp_p_single(p,x.col(i));
  }
  return q;
}

template <typename T>
Matrix<T,Dynamic,1> Sphere<T>::sampleUnif(boost::mt19937* pRndGen)
{
//  Matrix<T,Dynamic,Dynamic> Sigma(Dd,Dd);
//  Sigma.setZero(Dd,Dd);
//  Sigma.diagonal() = Matrix<T,Dynamic,1>::Ones(Dd);
//  Matrix<T,Dynamic,1> mu(Dd);
//  mu.setZero(Dd,1);
//  cout<<Sigma<<endl;
//  cout<<mu<<endl;
  Normal<T> normal(D_,pRndGen);
  Matrix<T,Dynamic,1> q = normal.sample();
  return q/q.norm();
};



template class Sphere<double>;
template class Sphere<float>;
