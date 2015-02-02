#pragma once

#include <Eigen/Dense>
#include <clData.hpp>

using std::min;
using std::max;

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
  }else if(fabs(dot +1.) <1e-6) 
  {
    // direction does not matter since points are on opposing sides of sphere
    // -> pick one and rotate by percentage;
    bRa = -Matrix<T,Dynamic,Dynamic>::Identity(D_,D_);
    bRa(0,0) = cos(percentage*M_PI*0.5);
    bRa(1,1) = cos(percentage*M_PI*0.5);
    bRa(0,1) = -sin(percentage*M_PI*0.5);
    bRa(1,0) = sin(percentage*M_PI*0.5);
  }else{
    T alpha = acos(dot) * percentage;

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

template<typename T>
struct Spherical //: public DataSpace<T>
{
  class Cluster
  {
    protected:
    Matrix<T,Dynamic,1> centroid_;
    Matrix<T,Dynamic,1> xSum_;
    uint32_t N_;

    public:

    Cluster() : centroid_(0,1), xSum_(0,1), N_(0)
    {};

    Cluster(const Matrix<T,Dynamic,1>& x_i) : centroid_(x_i), xSum_(x_i), N_(1)
    {};

    Cluster(const Matrix<T,Dynamic,1>& xSum, uint32_t N) :
      centroid_(xSum/xSum.norm()), xSum_(xSum), N_(N)
    {};

    T dist (const Matrix<T,Dynamic,1>& x_i) const
    { return Spherical::dist(this->centroid_, x_i); };

    void computeSS(const Matrix<T,Dynamic,Dynamic>& x,  const VectorXu& z,
        const uint32_t k)
    {
      Spherical::computeSum(x,z,k,&N_);
      //TODO: cloud try to do sth more random here
      if(N_ == 0)
      {
        const uint32_t D = x.rows();
        xSum_ = Matrix<T,Dynamic,1>::Zero(D);
        xSum_(0) = 1.;
      }
    };

    void updateCenter()
    {
      centroid_ = xSum_/xSum_.norm();
    };

    void updateSS(const shared_ptr<ClData<T> >& cld, uint32_t k)
    {
      xSum_ = cld->xSum(k);
      N_ = cld->count(k);
    };

    void updateCenter(const shared_ptr<ClData<T> >& cld, uint32_t k)
    {
      updateSS(cld,k); 
      updateCenter();
    };

    void computeCenter(const Matrix<T,Dynamic,Dynamic>& x,  const VectorXu& z,
        const uint32_t k)
    {
      computeSS(x,z,k);
      updateCenter();
    };

    bool isInstantiated() const {return this->N_>0;};

    uint32_t N() const {return N_;};
    uint32_t& N(){return N_;};
    const Matrix<T,Dynamic,1>& centroid() const {return centroid_;};
    Matrix<T,Dynamic,1>& centroid() {return centroid_;};
    const Matrix<T,Dynamic,1>& xSum() const {return xSum_;};
  };


  class DependentCluster : public Cluster
  {
    protected:
    // variables
    T t_;
    T w_;
    // parameters
    T beta_;
    T lambda_;
    T Q_;
    Matrix<T,Dynamic,1> prevCentroid_;

    public:

    DependentCluster() : Cluster(), t_(0), w_(0), beta_(1), lambda_(1), Q_(1),
      prevCentroid_(centroid_)
    {};

    DependentCluster(const Matrix<T,Dynamic,1>& x_i) : Cluster(x_i), t_(0),
      w_(0), beta_(1), lambda_(1), Q_(1), prevCentroid_(centroid_)
    {};

    DependentCluster(const Matrix<T,Dynamic,1>& x_i, T beta, T lambda, T Q) :
      Cluster(x_i), t_(0), w_(0), beta_(beta), lambda_(lambda), Q_(Q), 
      prevCentroid_(centroid_)
    {};

    DependentCluster(const Matrix<T,Dynamic,1>& x_i, const DependentCluster& cl0) :
      Cluster(x_i), t_(0), w_(0), beta_(cl0.beta()), lambda_(cl0.lambda()),
      Q_(cl0.Q()), prevCentroid_(centroid_)
    {};

    DependentCluster(T beta, T lambda, T Q) :
      Cluster(), t_(0), w_(0), beta_(beta), lambda_(lambda), Q_(Q), 
      prevCentroid_(centroid_)
    {};

    DependentCluster(const DependentCluster& b) :
      Cluster(b.xSum(), b.N()), t_(b.t()), w_(b.w()), beta_(b.beta()),
      lambda_(b.lambda()), Q_(b.Q()), prevCentroid_(b.prevCentroid())
    {};

    DependentCluster* clone(){return new DependentCluster(*this);}

    bool isDead() const {return t_*Q_ < lambda_;};
    bool isNew() const {return t_ == 0;};

    void incAge() { ++ t_; };

    void print() const 
    {
      cout<<"cluster " <<"\tN="<<this->N_ <<"\tage="<<t_ <<"\tweight="
        <<w_ <<"\t dead? "<<this->isDead()
        <<"  center: "<<this->centroid().transpose()<<endl;
    };

    const Matrix<T,Dynamic,1>& prevCentroid() const {return prevCentroid_;};
    Matrix<T,Dynamic,1>& prevCentroid() {return prevCentroid_;};

    void nextTimeStep()
    {
      this->N_ = 0;
      this->prevCentroid_ = this->centroid_;
    };

    void updateWeight()
    {
      T phi, theta, eta;
      T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
              Spherical::dist(this->xSum_,this->centroid_)/this->xSum_.norm())));
      Spherical::solveProblem2(this->xSum_ , zeta, t_, w_, beta_, phi,theta,eta);
      w_ = w_ == 0.0? this->xSum_.norm() : w_ * cos(theta) + beta_*t_*cos(phi)
        + this->xSum_.norm()*cos(eta);
      t_ =  0;
    };

    void reInstantiate()
    {
      T phi, theta, eta;
      T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
              Spherical::dist(this->xSum_,this->prevCentroid_)/this->xSum_.norm())));
      Spherical::solveProblem2(this->xSum_ , zeta, t_, w_, beta_, phi,theta,eta);

      // rotate point from mean_k towards previous mean by angle eta?
      this->centroid_ = rotationFromAtoB<T>(this->xSum_/this->xSum_.norm(), 
          this->prevCentroid_, eta/(phi*t_+theta+eta)) * this->xSum_/this->xSum_.norm(); 
    };

    void reInstantiate(const Matrix<T,Dynamic,Dynamic>& x_i)
    {
      this->xSum_ = x_i; this->N_ = 1;
      reInstantiate();
    };

    T maxDist() const { return this->lambda_+1.;};
    T dist (const Matrix<T,Dynamic,1>& x_i) const
    {
      if(this->isInstantiated())
        return Spherical::dist(this->centroid_, x_i);
      else{
        T phi, theta, eta;
        T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
                Spherical::dist(x_i,this->prevCentroid_) )));
        cout<<"zeta="<<zeta;
        // apprixmation here for small angles -> same as on GPU
        Spherical::solveProblem2Approx(x_i, zeta, t_, w_, beta_, phi,theta,eta);
        cout<<" phi="<<phi<<" theta="<<theta<<" eta="<<eta<<" w_="<<w_
          <<" beta="<<beta<<" Q="<<Q_<<" t="<<t_<<endl;

        return w_*(cos(theta)-1.) + t_*beta_*(cos(phi)-1.) + Q_*t_
          + cos(eta); // no minus 1 here cancels with Z(beta) from the two other assignments
      }
    };

    T beta() const {return beta_;};
    T lambda() const {return lambda_;};
    T Q() const {return Q_;};
    T t() const {return t_;};
    T w() const {return w_;};

    uint32_t globalId; // id globally - only increasing id
  };

  static T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return a.transpose()*b; };

  static T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return acos(min(static_cast<T>(1.0),max(static_cast<T>(-1.0),(a.transpose()*b)(0)))); };

  static bool closer(const T a, const T b)
  { return a > b; };

  private:

  static void solveProblem1(T gamma, T age, const T beta, T& phi, T& theta); 
  static void solveProblem2(const Matrix<T,Dynamic,1>& xSum, T zeta, T age, T w,
      const T beta, T& phi, T& theta, T& eta); 

  static void solveProblem1Approx(T gamma, T age, const T beta, T& phi, T& theta); 
  static void solveProblem2Approx(const Matrix<T,Dynamic,1>& xSum, T zeta, T age, T w,
      const T beta, T& phi, T& theta, T& eta); 

  static Matrix<T,Dynamic,1> computeSum(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, const uint32_t k, uint32_t* N_k);
};

// ================================ impl ======================================

  template<typename T>                                                            
Matrix<T,Dynamic,1> Spherical<T>::computeSum(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, const uint32_t k, uint32_t* N_k)
{
  const uint32_t D = x.rows();
  const uint32_t N = x.cols();
  Matrix<T,Dynamic,1> xSum(D);
  xSum.setZero(D);
  if(N_k) *N_k = 0;
  for(uint32_t i=0; i<N; ++i)
    if(z(i) == k)
    {
      xSum += x.col(i); 
      if(N_k) (*N_k) ++;
    }
  return xSum;
};


template<class T>
void Spherical<T>::solveProblem1(T gamma, T age, const T beta, T& phi, T& theta)
{
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  gamma = T phi + theta
  // for phi and theta
  phi = 0.0; 

  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = sin(phi);
    T f = - gamma + age*phi + asin(beta*sinPhi);
    // mathematica
    T df = age + (beta*cos(phi))/sqrt(1.-beta*beta*sinPhi*sinPhi); 
    T dPhi = f/df;
    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta*sin(phi));
};


template<class T>
void Spherical<T>::solveProblem2(const Matrix<T,Dynamic,1>& xSum, T zeta, 
    T age, T w, const T beta, T& phi, T& theta, T& eta)
{
  // solves
  // w sin(theta) = beta sin(phi) = ||xSum||_2 sin(eta) 
  // eta + T phi + theta = zeta = acos(\mu0^T xSum/||xSum||_2)
  phi = 0.0;

  //  cout<<"w="<<w<<" age="<<age<<" zeta="<<zeta<<endl;

  T L2xSum = xSum.norm();
  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = sin(phi);
    T cosPhi = cos(phi);
    T f = - zeta + asin(beta/L2xSum *sinPhi) + age * phi + asin(beta/w *sinPhi);
    T df = age + (beta*cosPhi)/sqrt(L2xSum*L2xSum -
        beta*beta*sinPhi*sinPhi) + (beta*cosPhi)/sqrt(w*w -
        beta*beta*sinPhi*sinPhi); 

    T dPhi = f/df;

    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<"f="<<f<<" df="<<df<<" phi="<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta/w *sin(phi));
  eta = asin(beta/L2xSum *sin(phi));
};


template<class T>
void Spherical<T>::solveProblem1Approx(T gamma, T age, const T beta, T& phi, T& theta)
{
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  gamma = T phi + theta
  // for phi and theta
  phi = 0.0; 

  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = phi;
    T cosPhi = 1.;
    T f = - gamma + age*phi + asin(beta*sinPhi);
    // mathematica
    T df = age + (beta*cosPhi)/sqrt(1.-beta*beta*sinPhi*sinPhi); 
    T dPhi = f/df;
    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta*sin(phi));
};


template<class T>
void Spherical<T>::solveProblem2Approx(const Matrix<T,Dynamic,1>& xSum, T zeta, 
    T age, T w, const T beta, T& phi, T& theta, T& eta)
{
  // solves
  // w sin(theta) = beta sin(phi) = ||xSum||_2 sin(eta) 
  // eta + T phi + theta = zeta = acos(\mu0^T xSum/||xSum||_2)
  phi = 0.0;

  //  cout<<"w="<<w<<" age="<<age<<" zeta="<<zeta<<endl;

  T L2xSum = xSum.norm();
  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = phi;
    T cosPhi = 1.;
    T f = - zeta + asin(beta/L2xSum *sinPhi) + age * phi + asin(beta/w *sinPhi);
    T df = age + (beta*cosPhi)/sqrt(L2xSum*L2xSum -
        beta*beta*sinPhi*sinPhi) + (beta*cosPhi)/sqrt(w*w -
        beta*beta*sinPhi*sinPhi); 

    T dPhi = f/df;

    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<"f="<<f<<" df="<<df<<" phi="<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta/w *sin(phi));
  eta = asin(beta/L2xSum *sin(phi));
};

