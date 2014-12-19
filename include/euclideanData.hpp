#pragma once

#include <Eigen/Dense>


template<typename T>
struct Euclidean //: public DataSpace<T>
{

//  template <typename T>
  class Cluster
  {
    Matrix<T,Dynamic,1> centroid_;
    Matrix<T,Dynamic,1> xSum_;
    T N_;

    public:
    T dist (const Matrix<T,Dynamic,1>& x_i) const
    { return dist(this->centroid_, x_i); };
  };

//  template <typename T>
  class DependentCluster : public Cluster
  {
    // variables
    T t_;
    T w_;
    // parameters
    T tau_;
    T lambda_;
    T Q_;

    public:
    bool isDead() const {return t_*Q_ > lambda_;};
    bool isInstantiated() const {return this->N_>0;};

    void updateWeight(const Matrix<T,Dynamic,1>& xSum, const uint32_t N_k)
    {w_ =  1./(1./w_ + t_*tau_) + N_k;};

    Matrix<T,Dynamic,1> reInstantiate(const Matrix<T,Dynamic,1>& xSum)
    { const T gamma = 1.0/(1.0/w_ + t_*tau_);
     this->centroid_ = (this->centroid_ * gamma + xSum)/(gamma+1.);};

    T dist (const Matrix<T,Dynamic,1>& x_i) const
    {
      if(isInstantiated())
        return dist(this->centroid_, x_i);
      else{
        return dist(this->centroid_,x_i) / (tau_*t_+1.+ 1.0/ w_) + Q_*t_; 
      }
    };
  };
   
  static T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return (a-b).squaredNorm(); };

  static T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return (a-b).squaredNorm();};

  static bool closer(const T a, const T b) { return a<b; };

  static Matrix<T,Dynamic,1> computeSum(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, const uint32_t k, uint32_t* N_k);

  static Matrix<T,Dynamic,Dynamic> computeCenters(const
      Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t K, 
      VectorXu& Ns);

  static Matrix<T,Dynamic,1> computeCenter(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, const uint32_t k, uint32_t* N_k);

  // TODO deprecate soon and use cluster classes instead
  static T distToUninstantiated(const Matrix<T,Dynamic,1>& x_i, const
      Matrix<T,Dynamic,1>& ps_k, const T t_k, const T w_k, const T tau, 
      const T Q)
  { return dist(x_i, ps_k) / (tau*t_k+1.+ 1.0/w_k) + Q*t_k; };

  static bool clusterIsDead(const T t_k, const T lambda, const T Q)
  { return t_k*Q > lambda;};

  static Matrix<T,Dynamic,1> reInstantiatedOldCluster(const
      Matrix<T,Dynamic,1>& xSum, const T N_k, const Matrix<T,Dynamic,1>& ps_k, const T t_k,
      const T w_k, const T tau);
  
  static T updateWeight(const Matrix<T,Dynamic,1>& xSum, const uint32_t N_k,
      const Matrix<T,Dynamic,1>& ps_k, const T t_k, const T w_k, const T tau)
  {return 1./(1./w_k + t_k*tau) + N_k;};

};

//============================= impl ==========================================
//
  template<typename T>                                                            
Matrix<T,Dynamic,1> Euclidean<T>::computeSum(const Matrix<T,Dynamic,Dynamic>& x, 
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

template<typename T>                                                            
Matrix<T,Dynamic,Dynamic> Euclidean<T>::computeCenters(const
    Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t K, 
    VectorXu& Ns)
{
  const uint32_t D = x.rows();
  Matrix<T,Dynamic,Dynamic> centroids(D,K);
#pragma omp parallel for 
  for(uint32_t k=0; k<K; ++k)
    centroids.col(k) = computeCenter(x,z,k,&Ns(k));
  return centroids;
};

template<typename T>                                                            
Matrix<T,Dynamic,1> Euclidean<T>::computeCenter(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, const uint32_t k, uint32_t* N_k)
{
  const uint32_t D = x.rows();
  const uint32_t N = x.cols();
  if(N_k) *N_k = 0;
  Matrix<T,Dynamic,1> mean_k(D);
  mean_k.setZero(D);
  for(uint32_t i=0; i<N; ++i)
    if(z(i) == k)
    {
      mean_k += x.col(i); 
      if(N_k) (*N_k) ++;
    }
  if(!N_k)
    return mean_k; // TODO anoygin
  cout<<mean_k.transpose()<<" N="<<(*N_k)<<endl;
  if(*N_k > 0)
  {
    cout<<"centroid inside "<<endl<<(mean_k/(*N_k))<<endl;
    return mean_k/(*N_k);
  }
  else
    //TODO: cloud try to do sth more random here
    return x.col(k); //Matrix<T,Dynamic,1>::Zero(D,1);
};

template<typename T>                                                            
Matrix<T,Dynamic,1> Euclidean<T>::reInstantiatedOldCluster(const
    Matrix<T,Dynamic,1>& xSum, const T N_k, const Matrix<T,Dynamic,1>& ps_k, const T t_k, const
    T w_k, const T tau)
{
  const T gamma = 1.0/(1.0/w_k + t_k*tau);
  return (ps_k*gamma + xSum)/(gamma+N_k);
};
