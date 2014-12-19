#pragma once

#include <Eigen/Dense>


template<typename T>
struct Euclidean //: public DataSpace<T>
{
   
  static T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return (a-b).squaredNorm(); };

  static T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  { return (a-b).squaredNorm();};

  static bool closer(const T a, const T b) { return a<b; };

  static T distToUninstantiated(const Matrix<T,Dynamic,1>& x_i, const
      Matrix<T,Dynamic,1>& ps_k, const T t_k, const T w_k, const T tau, 
      const T Q)
  { return dist(x_i, ps_k) / (tau*t_k+1.+ 1.0/w_k) + Q*t_k; };

  static Matrix<T,Dynamic,Dynamic> computeCenters(const
      Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t K, 
      VectorXu& Ns);

  static Matrix<T,Dynamic,1> computeCenter(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, const uint32_t k, uint32_t* N_k);
};

//============================= impl ==========================================

template<typename T>                                                            
static Matrix<T,Dynamic,Dynamic> Euclidean<T>::computeCenters(const
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
static Matrix<T,Dynamic,1> Euclidean<T>::computeCenter(const Matrix<T,Dynamic,Dynamic>& x, 
    const VectorXu& z, const uint32_t k, uint32_t* N_k)
{
  const uint32_t D = x.rows();
  const uint32_t N = x.cols();
  *N_k = 0;
  Matrix<T,Dynamic,1> mean_k(D);
  mean_k.setZero(D);
  for(uint32_t i=0; i<N; ++i)
    if(z(i) == k)
    {
      mean_k += x.col(i); 
      (*N_k) ++;
    }
  if(*N_k > 0)
    return mean_k/(*N_k);
  else
    //TODO: cloud try to do sth more random here
    return x.col(k); //Matrix<T,Dynamic,1>::Zero(D,1);
};

