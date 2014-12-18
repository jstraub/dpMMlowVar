#pragma once

#include <Eigen/Dense>

//template<typename T>
//struct DataSpace
//{
//  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) const = 0;
//  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) = 0;
//  virtual bool closer(const T a, const T b) = 0;
//};

template<typename T>
struct Euclidean //: public DataSpace<T>
{
   
  static T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  {
    return (a-b).squaredNorm();
  };
  static T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  {
    return (a-b).squaredNorm();
  };

  static bool closer(const T a, const T b)
  {
    return a<b;
  };

  static Matrix<T,Dynamic,Dynamic> computeCenters(const
      Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t K, 
      VectorXu& Ns)
  {
    const uint32_t D = x.rows();
    Matrix<T,Dynamic,Dynamic> centroids(D,K);
#pragma omp parallel for 
    for(uint32_t k=0; k<K; ++k)
      centroids.col(k) = computeCenter(x,z,k,&Ns(k));
    return centroids;
  }

  static Matrix<T,Dynamic,1> computeCenter(const Matrix<T,Dynamic,Dynamic>& x, 
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
};

template<typename T>
struct Spherical //: public DataSpace<T>
{
  static T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  {
    return a.transpose()*b;
  };
  static T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  {
    return acos(min(1.0,max(-1.0,(a.transpose()*b)(0))));
  };

  static bool closer(const T a, const T b)
  {
    return a > b;
  };

  static Matrix<T,Dynamic,Dynamic> computeSums(const
      Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t K)
  {
    const uint32_t D = x.rows();
    Matrix<T,Dynamic,Dynamic> xSums(D,K);
#pragma omp parallel for 
    for(uint32_t k=0; k<K; ++k)
      xSums.col(k) = computeSum(x,z,k,NULL);
    return xSums;
  }

  static Matrix<T,Dynamic,1> computeSum(const Matrix<T,Dynamic,Dynamic>& x, 
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

  static Matrix<T,Dynamic,Dynamic> computeCenters(const
      Matrix<T,Dynamic,Dynamic>& x, const VectorXu& z, const uint32_t K, 
      VectorXu& Ns)
  {
    const uint32_t D = x.rows();
    Matrix<T,Dynamic,Dynamic> centroids(D,K);
#pragma omp parallel for 
    for(uint32_t k=0; k<K; ++k)
      centroids.col(k) = computeCenter(x,z,k,&Ns(k));
    return centroids;
  }

  static Matrix<T,Dynamic,1> computeCenter(const Matrix<T,Dynamic,Dynamic>& x, 
      const VectorXu& z, const uint32_t k, uint32_t* N_k)
  {
    const uint32_t D = x.rows();
    Matrix<T,Dynamic,1> mean_k = computeSum(x,z,k,N_k);
    if(*N_k > 0)
      return mean_k/mean_k.norm();
    else
    {
      mean_k = Matrix<T,Dynamic,1>::Zero(D);
      mean_k(0) = 1.;
      return mean_k;
    }
  };

};


