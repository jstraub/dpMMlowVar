#pragma once

#include <Eigen/Dense>

template<typename T>
struct DataSpace
{
  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) const = 0;
  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) = 0;
  virtual bool closer(const T a, const T b) = 0;
};

template<typename T>
struct Euclidean : public DataSpace<T>
{
   
  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b) const
  {
    return (a-b).norm();
  };
  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  {
    return (a-b).norm();
  };

  virtual bool closer(const T a, const T b)
  {
    return a<b;
  };
};

template<typename T>
struct Spherical : public DataSpace<T>
{
  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  {
    return (a-b).norm();
  };
  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
  {
    return (a-b).norm();
  };

  virtual bool closer(const T a, const T b)
  {
    return a<b;
  };
};


