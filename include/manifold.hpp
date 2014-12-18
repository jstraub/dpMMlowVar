#pragma once

#include <Eigen/Dense>

template<typename T>
class DataSpace
{
  virtual T dist() = 0;
  virtual T closer() = 0;
};

template<typename T>
class Euclidean : public DataSpace<T>
{
   
  T dist();
  T closer();
};

template<typename T>
class Spherical : public DataSpace<T>
{

  T dist();
  T closer();
};


