#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SO3 test
#include <boost/test/unit_test.hpp>

#include <boost/shared_ptr.hpp>

#include <stdint.h>

#include <dpMMlowVar/SO3.hpp>

using namespace Eigen;
using namespace dplv;
using std::cout;
using std::endl;


BOOST_AUTO_TEST_CASE(SO3_test)
{
  MatrixXf R = MatrixXf::Identity(3,3); 
  MatrixXf R2 = MatrixXf::Identity(3,3); 
  VectorXf w2(3); w2 << 0.00,0.1,0.0;
  R2 = SO3<float>::expMap(w2); 

  cout<<"R:"<<endl<< R << endl;
  cout<< SO3<float>::logMap(R).transpose() << endl;
  cout<<"R2:"<<endl<< R2 << endl;
  cout<< SO3<float>::logMap(R2).transpose() << endl;
  cout<< w2.transpose()<<endl;

  std::vector<MatrixXf> Rs;
  Rs.push_back(R);
  Rs.push_back(R2);
  MatrixXf muR = SO3<float>::meanRotation(Rs, 10);
  cout<<"meanR"<<endl<<muR<<endl;
  cout<< SO3<float>::logMap(muR).transpose() << endl;
  
}
