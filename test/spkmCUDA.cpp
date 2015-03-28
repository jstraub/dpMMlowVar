
#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE spkm test
#include <boost/test/unit_test.hpp>

#include <boost/shared_ptr.hpp>

#include <stdint.h>

#include <jsCore/clDataGpu.hpp>
#include <dpMMlowVar/kmeansCUDA.hpp>

using namespace Eigen;
using namespace dplv;
using std::cout;
using std::endl;

BOOST_AUTO_TEST_CASE(spkm_test)
{
//  boost::mt19937 rndGen(91);
  
  uint32_t N=20;
  uint32_t D=3;
  uint32_t K=2;
  boost::shared_ptr<MatrixXd> spx(new MatrixXd(MatrixXd::Zero(D,N)));
  for(uint32_t i=0; i<N; ++i)
    if((2*i)/N == 0)
    {
      (*spx)(0,i) = 1.;
    }else{
      (*spx)(0,i) = -1.;
    }
  cout<<(*spx)<<endl;
//  sampleClustersOnSphere<double>(*spx, K);
  boost::shared_ptr<jsc::ClDataGpud> cld( new jsc::ClDataGpud(spx,K));

  uint32_t T=10;
  cout<<" -------------------- spkm ----------------------"<<endl;
  spkmCUDAd spkm(cld);
  cout<<spkm.centroids()<<endl;
  for(uint32_t t=0; t<T; ++t)
  {
    spkm.updateCenters();
    cout<<spkm.centroids()<<endl;
    spkm.updateLabels();
    cout<<spkm.z().transpose()<<" "
      <<spkm.avgIntraClusterDeviation()<<endl;
  }

  cout<<" ---------------- kmeans -------------------"<<endl;
  kmeansCUDAd kmeans(cld);
  cout<<kmeans.centroids()<<endl;
  for(uint32_t t=0; t<T; ++t)
  {
    kmeans.updateCenters();
    cout<<kmeans.centroids()<<endl;
    kmeans.updateLabels();
    cout<<kmeans.z().transpose()<<" "
      <<kmeans.avgIntraClusterDeviation()<<endl;
//    cout<<kmeans.centroids()<<endl;
  }
}
