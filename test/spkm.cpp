
#include <iostream>
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE spkm test
#include <boost/test/unit_test.hpp>

#include <boost/shared_ptr.hpp>

#include <stdint.h>

#include "kmeans.hpp"
#include "sphericalKMeans.hpp"
#include "normalSphere.hpp"
#include "dpvMFmeans.hpp"
#include "ddpvMFmeans.hpp"
#include "ddpmeans.hpp"
#include "karcherMean.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

BOOST_AUTO_TEST_CASE(spkm_test)
{
  boost::mt19937 rndGen(91);
  
  uint32_t N=20;
  uint32_t D=3;
  uint32_t K=2;
  boost::shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  sampleClustersOnSphere<double>(*spx, K);


  uint32_t T=10;
  cout<<" -------------------- spkm ----------------------"<<endl;
  SphericalKMeans<double> spkm(spx,K,&rndGen);
  for(uint32_t t=0; t<T; ++t)
  {
    spkm.updateCenters();
    spkm.updateLabels();
    cout<<spkm.z().transpose()<<" "<<spkm.avgIntraClusterDeviation()<<endl;
//    cout<<spkm.centroids()<<endl;
  }
  MatrixXd deviates;
  MatrixXu inds = spkm.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;

  cout<<" ---------------- spkm_karch -------------------"<<endl;
  boost::mt19937 rndGen2(91);
  SphericalKMeansKarcher<double> spkmKarch(spx,K,&rndGen2);
 
  for(uint32_t t=0; t<T; ++t)
  {
    spkmKarch.updateCenters();
    spkmKarch.updateLabels();
    cout<<spkmKarch.z().transpose()<<" "
      <<spkmKarch.avgIntraClusterDeviation()<<endl;
//    cout<<spkmKarch.centroids()<<endl;
  }
  inds = spkmKarch.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;

  cout<<" ---------------- kmeans -------------------"<<endl;
  boost::mt19937 rndGen3(91);
  KMeans<double> kmeans(spx,K,&rndGen3);
  for(uint32_t t=0; t<T; ++t)
  {
    kmeans.updateCenters();
    kmeans.updateLabels();
    cout<<kmeans.z().transpose()<<" "
      <<kmeans.avgIntraClusterDeviation()<<endl;
//    cout<<kmeans.centroids()<<endl;
  }
  inds = kmeans.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;

  double lambda = - 0.9; //cos(15.0*M_PI/180.0);
  cout<<" -------------------- DpvMF means ----------------------"<<endl;
  DPvMFMeans<double> dpvmfmeans(spx,K,lambda,&rndGen);
  for(uint32_t t=0; t<T; ++t)
  {
    dpvmfmeans.updateCenters();
    cout<<dpvmfmeans.z().transpose()<<" "
      <<dpvmfmeans.avgIntraClusterDeviation()<<endl;
    dpvmfmeans.updateLabels();
    cout<<dpvmfmeans.z().transpose()<<" "
      <<dpvmfmeans.avgIntraClusterDeviation()<<endl;
//    cout<<spkm.centroids()<<endl;
  }
  inds = dpvmfmeans.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;

  lambda = cos(15.0*M_PI/180.0);
  cout<<" -------------------- DP-means ----------------------"<<endl;
  DPMeans<double> dpmeans(spx,K,lambda,&rndGen);
  for(uint32_t t=0; t<T; ++t)
  {
    dpmeans.updateCenters();
    cout<<dpmeans.z().transpose()<<" "
      <<dpmeans.avgIntraClusterDeviation()<<endl;
    dpmeans.updateLabels();
    cout<<dpmeans.z().transpose()<<" "
      <<dpmeans.avgIntraClusterDeviation()<<endl;
//    cout<<spkm.centroids()<<endl;
  }
  inds = dpmeans.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;

//  if(false)
//  {
    lambda = cos(15.0*M_PI/180.0);
    double Q = 5.0*M_PI/180.0;
    double tau = 5.0*M_PI/180.0;
    cout<<" -------------------- DDP-means ----------------------"<<endl;
    DDPMeans<double> ddpmeans(spx,lambda,Q,tau,&rndGen);

    for(uint32_t t=0; t<10; ++t)
    {
      cout<<" -- t = "<<t<<endl;
      if(t<4)
        for (uint32_t i=0; i<N; ++i)
          spx->col(i) = spx->col(i) + VectorXd::Ones(D)*0.1;
      else if(t==7)
        for (uint32_t i=0; i<N/2; ++i)
          spx->col(i+N/2) = spx->col(i); // single cluster from now on
      else if(t==4)
      {
        boost::shared_ptr<MatrixXd> spx3(new MatrixXd(D,N*2));
        //      boost::shared_ptr<MatrixXd> spxTmp(new MatrixXd(D,N));
        //      sampleClustersOnSphere<double>(*spxTmp, 1);
        spx3->leftCols(N) = *spx;
        spx3->rightCols(N) = MatrixXd::Zero(D,N);
        spx = spx3;
      }

      ddpmeans.nextTimeStep(spx); // feed in new data (here just the same
      for(uint32_t i=0; i<10; ++i)
      { // run clustering till "converence"
        ddpmeans.updateLabels();
        //      cout<<ddpmeans.z().transpose()<<" "
        //        <<ddpmeans.avgIntraClusterDeviation()<<endl;
        ddpmeans.updateCenters();
        cout<<ddpmeans.z().transpose()<<" "
          <<ddpmeans.avgIntraClusterDeviation()<<endl;
        //    cout<<spkm.centroids()<<endl;
      }
      ddpmeans.updateState(); // update the state internally
    }

  inds = ddpmeans.mostLikelyInds(10,deviates);
  cout<<"most likely indices"<<endl;
  cout<<inds<<endl;
  return;
//  }

  cout<<spx->transpose()<<endl;

  lambda = -0.5; //cos(15.0*M_PI/180.0);
  double beta = 5.0*M_PI/180.0;
  double w = 5.0*M_PI/180.0;
  cout<<" -------------------- DDP-vMF-means ----------------------"<<endl;
  DDPvMFMeans<double> ddpvmfmeans(spx,lambda,beta,w,&rndGen);

  double dAng = 5.0*M_PI/180.0;
  MatrixXd dR = MatrixXd::Zero(3,3);
  dR << cos(dAng), sin(dAng), 0,
       -sin(dAng), cos(dAng), 0,
       0         , 0        , 1;

  MatrixXd means = spkm.centroids();

  for(uint32_t t=0; t<10; ++t)
  {
    cout<<" -- t = "<<t<<endl;

    if(t==7)
    {
      boost::shared_ptr<MatrixXd> spx3(new MatrixXd(D,N*2));
      //      boost::shared_ptr<MatrixXd> spxTmp(new MatrixXd(D,N));
      //      sampleClustersOnSphere<double>(*spxTmp, 1);
      spx3->leftCols(N) = *spx;
      spx3->rightCols(N) = - (*spx);
      spx = spx3;
    }

    if(t>=3)
    {
      *spx = dR * (*spx);

      means = dR*means;
      cout<<"new means:"<<endl
        <<means.transpose()<<endl
        <<" ----------------------------- "<<endl;
    }


    ddpvmfmeans.nextTimeStep(spx); // feed in new data (here just the same
    for(uint32_t i=0; i<20; ++i)
    { // run clustering till "converence"
//      cout<<"========================= label udaptes ========================="<<endl;
      ddpvmfmeans.updateLabels();
//      cout<<ddpvmfmeans.z().transpose()<<" "
//        <<ddpvmfmeans.avgIntraClusterDeviation()<<endl;
//      cout<<"========================= center udaptes ========================="<<endl;
      ddpvmfmeans.updateCenters();
      cout<<ddpvmfmeans.z().transpose()<<" "
        <<ddpvmfmeans.avgIntraClusterDeviation()<<endl;
      //    cout<<spkm.centroids()<<endl;
    }
    ddpvmfmeans.updateState(); // update the state internally
  }

//  inds = ddpvmfmeans.mostLikelyInds(10,deviates);
//  cout<<"most likely indices"<<endl;
//  cout<<inds<<endl;
}
