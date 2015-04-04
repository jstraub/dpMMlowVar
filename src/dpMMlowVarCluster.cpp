/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include <jsCore/timer.hpp>

#include <dpMMlowVar/kmeans.hpp>
#include <dpMMlowVar/dpmeans.hpp>
#include <dpMMlowVar/ddpmeans.hpp>

using namespace Eigen;
using namespace std;
using namespace dplv;
namespace po = boost::program_options;

typedef double flt;

int main(int argc, char **argv)
{

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("seed", po::value<int>(), "seed for random number generator")
    ("N,N", po::value<int>(), "number of input datapoints")
    ("D,D", po::value<int>(), "number of dimensions of the data")
    ("T,T", po::value<int>(), "iterations")
    ("alpha,a", po::value< vector<double> >()->multitoken(), 
      "alpha parameter of the DP (if single value assumes all alpha_i are the "
      "same")
    ("K,K", po::value<int>(), "number of initial clusters ")
    ("base", po::value<string>(), 
      "which base measure to use (only spkm, kmeans, DPvMFmeans right now)")
    ("params,p", po::value< vector<double> >()->multitoken(), 
      "parameters of the base measure")
    ("input,i", po::value<string>(), 
      "path to input dataset .csv file (rows: dimensions; cols: different "
      "datapoints)")
    ("output,o", po::value<string>(), 
      "path to output labels .csv file (rows: time; cols: different "
      "datapoints)")
    ("mlInds", "output ml indices")
    ("centroids", "output centroids of clusters")
    ("silhouette", "output average silhouette")
    ("shuffle", "shuffle the data before processing")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  uint64_t seed = time(0);
  if(vm.count("seed"))
    seed = static_cast<uint64_t>(vm["seed"].as<int>());
//  boost::mt19937 rndGen(seed);
  std::srand(seed);
  uint32_t K=5;
  if (vm.count("K")) K = vm["K"].as<int>();
  // number of iterations
  uint32_t T=100;
  if (vm.count("T")) T = vm["T"].as<int>();
  uint32_t N=100;
  if (vm.count("N")) N = vm["N"].as<int>();
  uint32_t D=2;
  if (vm.count("D")) D = vm["D"].as<int>();
  cout << "T="<<T<<endl;
  // DP alpha parameter
  VectorXd alpha(K);
  alpha.setOnes(K);
  if (vm.count("alpha"))
  {
    vector<double> params = vm["alpha"].as< vector<double> >();
    if(params.size()==1)
      alpha *= params[0];
    else
      for (uint32_t k=0; k<K; ++k)
        alpha(k) = params[k];
  }
  cout << "alpha="<<alpha.transpose()<<endl;

  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  MatrixXd& x(*spx);
  string pathIn ="";
  if(vm.count("input")) pathIn = vm["input"].as<string>();
  cout<<"loading data from "<<pathIn<<endl;
  ifstream fin(pathIn.data(),ifstream::in);
//  fin >> D,N;

  vector<uint32_t> ind(N);
  for (uint32_t i=0; i<N; ++i)
    ind[i] = i;
  if(vm.count("shuffle"))
  {
    cout<<"shuffling input"<<endl;
    std::random_shuffle(ind.begin(),ind.end());
  }
  for (uint32_t j=0; j<D; ++j)
    for (uint32_t i=0; i<N; ++i)
      fin>>x(j,ind[i]);
    //cout<<x<<endl;

  // which base distribution
  string base = "DPvMFmeans";
  if(vm.count("base")) base = vm["base"].as<string>();

  if(base.compare("kmeans")){
    // normalize to unit length
    int err = 0;
#pragma omp parallel for
    for (uint32_t i=0; i<N; ++i)
      if(fabs(x.col(i).norm() - 1.0) > 1e-1)
      {
        err++;
        cout<<x.col(i).norm() <<endl;
      }else
        x.col(i) /= x.col(i).norm();
    if(err>0) return 0;
  }
  
  Clusterer<double, Spherical<double> > *clustSp = NULL;
  Clusterer<double, Euclidean<double> > *clustEu = NULL;
  if(!base.compare("DPvMFmeans")){
    double lambda = cos(5.0*M_PI/180.0);
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<endl;
      lambda = params[0];
    }
//    clustSp = new DPvMFMeans<double>(spx, K, lambda, &rndGen);
//    clustSp = new DDPvMFMeans<double>(spx, lambda, 1. , 0. , &rndGen);
    clustSp = new DDPMeans<double,Spherical<double> >(spx, lambda, 1. , 0.);
  }else if(!base.compare("spkm")){
    clustSp = new KMeans<double,Spherical<double> >(spx, K);
//    clustSp = new SphericalKMeans<double>(spx, K, &rndGen);
//  }else if(!base.compare("spkmKarcher")){
//    clustSp = new SphericalKMeansKarcher<double>(spx, K, &rndGen);
  }else if(!base.compare("kmeans")){
    clustEu = new KMeans<double,Euclidean<double> >(spx, K);
  }else{
    cout<<"base "<<base<<" not supported"<<endl;
    return 1;
  }

  string pathOut ="./labels.csv";
  if(vm.count("output")) 
    pathOut = vm["output"].as<string>();
  cout<<"output to "<<pathOut<<endl;

  double silhouette = -1.;
  MatrixXd deviates;
  MatrixXd centroids;
  MatrixXu inds;
  ofstream fout(pathOut.data(),ofstream::out);
  ofstream foutJointLike((pathOut+"_jointLikelihood.csv").data(),ofstream::out);
  jsc::Timer watch;
  if(clustSp != NULL)
  {
    for (uint32_t t=0; t<T; ++t)
    {
      cout<<"------------ t="<<t<<" -------------"<<endl;
      watch.tic();
      clustSp->updateCenters();
      watch.toctic("-- updateCenters");

      const VectorXu& z = clustSp->z();
      for (uint32_t i=0; i<z.size()-1; ++i) 
        fout<<z(ind[i])<<" ";
      fout<<z(ind[z.size()-1])<<endl;
      double deviation = clustSp->avgIntraClusterDeviation();

      cout<<"   K="<<clustSp->getK()<<" " <<z.size()<<endl;
      if(clustSp->getK()>0)
      {
        cout<<"  counts=   "<< clustSp->counts().transpose();
        cout<<" avg deviation  "<<deviation<<endl;
      }
      foutJointLike<<deviation<<endl;

      watch.tic();
      clustSp->updateLabels();
      watch.toctic("-- updateLabels");
      cout<<" cost fct value "<<clustSp->cost()<< "\tconverged? "<<clustSp->converged()<<endl;
      if(clustSp->converged()) break;
    }
    if(vm.count("silhouette")) silhouette = clustSp->silhouette();
    if(vm.count("mlInds")) inds = clustSp->mostLikelyInds(10,deviates);
    if(vm.count("centroids")) centroids = clustSp->centroids();
  }else if(clustEu != NULL)
  {
    for (uint32_t t=0; t<T; ++t)
    {
      cout<<"------------ t="<<t<<" -------------"<<endl;
      watch.tic();
      clustEu->updateCenters();
      watch.toctic("-- updateCenters");

      const VectorXu& z = clustEu->z();
      for (uint32_t i=0; i<z.size()-1; ++i) 
        fout<<z(ind[i])<<" ";
      fout<<z(ind[z.size()-1])<<endl;
      double deviation = clustEu->avgIntraClusterDeviation();

      cout<<"   K="<<clustEu->getK()<<" " <<z.size()<<endl;
      if(clustEu->getK()>0)
      {
        cout<<"  counts=   "<< clustEu->counts().transpose();
        cout<<" avg deviation  "<<deviation<<endl;
      }
      foutJointLike<<deviation<<endl;

      watch.tic();
      clustEu->updateLabels();
      watch.toctic("-- updateLabels");
      cout<<" cost fct value "<<clustEu->cost()<< "\tconverged? "<<clustEu->converged()<<endl;
      if(clustEu->converged()) break;
    }
    if(vm.count("silhouette")) silhouette = clustEu->silhouette();
    if(vm.count("mlInds")) inds = clustEu->mostLikelyInds(10,deviates);
    if(vm.count("centroids")) centroids = clustEu->centroids();
  }
  fout.close();

  if(vm.count("silhouette")) 
  {
    cout<<"silhouette = "<<silhouette<<" saved to "<<(pathOut+"_measures.csv")<<endl;
    fout.open((pathOut+"_measures.csv").data(),ofstream::out);
    fout<<silhouette<<endl;
    fout.close();
  }

  if(vm.count("mlInds")) 
  {
    cout<<"most likely indices"<<endl;
    cout<<inds<<endl;
    cout<<"----------------------------------------"<<endl;
    fout.open((pathOut+"mlInds.csv").data(),ofstream::out);
    fout<<inds<<endl;
    fout.close();
    fout.open((pathOut+"mlLogLikes.csv").data(),ofstream::out);
    fout<<deviates<<endl;
    fout.close();
  }
  if(vm.count("centroids")) 
  {
    ofstream foutMeans((pathOut+"_means.csv").data(),ofstream::out);
    foutMeans << centroids <<endl;
    foutMeans.close();
  }

};

