
#include <iostream>
#include <fstream>
#include <string>
#include <boost/program_options.hpp>

#include "spkm.hpp"
#include "kmeans.hpp"
#include "ddpvMFmeans.hpp"
#include "timer.hpp"

using namespace Eigen;
using namespace std;
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
      "which base measure to use (only spkm, kmeans, DPvMFMeans right now)")
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
  boost::mt19937 rndGen(seed);
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
  for (uint32_t j=0; j<D; ++j)
    for (uint32_t i=0; i<N; ++i)
      fin>>x(j,i);
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
  
  Clusterer<double> *clusterer = NULL;
  if(!base.compare("DPvMFmeans")){
    double lambda = cos(5.0*M_PI/180.0);
    if(vm.count("params"))
    {
      vector<double> params = vm["params"].as< vector<double> >();
      cout<<"params length="<<params.size()<<endl;
      lambda = params[0];
    }
//    clusterer = new DPvMFMeans<double>(spx, K, lambda, &rndGen);
    clusterer = new DDPvMFMeans<double>(spx, lambda, 1. , 0. , &rndGen);
  }else if(!base.compare("spkm")){
    clusterer = new SphericalKMeans<double>(spx, K, &rndGen);
//  }else if(!base.compare("spkmKarcher")){
//    clusterer = new SphericalKMeansKarcher<double>(spx, K, &rndGen);
  }else if(!base.compare("kmeans")){
    clusterer = new KMeans<double>(spx, K, &rndGen);
  }else{
    cout<<"base "<<base<<" not supported"<<endl;
    return 1;
  }

  string pathOut ="./labels.csv";
  if(vm.count("output")) 
    pathOut = vm["output"].as<string>();
  cout<<"output to "<<pathOut<<endl;

  ofstream fout(pathOut.data(),ofstream::out);
  ofstream foutJointLike((pathOut+"_jointLikelihood.csv").data(),ofstream::out);
  Timer watch;
  for (uint32_t t=0; t<T; ++t)
  {
    cout<<"------------ t="<<t<<" -------------"<<endl;
    watch.tic();
    clusterer->updateCenters();
    watch.toctic("-- updateCenters");

    const VectorXu& z = clusterer->z();
    for (uint32_t i=0; i<z.size()-1; ++i) 
      fout<<z(i)<<" ";
    fout<<z(z.size()-1)<<endl;
    double deviation = clusterer->avgIntraClusterDeviation();

    cout<<"   K="<<clusterer->getK()<<" " <<z.size()<<endl;
    if(clusterer->getK()>0)
    {
      cout<<"  counts=   "<<counts<double,uint32_t>(z,clusterer->getK()).transpose();
      cout<<" avg deviation  "<<deviation<<endl;
    }
    foutJointLike<<deviation<<endl;

    watch.tic();
    clusterer->updateLabels();
    watch.toctic("-- updateLabels");
    cout<<" cost fct value "<<clusterer->cost()<< "\tconverged? "<<clusterer->converged()<<endl;
    if(clusterer->converged()) break;
  }
  //TODO do I need updateState for DPvMF means here??
  fout.close();

  if(vm.count("silhouette")) 
  {
    double silhouette = clusterer->silhouette();
    cout<<"silhouette = "<<silhouette<<" saved to "<<(pathOut+"_measures.csv")<<endl;
    fout.open((pathOut+"_measures.csv").data(),ofstream::out);
    fout<<silhouette<<endl;
    fout.close();
  }

  if(vm.count("mlInds")) 
  {
    MatrixXd deviates;
    MatrixXu inds = clusterer->mostLikelyInds(10,deviates);
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
    foutMeans << clusterer->centroids()<<endl;
    foutMeans.close();
  }

};

