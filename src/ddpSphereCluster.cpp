#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <sys/stat.h>
#include <errno.h>
#include <boost/program_options.hpp>

#include <dpMMlowVar/ddpmeansCUDA.hpp>
#include <dpMMlowVar/sphericalData.hpp>
#include <dpMMlowVar/opencvHelper.hpp>

typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> VXu;
typedef Eigen::MatrixXf MXf;
typedef Eigen::Vector3f V3f;

using namespace std;
using namespace cv;
using namespace dplv;
namespace po = boost::program_options;

int makeDirectory(const char* name);
shared_ptr<MXf> extractVectorData(Mat& frame);
Mat compress(int rw, int cl, VXu z, MXf p);

void printProgress(string pre, double pct);
int main(int argc, char** argv){
	// Set up the program options.
  	po::options_description desc("Option Flags");
  	desc.add_options()
  	  ("help,h", "produce help message")
      ("input,i", po::value<string>(), "path to folder with surface normals")
      ("frame_folder_name,f", po::value<string>()->default_value("frames"), "The folder to store frames in")
      ("seed,s", po::value<int>()->default_value(time(0)), "Seed for the random number generator")
      ("lambda,l", po::value<double>(), "The value of lambda (in deg)")
      ("T_Q,t", po::value<double>(), "The value of T_Q (determines Q) - how many frames does a point survive")
      ("beta,b", po::value<double>(), "The value of beta")
  	  ;

  	po::variables_map vm;
  	po::store(po::parse_command_line(argc, argv, desc), vm);
  	po::notify(vm);    

	//if the user asked for help, display it and quit
	if(vm.count("help")){
		cout << desc << endl;
		return 0;
	}

	string inputPath = vm["input"].as<string>();

	//make the directory for storing frames if it doesn't already exist
	if(makeDirectory(vm["frame_folder_name"].as<string>().c_str()) == -1){return -1;}

	//pull double constants from the command line using stream
	double lambda = cos(vm["lambda"].as<double>()*M_PI/180.0) -1.; 
	double T_Q = vm["T_Q"].as<double>();
	double beta = vm["beta"].as<double>();
	double Q = T_Q == 0.? -2. : lambda/T_Q;
	
	//set up the DDP Means object
//	shared_ptr<MXf> tmp(new MXf(3, 1));
//  shared_ptr<ClDataGpuf> cld(new ClDataGpuf(tmp,0));
////  DDPMeansCUDA<float,Spherical<float> > *clusterer = new
//  Clusterer<float,Spherical<float> > *clusterer = new
//    DDPMeansCUDA<float,Spherical<float> >(cld, lambda, Q, beta);
   
	shared_ptr<MXf> tmp(new MXf(3, 1));
  shared_ptr<ClDataGpuf> cld(new ClDataGpuf(tmp,6));
  Clusterer<float,Spherical<float> > *clusterer = new
    KMeans<float,Spherical<float> >(cld);

	//loop over frames, resize if necessary, and cluster
	int fr = 0;
	for(;;){
    char fileName[100];
    sprintf(fileName,"/%05d.bin",fr);
		Mat frame = imreadBinary(inputPath+std::string(fileName));
		if(frame.rows == 0 || frame.cols == 0) break;

		shared_ptr<MXf> data = extractVectorData(frame);
    std::cout<<"spherical data size: "<<data->rows()<<"x"<<data->cols()<<std::endl; 
		clusterer->nextTimeStep(data);
		do{
			clusterer->updateLabels();
    	clusterer->updateCenters();
      cout<<"cost = "<<clusterer->cost()<<endl;
		}while (!clusterer->converged());
		clusterer->updateState();
    const VXu& z = clusterer->z();
    const MXf& p = clusterer->centroids();
    cout<<p<<endl;
    cout<<"z min/max: "<<z.maxCoeff()<<" "<<z.minCoeff()<<endl;
//		Mat compressedFrame = compress(frame.rows, frame.cols, z, p);
//		ostringstream oss;
//		oss << vm["frame_folder_name"].as<string>() << "/" << setw(7) << setfill('0') << fr++ << ".png";
//		imwrite(oss.str(), compressedFrame, compression_params);
    ++ fr;
	}
	cout << endl;
	delete clusterer;
	return 0;
}

int makeDirectory(const char* name){
	if(mkdir(name, S_IRWXU) == -1){
		if (errno != EEXIST){ //if the folder exists, who cares just go for it
			cout << "Couldn't create " << name << "/ for writing frames" << endl;
			switch(errno){
				case EACCES:
					cout << "Need higher permissions" << endl;
					break;
				case ENAMETOOLONG:
					cout << "Folder name too long" << endl;
					break;
				case ENOSPC:
					cout << "Not enough space" << endl;
					break;
				case ENOENT:
					cout << "Something wrong with the path..." << endl;
					break;
				default:
					cout << "Unknown error" << endl;
			}
			return -1;
		}
	}
	return 0;
}

void printProgress(string pre, double pct){
	cout << pre << ": [";
	int nEq = pct*50;
	for (int i = 0; i < nEq; i++){
		cout << "=";
	}
	for (int i = nEq; i < 50; i++){
		cout << " ";
	}
	cout << "] " << (int)(pct*100) << "%" << 
	///////////////////////////////space buffer after text to prevent weird looking output///////////////////////
	"                                                                                                     \r"; //
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << flush;
	return;
}

shared_ptr<MXf> extractVectorData(Mat& frame){
	MXf* data = new MXf(3, frame.rows*frame.cols);
	int idx = 0;
	for(int y = 0; y < frame.rows; y++){
		for (int x = 0; x <frame.cols; x++){
      const Vec3f& vec = frame.at<Vec3f>(y, x);
      (*data)(0, idx) = vec.val[0];
      (*data)(1, idx) = vec.val[1];
      (*data)(2, idx) = vec.val[2];
      data->col(idx) /= data->col(idx).norm();
      idx++;
    }
	}
	//cout << "RGB: " << frame.at<Vec3b>(0, 0) << " Lab: " << frameLab.at<Vec3f>(0, 0) << " data: " << data[0].v.transpose() << endl;
	return shared_ptr<MXf>(data);
}

