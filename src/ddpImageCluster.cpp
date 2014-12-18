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
#include <ddpmeansCUDA.hpp>

typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> VXu;
typedef Eigen::MatrixXf MXf;
typedef Eigen::Vector3f V3f;

using namespace std;
using namespace cv;
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
  	  ("video_name,v", po::value<string>()->required(), "The name of the video")
  	  ("frame_folder_name,f", po::value<string>()->default_value("frames"), "The folder to store frames in")
  	  ("resize_factor,r", po::value<double>()->default_value(1.0), "The factor to resize the video by")
	  ("residual_bits,b", po::value<int>()->default_value(0), "The number of residual bits to store")
	  ("spline_error,e", po::value<double>()->default_value(1.0), "Allowable error on the color dictionary spline")
	  ("seed,s", po::value<int>()->default_value(time(0)), "Seed for the random number generator")
  	  ("lambda,l", po::value<double>()->required(), "The value of lambda")
  	  ("T_Q,t", po::value<double>()->required(), "The value of T_Q")
  	  ("k_tau,k", po::value<double>()->required(), "The value of k_tau")
  	  ;


  	po::variables_map vm;
  	po::store(po::parse_command_line(argc, argv, desc), vm);
  	po::notify(vm);    

	//if the user asked for help, display it and quit
	if(vm.count("help")){
		cout << desc << endl;
		return 0;
	}
	
	//create the capture object for the video
	string vidname = vm["video_name"].as<string>();
	VideoCapture cap(vidname);
	if(!cap.isOpened()){
		cout << "Couldn't open " << vidname << " for reading" << endl;
		return -1;
	} 

	//make the directory for storing frames if it doesn't already exist
	if(makeDirectory(vm["frame_folder_name"].as<string>().c_str()) == -1){return -1;}

	int n_fr = cap.get(CV_CAP_PROP_FRAME_COUNT);
	int fps = cap.get(CV_CAP_PROP_FPS);
	int fr_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int fr_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	cout << "Opened " << vidname << endl;
	cout << "Frame Dimensions: " << fr_w << " x " << fr_h << endl;
	cout << "Framerate: " << fps << endl;
	cout << "# Frames: " << n_fr << endl;
	cout << "Frame Storage Directory: " << vm["frame_folder_name"].as<string>() << endl;

	//compression params for PNG frame image writing
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9); //9 means maximum compression/slowest

	//pull double constants from the command line using stream
	double fsz = vm["resize_factor"].as<double>(); 
	double lambda = vm["lambda"].as<double>(); 
	double T_Q = vm["T_Q"].as<double>();
	double k_tau = vm["k_tau"].as<double>();
	double Q = lambda/T_Q;
	double tau = (T_Q*(k_tau-1.0)+1.0)/(T_Q-1.0);
	
	cout << "Video Resize Factor " << fsz << endl;
	int nfr_w = fsz*fr_w;
	int nfr_h = fsz*fr_h;
	cout << "New Frame Dimensions: " << nfr_w << " x " << nfr_h << endl;

	//set up the DDP Means object
	mt19937 rng; rng.seed(vm["seed"].as<int>());
	shared_ptr<MXf> tmp(new MXf(3, 1));
  	DDPMeansCUDA<float> *clusterer = new DDPMeansCUDA<float>(tmp, lambda, Q, tau, &rng);


	//loop over frames, resize if necessary, and cluster
	int fr = 1;
	for(;;){
		printProgress(string("Compressing ") + vidname, ((double)fr)/n_fr);
		Mat frame;
		cap.grab();
		bool empty = !cap.retrieve(frame);
		if(empty) break;
		if (nfr_w != fr_w || nfr_h != fr_h){
			Mat frameresized;
			resize(frame, frameresized, Size(nfr_w, nfr_h), 0, 0, INTER_CUBIC);
			frameresized.copyTo(frame);
		}
		shared_ptr<MXf> data = extractVectorData(frame);
		clusterer->nextTimeStep(data);
		do{
    	clusterer->updateCenters();
			clusterer->updateLabels();
		}while (!clusterer->converged());
		clusterer->updateState();
    		const VXu& z = clusterer->z();
    		const MXf& p = clusterer->centroids();
		Mat compressedFrame = compress(frame.rows, frame.cols, z, p);
		ostringstream oss;
		oss << vm["frame_folder_name"].as<string>() << "/" << setw(7) << setfill('0') << fr++ << ".png";
		imwrite(oss.str(), compressedFrame, compression_params);
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
	Mat framef, frameLab;
	frame.convertTo(framef, CV_32F, 1.0/255.0);
	cvtColor(framef, frameLab, CV_RGB2Lab);
	MXf* data = new MXf(3, frame.rows*frame.cols);
	int idx = 0;
	for(int y = 0; y < frame.rows; y++){
		for (int x = 0; x <frame.cols; x++){
    			const Vec3f& veclab = frameLab.at<Vec3f>(y, x);
			(*data)(0, idx) = veclab.val[0];
			(*data)(1, idx) = veclab.val[1];
			(*data)(2, idx) = veclab.val[2];
			idx++;
		}
	}
	//cout << "RGB: " << frame.at<Vec3b>(0, 0) << " Lab: " << frameLab.at<Vec3f>(0, 0) << " data: " << data[0].v.transpose() << endl;
	return shared_ptr<MXf>(data);
}

Mat compress(int rw, int cl, VXu z, MXf p){
	Mat frameLabOut(rw, cl, CV_32FC3);
	Mat frameOutF(rw, cl, CV_32FC3);
	Mat frameOut(rw, cl, CV_8UC3);
	int idx = 0;
	for(int y = 0; y < rw; y++){
		for (int x = 0; x < cl; x++){
			Vec3f& clr = frameLabOut.at<Vec3f>(y, x);
			clr.val[0] = p(0, z(idx));
			clr.val[1] = p(1, z(idx));
			clr.val[2] = p(2, z(idx));
      idx ++;
		}
	}
	cvtColor(frameLabOut, frameOutF, CV_Lab2RGB);
//  imshow("compressed",frameOutF);
//  waitKey(1);
	frameOutF.convertTo(frameOut, CV_8U, 255.0);
	return frameOut;
	//Mat medianFrame;
	//medianBlur(frameOut, medianFrame, 3);
	//return medianFrame;
}
