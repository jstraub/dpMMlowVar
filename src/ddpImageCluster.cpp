/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <opencv2/opencv.hpp>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include<boost/program_options.hpp>
#include<errno.h>
#include<sys/stat.h>
//#include "kmeans.hpp"

//#include <dmeans/core>
//#include <dmeans/iterative>
//#include <dmeans/model>
//#include <dmeans/utils>

#include <jsCore/clDataGpu.hpp>
#include <dpMMlowVar/ddpmeansCUDA.hpp>
#include <dpMMlowVar/euclideanData.hpp>
//#include <dpMMlowVar/spline.h>

typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> VXu;
typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> MXu;
typedef Eigen::MatrixXf MXf;
typedef Eigen::MatrixXi MXi;
typedef Eigen::Vector3f V3f;
typedef Eigen::Vector3i V3i;
typedef Eigen::Vector3d V3d;
//typedef dmeans::VectorSpaceModel<3> VSModel;

using namespace std;
using namespace dplv;
using namespace cv;
namespace po = boost::program_options;


int makeDirectory(const char* name);
//vector<VSModel::Data> extractVSModelData(Mat& frame);
void printProgress(string pre, double pct);
shared_ptr<MXf> extractVectorData(Mat& frame);
//Mat createOutputFrame(Mat& frame, dmeans::Results<VSModel>& res);
cv::Mat posterize(int rw, int cl, VXu z, MXf p);
cv::Mat boundaries(cv::Mat frame, VXu z);

int main(int argc, char** argv){
	// Set up the program options.
  	po::options_description desc("Option Flags");
  	desc.add_options()
  	  ("help,h", "produce help message")
  	  ("video_name,v", po::value<string>()->required(), "The name of the video")
  	  ("frame_folder_name,f", po::value<string>()->default_value("frames"), "The folder to store frames in")
  	  ("batch_size,b", po::value<int>()->default_value(-1), "The max number of pixels to cluster at each step")
  	  ("lambda,l", po::value<double>()->required(), "The value of lambda")
  	  ("T_Q,t", po::value<double>()->required(), "The value of T_Q")
  	  ("k_tau,k", po::value<double>()->required(), "The value of k_tau");


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
		return 1;
	}

	//get the video frame size/fps/etc
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

	//pull double constants from the command line options
	int bsz = vm["batch_size"].as<int>(); 
	double lambda = vm["lambda"].as<double>(); 
	double T_Q = vm["T_Q"].as<double>();
	double k_tau = vm["k_tau"].as<double>();
	double Q = lambda/T_Q;
	double tau = (T_Q*(k_tau-1.0)+1.0)/(T_Q-1.0);

	//find the video resizing factor based on how many pixels should be clustered in each frame
	double fsz;
	fsz = bsz > 0 ?  sqrt((double)bsz/(double)(fr_w*fr_h)) : 1.0;
	cout << "Video Resize Factor " << fsz << endl;
	int nfr_w = fsz*fr_w;
	int nfr_h = fsz*fr_h;
	cout << "New Frame Dimensions: " << nfr_w << " x " << nfr_h << endl;

	//make the directory for storing frames if it doesn't already exist
	ostringstream oss;
	oss << vm["frame_folder_name"].as<string>() << "-" << lambda <<"-" << T_Q << "-" << k_tau;
	if(makeDirectory(oss.str().c_str()) == -1){return 1;}



	//JULIAN: this is where you set up the dmeans object
	//the below  uses my new oop branch on github
	//here is where you'd insert your CUDA version setup
//	dmeans::Config dynm_cfg;
//	dynm_cfg.set("lambda", lambda);
//	dynm_cfg.set("Q", Q);
//	dynm_cfg.set("tau", tau);
//	dynm_cfg.set("nRestarts", 1);
//	dynm_cfg.set("verbose", false);
//	dmeans::DMeans<VSModel, dmeans::Iterative> dynm(dynm_cfg);

  //set up the DDP Means object
	shared_ptr<MXf> tmp(new MXf(3, 1));
  shared_ptr<jsc::ClDataGpuf> cld(new jsc::ClDataGpuf(tmp,0));
  DDPMeansCUDA<float,Euclidean<float> > *clusterer = new
    DDPMeansCUDA<float,Euclidean<float> >(cld, lambda, Q, tau);

	//loop over extracting a frame, possibly resizing it, and clustering
	int fr = 0;
	for(;;){
		printProgress(string("Posterizing ") + vidname, ((double)fr)/n_fr);
		Mat frame;
		cap.grab();
		bool empty = !cap.retrieve(frame);
		if(empty) break;
//    if(fr < 100) {++ fr; continue;}
    shared_ptr<MXf> data;
	  Mat frameresized = frame;
		if (nfr_w != fr_w || nfr_h != fr_h){
			resize(frame, frameresized, Size(nfr_w, nfr_h), 0, 0, INTER_CUBIC);
			data = extractVectorData(frameresized);
		} else {
			data = extractVectorData(frame);
		}


		//JULIAN: This is where you cluster vector space data
//		dmeans::Results<VSModel> res = dynm.cluster(data);
    jsc::Timer t0;
    jsc::Timer t1;
		clusterer->nextTimeStep(data);
    t1.toctic("init");
//		clusterer->nextTimeStep(data);
//    cout<<"serial assign"<<endl;
//		clusterer->updateLabelsSerial();
//    			clusterer->updateCenters();
    uint32_t t = 0;
		do{
			clusterer->updateLabels();
      clusterer->updateCenters();
      t1.toctic("iteration");
      ++ t;
		}while (!clusterer->convergedCounts(nfr_h*nfr_w/100)
        && t < 10);
//		}while (!clusterer->converged());
		clusterer->updateState(false);
    t0.toctic("whole iteration");

    const VXu& z = clusterer->z();
//    const MXf& p = clusterer->centroids();

		//JULIAN: here is where you take the results from your cuda algorithm and draw the superpixel boundaries in red/whatever color
//		Mat postFrame = createOutputFrame(frame, res);
//    cv::Mat postFrame = posterize(nfr_h,nfr_w, z, p);
    cv::Mat postFrame = boundaries(frameresized, z);

    cv::imshow("rgb",postFrame);
    cv::waitKey(40);

		//JULIAN: This is where you write out the frame + superpixel boundaries
		ostringstream oss;
		oss << vm["frame_folder_name"].as<string>() << "-" << lambda <<"-" << T_Q << "-" << k_tau << "/" << setw(7) << setfill('0') << fr++ << ".png";
		imwrite(oss.str(), postFrame, compression_params);
	}
	cout << endl;

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
	cout << "] " << (int)(pct*100) << "%" << "   \r" << flush;
	return;
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

//JULIAN: Change this function to include the (x, y) position of the pixel in the vector space data
//In my code, VSModel::Data is essentially a 3d vector type for (R, G, B) coords
//in your code, you should replace it with a 5d vector of some sort
//I have already handled the RGB -> LAB colorspace conversion
//vector<VSModel::Data> extractVSModelData(Mat& frame){
//	Mat framef, frameLab;
//	frame.convertTo(framef, CV_32F, 1.0/255.0);
//	cvtColor(framef, frameLab, CV_RGB2Lab);
//	vector<VSModel::Data> data;
//	for(int y = 0; y < frame.rows; y++){
//		for (int x = 0; x <frame.cols; x++){
//    		Vec3f veclab = frameLab.at<Vec3f>(y, x);
//			VSModel::Data d;
//			d.v(0) = veclab.val[0];
//			d.v(1) = veclab.val[1];
//			d.v(2) = veclab.val[2];
//			data.push_back(d);
//		}
//	}
//	return data;
//}

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
			(*data)(1, idx) = 100.*float(x)/float(frame.cols); //veclab.val[1];
			(*data)(2, idx) = 100.*float(y)/float(frame.rows); //veclab.val[2];
			idx++;
		}
	}
	//cout << "RGB: " << frame.at<Vec3b>(0, 0) << " Lab: " << frameLab.at<Vec3f>(0, 0) << " data: " << data[0].v.transpose() << endl;
	return shared_ptr<MXf>(data);
}

//JULIAN: Here is where you take the dmeans results and draw superpixel boundaries
//on the frame. The below code was used in my experiments to posterize the video.
//I've already handled all the LAB->RGB conversion here, so you shouldn't need to change that stuff too much.
//Mat createOutputFrame(Mat& frame, dmeans::Results<VSModel>& res){
//	Mat framef, frameLab;
//	frame.convertTo(framef, CV_32F, 1.0/255.0);
//	cvtColor(framef, frameLab, CV_RGB2Lab);
//	Mat frameOut(frame.rows, frame.cols, CV_8UC3);
//	Mat frameOutLab(frame.rows, frame.cols, CV_32FC3);
//	for(int y = 0; y < frame.rows; y++){
//		for (int x = 0; x <frame.cols; x++){
//			double minDistSq = std::numeric_limits<double>::infinity();
//			int minId = -1;
//			for (auto it = res.prms.begin(); it != res.prms.end(); ++it){
//    			const Vec3f& vl = frameLab.at<Vec3f>(y, x);
//    			Vec3f& vOut = frameOutLab.at<Vec3f>(y, x);
//				const Eigen::Vector3d& v = it->second.v;
//				double distsq = (v(0)-vl.val[0])*(v(0)-vl.val[0]) + (v(1)-vl.val[1])*(v(1)-vl.val[1]) + (v(2)-vl.val[2])*(v(2)-vl.val[2]);
//    			if ( distsq < minDistSq ){
//    				minDistSq = distsq;
//    				vOut.val[0] = v(0);
//    				vOut.val[1] = v(1);
//    				vOut.val[2] = v(2);
//				}
//			}
//		}
//	}
//	cvtColor(frameOutLab, framef, CV_Lab2RGB);
//	framef.convertTo(frameOut, CV_8U, 255.0);
//	return frameOut;
//}

cv::Mat posterize(int rw, int cl, VXu z, MXf p){
  Mat frameLabOut(rw, cl, CV_32FC3);
  Mat frameOutF(rw, cl, CV_32FC3);
  Mat frameOut(rw, cl, CV_8UC3);
  int idx = 0;
  uint32_t max = z.maxCoeff();
  for(int y = 0; y < rw; y++){
    for (int x = 0; x < cl; x++){
      Vec3f& clr = frameLabOut.at<Vec3f>(y, x);
      clr.val[0] = p(0, z(idx));
      clr.val[1] = (float(z(idx))*(255./float(max)))-127.;//;p(1, z(idx));
      clr.val[2] = (float(z(idx))*(255./float(max)))-127.;//;p(1, z(idx));
//      clr.val[2] = (float(z(idx))-(float(max)*0.5))*(255./float(max));//;p(1, z(idx));
//      clr.val[2] = float(z(idx));//;p(2, z(idx));
      idx ++;
    }
  }
  cvtColor(frameLabOut, frameOutF, CV_Lab2RGB);
  frameOutF.convertTo(frameOut, CV_8U, 255.0);
  return frameOut;
//  //Mat medianFrame;
//  //medianBlur(frameOut, medianFrame, 3);
//  ostringstream oss;
//  oss << fldrnm << "/post-" << setw(7) << setfill('0') << fr << ".png";
//  imwrite(oss.str(), frameOut, compression_params);
}

cv::Mat boundaries(cv::Mat frame, VXu z)
{
  int cl = frame.cols;
  int rw = frame.rows;
  Mat frameOut;
  frame.copyTo(frameOut);
  for(int y = 1; y < rw; y++)
    for (int x = 1; x < cl; x++)
    {
//      cout<<y<<" "<<x<<" "<<frameOut.rows<<" "<<frameOut.cols<<" "<<(x+y*cl)<<" "<<(x-1+y*cl)<<" "<<(x+(y-1)*cl)<<endl;
      if((z(x+y*cl) != z(x-1+y*cl)) || (z(x+y*cl) != z(x+(y-1)*cl)))
      {
        Vec3b& clr = frameOut.at<Vec3b>(y, x);
        clr.val[0] = 255;
        clr.val[1] = 0;
        clr.val[2] = 0;
      }  }
  return frameOut;
//  //Mat medianFrame;
//  //medianBlur(frameOut, medianFrame, 3);
//  ostringstream oss;
//  oss << fldrnm << "/post-" << setw(7) << setfill('0') << fr << ".png";
//  imwrite(oss.str(), frameOut, compression_params);
}
