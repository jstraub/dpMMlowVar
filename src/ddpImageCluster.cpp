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
#include <euclideanData.hpp>
#include "spline.h"

typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> VXu;
typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> MXu;
typedef Eigen::MatrixXf MXf;
typedef Eigen::MatrixXi MXi;
typedef Eigen::Vector3f V3f;
typedef Eigen::Vector3i V3i;

using namespace std;
using namespace cv;
namespace po = boost::program_options;

class DMeansPaletteEncoder{
	public:
		DMeansPaletteEncoder(string fldrnm){
			paletteDiffsOut.open( (fldrnm + "/framediffs.log").c_str(), ios_base::out | ios_base::trunc);
			paletteOut.open( (fldrnm + "/palette.log").c_str(), ios_base::out | ios_base::trunc);
			this->fldrnm = fldrnm;
			fr = 0;
			doneFirst = false;
		}
		bool doneFirst;
		string fldrnm;
		int fr;
		ofstream paletteDiffsOut, paletteOut;
		MXu prevPaletteIds;
		map<int, vector<V3f> > paletteToColorSeq;
		map<int, int > paletteToFrameStart;

		void outputResiduals(Mat& frame, const VXu& z, const MXf& p){
			int rw = frame.rows;
			int cl = frame.cols;
			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9); //9 means maximum compression/slowest

			//now compute the residual image and output
			//change the true frame to RGB
			Mat tru(rw, cl, CV_8UC3);
			Mat truF(rw, cl, CV_8UC3);
			cvtColor(frame, truF, CV_Lab2RGB);
			truF.convertTo(tru, CV_8U, 255.0);
			//change the posterized frame to RGB
			Mat postLab(rw, cl, CV_32FC3);
			Mat postF(rw, cl, CV_32FC3);
			Mat post(rw, cl, CV_8UC3);
			int idx = 0;
			for(int y = 0; y < rw; y++){
				for (int x = 0; x < cl; x++){
					Vec3f& clr = postLab.at<Vec3f>(y, x);
					clr.val[0] = p(0, z(idx));
					clr.val[1] = p(1, z(idx));
					clr.val[2] = p(2, z(idx));
    		  		idx ++;
				}
			}
			cvtColor(postLab, postF, CV_Lab2RGB);
			postF.convertTo(post, CV_8U, 255.0);
			//get the residual image
			Mat res(rw, cl, CV_8UC3);
			for(int i = 0; i < rw; i++){
				for (int j = 0; j < cl; j++){
					Vec3f& clr = res.at<Vec3f>(i, j);
					Vec3f& clrt = tru.at<Vec3f>(i, j);
					Vec3f& clrp = post.at<Vec3f>(i, j);
				int tmp = 127 +clrt.val[0] - clrp.val[0]; clr.val[0] = tmp;
					tmp = 127 +clrt.val[1] - clrp.val[1]; clr.val[1] = tmp;
					tmp = 127 +clrt.val[2] - clrp.val[2]; clr.val[2] = tmp;
				}
			}
			ostringstream oss;
			oss << fldrnm << "/res-" << setw(7) << setfill('0') << fr << ".png";
			imwrite(oss.str(), res, compression_params);
		}
		void posterize(int rw, int cl, VXu z, MXf p){
			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9); //9 means maximum compression/slowest
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
			frameOutF.convertTo(frameOut, CV_8U, 255.0);
			//Mat medianFrame;
			//medianBlur(frameOut, medianFrame, 3);
			ostringstream oss;
			oss << fldrnm << "/post-" << setw(7) << setfill('0') << fr << ".png";
			imwrite(oss.str(), frameOut, compression_params);
		}
		void addFrame(Mat& nextFrame, const VXu& z, const MXf& p){
			int rw = nextFrame.rows;
			int cl = nextFrame.cols;
			if (doneFirst){
				//output the palette differences
				vector<int> diffrws, diffcls, newzs;
				for(int i = 0; i < rw; i++){
					for (int j = 0; j < cl; j++){
						if(z(i*cl+j) != prevPaletteIds(i, j)){
							diffrws.push_back(i);
							diffcls.push_back(j);
							newzs.push_back(z(i*cl+j));
						}
						prevPaletteIds(i, j) = z(i*cl+j);
					}
				}
				paletteDiffsOut<<newzs.size() << endl;
				for(int i = 0; i < newzs.size(); i++){
					paletteDiffsOut << diffrws[i] << " " << diffcls[i] << " " << newzs[i] << endl;
				}
			} else {
				vector<int> zs;
				for(int i = 0; i < rw; i++){
					for (int j = 0; j < cl; j++){
						zs.push_back(z(i*cl+j));
						prevPaletteIds(i, j) = z(i*cl+j);
					}
				}
				paletteDiffsOut << rw << " " << cl << endl;
				paletteDiffsOut << zs.size() << endl;
				for(int i = 0; i < zs.size(); i++){
					paletteDiffsOut << zs[i] << endl;
				}
			}

			outputResiduals(nextFrame, z, p);

			for (int i = 0; i < p.cols(); i++){
				if (paletteToColorSeq.find(i) == paletteToColorSeq.end()){
					paletteToFrameStart[i] = fr;
				}
				paletteToColorSeq[i].push_back(p.col(i));
			}
			fr++;
		}
		void outputPaletteSplines(){
			////spline the palette sequence 
			//for (auto it = paletteToColorSeq.begin(), it != paletteToColorSeq.end(); ++it){
			//	int stfr = paletteToFrameStart[it->first];
			//	vector<V3f>& clrseq = it->second;
			//	vector<int> toAdd;
			//	toAdd.push_back(0);
			//	toAdd.push_back(clrseq.size()-1);
			//	double maxerr = 10.0;
			//	while(maxerr > 1.0){
			//		vector<double> r_seq, g_seq, b_seq;
			//		vector<double> frms;
			//		for(int i = 0; i < toAdd.size(); i++){
			//			frms.push_back(toAdd[i]+stfr);
			//			r_seq.push_back(clrseq[toAdd[i]](0));
			//			g_seq.push_back(clrseq[toAdd[i]](1));
			//			b_seq.push_back(clrseq[toAdd[i]](2));
			//		}
			//		tk::spline s_r, s_g, s_b;
			//		s_r.set_points(frms, r_seq);
			//		s_g.set_points(frms, g_seq);
			//		s_b.set_points(frms, b_seq);
			//		maxerr = -1;
			//		int maxid = -1;
			//		for(int i = 0; i < clrseq.size(); i++){
			//			double pxerr = 
			//				  (s_r(stfr+i) - r_seq[i])*(s_r(stfr+i) - r_seq[i])
			//				 + (s_g(stfr+i) - g_seq[i])*(s_g(stfr+i) - g_seq[i])
			//				 + (s_b(stfr+i) - b_seq[i])*(s_b(stfr+i) - b_seq[i]);
			//			if (pxerr > maxerr){
			//				maxerr = pxerr;
			//				maxid = i;
			//			}
			//		}
			//		if(maxerr > 1.0){
			//			toAdd.push_back(maxid);
			//		}
			//	}
			//	//output the spline
			//	vector<double> r_seq, g_seq, b_seq;
			//	vector<double> frms;
			//	for(int i = 0; i < toAdd.size(); i++){
			//		frms.push_back(toAdd[i]+stfr);
			//		r_seq.push_back(clrseq[toAdd[i]](0));
			//		g_seq.push_back(clrseq[toAdd[i]](1));
			//		b_seq.push_back(clrseq[toAdd[i]](2));
			//	}
			//	tk::spline s_r, s_g, s_b;
			//	s_r.set_points(frms, r_seq);
			//	s_g.set_points(frms, g_seq);
			//	s_b.set_points(frms, b_seq);
			//	//TODO output
			//	paletteOut << "You didn't implement this yet, dummy :)" <<endl;
			//}

			paletteDiffsOut.close();
			paletteOut.close();
		}
		
};

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
	shared_ptr<MXf> tmp(new MXf(3, 1));
    	shared_ptr<ClDataGpuf> cld(new ClDataGpuf(tmp,0));
    	DDPMeansCUDA<float,Euclidean<float> > *clusterer = new
    	DDPMeansCUDA<float,Euclidean<float> >(cld, lambda, Q, tau);

	//set up the palette encoder
	DMeansPaletteEncoder dmpe(vm["frame_folder_name"].as<string>());

	//loop over frames, resize if necessary, and cluster
	int fr = 1;
	for(;;){
		printProgress(string("Processing ") + vidname, ((double)fr)/n_fr);
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
			clusterer->updateLabels();
    			clusterer->updateCenters();
		}while (!clusterer->converged());
		clusterer->updateState();
    		const VXu& z = clusterer->z();
    		const MXf& p = clusterer->centroids();
    		//posterize /compress the video
		dmpe.posterize(frame.rows, frame.cols, z, p);
		dmpe.addFrame(frame, z, p);
	}
	dmpe.outputPaletteSplines();
	cout << endl;
	delete clusterer;
	return 0;
}

