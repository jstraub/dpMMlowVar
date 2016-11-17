#include "mex.h"
#include <limits>
#include <Eigen/Eigen>
#include <stdio.h>
#include <../../include/dpMMlowVar/ddpmeans.hpp>

using namespace Eigen;
using namespace std;
using namespace dplv;

typedef Map<MatrixXd,Aligned> MexMat;

#define IS_R1(P) (mxGetNumberOfElements(P)==1 && !mxIsSparse(P) && \
        mxIsDouble(P))
#define IS_REAL(P) (!mxIsSparse(P) && mxIsDouble(P))

static DDPMeans<double, Euclidean<double> > *clusterer = NULL;
static int D = 0;
static int N = 0;
static int T = 0;


void RunClusterer() {
  if (D==0 || N==0 || !clusterer) mexErrMsgTxt("Invalid state.");
  for (int t=0; t<T; ++t) {
    clusterer->updateCenters();
    clusterer->updateLabels();
    if (clusterer->converged()) break;
  }
}

// varargout = mex_dpmm(command, varargin)
// EXAMPLES
//   [z, ctr, cost, dev, wts, age] = mex_ddp('init', X1, lambda, Q, tau, T);
//   [z2, ctr2, cost2, dev2, wts2, age2] = mex_ddp('step', X2, lambda, Q,tau,T);
//   mex_ddp('close');
//
// INPUTS
//   command: matlab string, one of {'init', 'step', 'close'}
//   varargin: depends on command:
//     'init':
//       X: (D x N) data matrix, for N data points, D dimensions.
//       lambda: real, see original code/paper.
//       Q: real, see original code/paper.
//       Tau: real, see original code/paper.
//       T: real, number of clustering iterations
//     'step':
//       X: (D x N) data matrix, for N data points, D dimensions.
//       doRevive: real, boolean-interpreted flag
//     'close':
//       No arguments
//
//   varargout: depends on command:
//     'init' and 'step':
//       z: cluster indices, 1xN
//       ctr: cluster centroids, (D x K') for K' the current # of clusters
//       cost: real, clustering cost
//       dev: real, clustering deviation
//       wts: cluster weights (always 0? Doesn't seem to be meaningful)
//       ages: cluster ages (doesn't seem to work)
//    'close':
//       No return values.
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Define and check inputs
  #define IN_CMD prhs[0]

  if (nrhs < 1) mexErrMsgTxt("Must pass at least one argument.");

  char command[128];
  mxGetString(IN_CMD,command,128);
  
  if (!strcmp(command,"init")) {
    #define IN_INIT_X prhs[1]
    #define IN_INIT_LAMBDA prhs[2]
    #define IN_INIT_Q prhs[3]
    #define IN_INIT_TAU prhs[4]
    #define IN_INIT_T prhs[5]
    if (nrhs != 6) mexErrMsgTxt("Must provide X, lambda, Q, Tau and T for init .");
    if (!IS_REAL(IN_INIT_X)) mexErrMsgTxt("Input X must be real.");
    if (!IS_R1(IN_INIT_LAMBDA)) mexErrMsgTxt("Input lambda must be real.");
    if (!IS_R1(IN_INIT_Q)) mexErrMsgTxt("Input Q must be real.");
    if (!IS_R1(IN_INIT_TAU)) mexErrMsgTxt("Input tau must be real.");
    if (!IS_R1(IN_INIT_T)) mexErrMsgTxt("Input T must be real.");
    if (mxGetNumberOfDimensions(IN_INIT_X)!=2)
      mexErrMsgTxt("Input X must be DxN");
    if (clusterer) mexErrMsgTxt("Must call 'close' before another init.");

    mwSize nSz = mxGetNumberOfDimensions(IN_INIT_X);
    const mwSize* sz = mxGetDimensions(IN_INIT_X);
    N = sz[1]; D = sz[0];

    MexMat X(mxGetPr(IN_INIT_X),D,N);
    shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
    MatrixXd& x(*spx);
    x = X.replicate(1,1);
    
    float lambda = *mxGetPr(IN_INIT_LAMBDA);
    float Q = *mxGetPr(IN_INIT_Q);
    float tau = *mxGetPr(IN_INIT_TAU);

    T = static_cast<int>(*mxGetPr(IN_INIT_T));

    if (!clusterer)
      clusterer = new DDPMeans<double,Euclidean<double> >(spx, lambda, Q, tau);
    RunClusterer();
  } else if (!strcmp(command,"step")) {
    #define IN_STEP_X prhs[1]
    #define IN_STEP_DOREVIVE prhs[2]
    if (!clusterer) mexErrMsgTxt("Invalid state.");
    if (nrhs != 3) mexErrMsgTxt("Must provide X and doRevive");
    if (!IS_REAL(IN_STEP_X)) mexErrMsgTxt("Input X must be real.");
    if (!IS_R1(IN_STEP_DOREVIVE)) mexErrMsgTxt("Input doRevive must be real.");
    float reviveFlag = *mxGetPr(IN_STEP_DOREVIVE);
    bool doRevive = abs(reviveFlag) > numeric_limits<float>::epsilon();
    
    mwSize nSz = mxGetNumberOfDimensions(IN_STEP_X);
    const mwSize* sz = mxGetDimensions(IN_STEP_X);
    N = sz[1]; D = sz[0];

    MexMat X(mxGetPr(IN_STEP_X),D,N);
    shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
    MatrixXd& x(*spx);
    x = X.replicate(1,1);

    clusterer->nextTimeStep(spx, doRevive);
    RunClusterer();
  }

  // Done, delete the shared objects so we can use it again
  if (!strcmp(command,"close")) {
    delete clusterer;
    clusterer = NULL;
    D = N = T = 0;
  } 
  
  // Prepare outputs for init or step calls
  if (!strcmp(command,"init") || !strcmp(command,"step")) {
    #define OUT_Z plhs[0]
    #define OUT_CENTROIDS plhs[1]
    #define OUT_COST plhs[2]
    #define OUT_DEVIATION plhs[3]
    #define OUT_WEIGHTS plhs[4]
    #define OUT_AGES plhs[5]
    if (D==0 || N==0 || !clusterer) mexErrMsgTxt("Invalid state.");
    
    OUT_Z = mxCreateNumericMatrix(1, N, mxUINT32_CLASS, mxREAL);
    unsigned int* pZ = (unsigned int*) mxGetData(OUT_Z);
    const VectorXu& z = clusterer->z();
    Map<VectorXu>(pZ, N, 1) = z;

    int finalK = clusterer->getK();
    OUT_CENTROIDS = mxCreateNumericMatrix(D, finalK, mxDOUBLE_CLASS, mxREAL);
    double* pC = mxGetPr(OUT_CENTROIDS);
    MatrixXd centroids = clusterer->centroids();
    Map<MatrixXd>(pC, D, finalK) = centroids;

    OUT_COST = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    double* pCost = mxGetPr(OUT_COST);
    *pCost = clusterer->cost();

    OUT_DEVIATION = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    double* pDev = mxGetPr(OUT_DEVIATION);
    *pDev = clusterer->avgIntraClusterDeviation();

    OUT_WEIGHTS = mxCreateNumericMatrix(finalK, 1, mxDOUBLE_CLASS, mxREAL);
    double* pW = mxGetPr(OUT_WEIGHTS);
    MatrixXd weights = clusterer->weights();
    Map<MatrixXd>(pW, finalK, 1) = weights;

    OUT_AGES = mxCreateNumericMatrix(finalK, 1, mxDOUBLE_CLASS, mxREAL);
    double* pA = mxGetPr(OUT_AGES);
    MatrixXd ages = clusterer->ages();
    Map<MatrixXd>(pA, finalK, 1) = ages;
  }
  return;
}
