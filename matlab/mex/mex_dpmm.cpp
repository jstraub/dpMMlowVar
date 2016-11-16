#include "mex.h"
#include <Eigen/Eigen>
#include <../../include/dpMMlowVar/kmeans.hpp>
#include <../../include/dpMMlowVar/dpmeans.hpp>
#include <../../include/dpMMlowVar/ddpmeans.hpp>

using namespace Eigen;
using namespace std;
using namespace dplv;

typedef Map<MatrixXd,Aligned> MexMat;

#define IS_R1(P) (mxGetNumberOfElements(P)==1 && !mxIsSparse(P) && \
        mxIsDouble(P))
#define IS_REAL(P) (!mxIsSparse(P) && mxIsDouble(P))


// [z, ctr, cost, dev] = mex_dpmm(X, alpha, k, base)
// inputs
//   X: (D x N) data matrix, for N the # of data points, D the dimensionality.
//   alpha: 1x1 or Kx1 vector of DP param. If 1x1, then all K alpha are same.
//   k: # of initial clusters.
//   T: # iterations
//   lambda: DP-means param
// outputs
//   z: max-likelihood indices of clusters, 1xN
//   ctr: cluster centroids, DxK', for K' the final number of clusters.
//   cost: clustering cost
//   dev: mean cluster deviation
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Define and check inputs
  #define IN_X prhs[0]
  #define IN_ALPHA prhs[1]
  #define IN_K prhs[2]
  #define IN_T prhs[3]
  #define IN_LAMBDA prhs[4]
  #define OUT_Z plhs[0]
  #define OUT_CENTROIDS plhs[1]
  #define OUT_COST plhs[2]
  #define OUT_DEVIATION plhs[3]

  if (nrhs != 5) mexErrMsgTxt("Wrong number of inputs, must provide 5.");
  if (nlhs > 4) mexErrMsgTxt("Wrong number of outputs, at most 4.");
  if (!IS_REAL(IN_X)) mexErrMsgTxt("Input X must be real.");
  if (mxGetNumberOfDimensions(IN_X)!=2) mexErrMsgTxt("Input X must be DxN");

  if (!IS_REAL(IN_K)) mexErrMsgTxt("Input k must be real.");
  if (!IS_REAL(IN_ALPHA)) mexErrMsgTxt("Input alpha must be real.");

  mwSize nSz = mxGetNumberOfDimensions(IN_X);
  const mwSize* sz = mxGetDimensions(IN_X);
  int D = sz[0], N = sz[1];

  int k = static_cast<int>(*mxGetPr(IN_K));
  float alpha = *mxGetPr(IN_ALPHA);
  int T = static_cast<int>(*mxGetPr(IN_T));
  float lambda = *mxGetPr(IN_LAMBDA);

  mexPrintf("Inputs X (D=%d x N=%d), k: %d, alpha: %.4f\n, T: %d", D, N, k, alpha, T);

  // todo: get MatrixXd* from Eigen::Map (through Ref?) so we don't copy data,
  MexMat X(mxGetPr(IN_X),D,N);
  shared_ptr<MatrixXd> spx(new MatrixXd(D,N));
  MatrixXd& x(*spx);
  x = X.replicate(1,1);

  Clusterer<double, Euclidean<double> > *clusterer = NULL;
  clusterer = new DPMeans<double,Euclidean<double> >(spx, k, lambda);
  for (int t=0; t<T; ++t) {
    clusterer->updateCenters();
    clusterer->updateLabels();
    if (clusterer->converged()) break;
  }

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

  delete clusterer;
  return;
}
