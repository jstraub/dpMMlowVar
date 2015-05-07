## Bayesian nonparametric small-variance asymptotic clustering algorithms

This is a library of Bayesian nonparametric small-variance asymptotic
clustering algorithms: DP-means, Dynamic means, DP-vMF-means,
DDP-vMF-means.

For comparison reasons the library also implements k-means as well as
spherical k-means.

This library comes with one executable that allows batch clustering
using DP-vMF-means, DP-means, spherical k-means and k-means.

For an example of using DDP-vMF-means refer to
[git@github.com:jstraub/rtDDPvMF.git](git@github.com:jstraub/rtDDPvMF.git) 
which relies on this package's dpMMlowVar library to perform real-time
directional segmentation from Kinect RGB-D streams using DDP-vMF-means.

If you use DP-vMF-means or DDP-vMF-means please cite:
```
Julian Straub, Trevor Campbell, Jonathan P. How, John W. Fisher III. 
"Small-Variance Nonparametric Clustering on the Hypersphere", In CVPR,
2015.
```
If you use Dynamic-means please cite:
```
T. Campbell, M. Liu, B. Kulis, J. How, and L. Carin. "Dynamic
Clustering via Asymptotics of the Dependent Dirichlet Process Mixture".
In Advances in Neural Information Processing Systems (NIPS), 2013.
```
## Usage
```
./dpMMlowVarCluster -h
Allowed options:
  -h [ --help ]         produce help message
  --seed arg            seed for random number generator
  -N [ --N ] arg        number of input datapoints
  -D [ --D ] arg        number of dimensions of the data
  -T [ --T ] arg        iterations
  -a [ --alpha ] arg    alpha parameter of the DP (if single value assumes all 
                        alpha_i are the same
  -K [ --K ] arg        number of initial clusters 
  --base arg            which base measure to use (only spkm, kmeans, 
                        DPvMFmeans right now)
  -p [ --params ] arg   parameters of the base measure
  -i [ --input ] arg    path to input dataset .csv file (rows: dimensions; 
                        cols: different datapoints)
  -o [ --output ] arg   path to output labels .csv file (rows: time; cols: 
                        different datapoints)
  --mlInds              output ml indices
  --centroids           output centroids of clusters
  --silhouette          output average silhouette
  --shuffle             shuffle the data before processing
```

## Clustering videos in RGB space using Dynamic means (video posterisation)
One example here is clustering of RGB values in images
```
./ddpImageCluster
```
## Contributors
Julian Straub and Trevor D. Campbell
