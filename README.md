
## Bayesian nonparametric low variance asymptotic clustering algorithms
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
