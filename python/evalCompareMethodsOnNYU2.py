import numpy as np
import cv2
import os, copy, re
import fnmatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from js.utils.plot.colors import colorScheme
from js.utils.config import Config2String

#from evalLowVarSphereMMsynthetic import plotOverParams
from plots import plotOverParams

#paper
mpl.rc('font',size=35) 
mpl.rc('lines',linewidth=4.)
figSize = (14, 5.5)
figSize = (14, 12)

cfg=dict()
cfg['path'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2'
cfg['base'] = ['DpNiw' , 'DpNiwSphereFull', 'spkm'];
cfg['base'] = ['DPvMFmeans','spkm']; 
#cfg['base'] += [ 'K_{}-base_spkm'.format(k) for k in range(6,7) ]

cfg['outName'] = '../results/nyuEval'

paramBase = {'spkm':np.array([2,3,4,5,6,7,8,9]), 'DPvMFmeans':np.array([np.cos(lamb*np.pi/180.0)-1. for lamb in [70.,80.,90.,100.,110.,115.,120.,130.,140.,150.]])}
# TODO waiting for 70, 80, 140, 150

paramName =  {'spkm':"$K$",'DPvMFmeans':"$\lambda$ [deg]"}
baseMap={'spkm':'spkm','kmeans':'k-means','NiwSphere':'DirSNIW', \
  'DpNiw':'DP-GMM','DpNiwSphere':'DpSNIW opt','DpNiwSphereFull':'DP-TGMM', \
  'DPvMFmeans':'DP-vMF-means'}

#cfg['base'] += [ 'K_{}-base_spkm'.format(k) for k in range(4,8) ]
cfg['T'] = 100

reIndex = False;
reIndex = True;

nFiles = 0
for base in cfg['base']:
  nFiles += len(paramBase[base])
print nFiles
# --------------------------- final cost function value -------------------- 
if reIndex:
  candidates = []
  for file in os.listdir(cfg['path']):
    if fnmatch.fnmatch(file, '*[0-9]_measures.csv'):
      candidates.append(file)
  print '# candidates = {}'.format(len(candidates))

  cfctFiles = []
  index = open('/data/vision/fisher/data1/nyu_depth_v2/index.txt')
  for i,name in enumerate(index):
    name =name[:-1]
    found = []; #[None for base in cfg['base']]
    for j,base in enumerate(cfg['base']):
      for K in paramBase[base]:
        if base == 'spkm':
          searchStr = '{}*K_{}-*{}*T_{}*lambda_{}_measures.csv'.format(name,K,base,cfg['T'],0.)
        else:
          searchStr = '{}*K_{}-*{}*T_{}*lambda_{}_measures.csv'.format(name,1,base,cfg['T'],K)
        for candidate in candidates:
          if fnmatch.fnmatch(candidate, searchStr):
            found.append(candidate)
            break
          
    if len(found) == nFiles : #found[0] is None and not found[1] is None:
      print found
      cfctFiles.append(found)
        
  with open('./cfctFiles.txt','w') as f:
    for cfctFile in cfctFiles:
      for cfctF in cfctFile:
        f.write(cfctF+'\n')
else:
  with open('./cfctFiles.txt','r') as f:
    cfctFiles =[]
    cfctFile = []
    while 42:
      cfctFile = []
      for i in range(nFiles):
        fil = f.readline()[:-1]
        if fil == '': 
          break
        else: 
          cfctFile.append(fil)
      if fil == "": 
        break
      cfctFiles.append(copy.deepcopy(cfctFile))

Sils = np.zeros((len(cfctFiles),nFiles))
Ks = np.zeros((len(cfctFiles),nFiles))
for i,cfctFile in enumerate(cfctFiles):
   for j,f in enumerate(cfctFile):
     print j,f
     if os.path.isfile(os.path.join(cfg['path'],f)):
       measure = np.loadtxt(os.path.join(cfg['path'],f))
       if measure.size > 0:
         Ks[i,j] = int(measure[0])
         Sils[i,j] = measure[1]
     else:
       raise ValueError
#print np.sum(cs == 0.0, axis=0)
#if np.sum(cs == 0.0) > 0:
#  print "warning there were zeros in the eval!"
print ' --------------------------------'
print 'number of scenes used for eval: {}'.format(Ks.shape[0])
print "Sils eval"
print 'mean',np.mean(Sils,axis=0)
print 'std',np.std(Sils,axis=0)

print "Ks eval"
print 'mean',np.mean(Ks,axis=0)
print 'std',np.std(Ks,axis=0)
print ' --------------------------------'

paramBase['DPvMFmeans'] = np.arccos(paramBase['DPvMFmeans'] + 1.)*180./np.pi
# mean number of clusters
values = {'DPvMFmeans': Ks[:,0:7].T, 'spkm': Ks[:,7::].T}
fig = plotOverParams(values,'K',paramBase,paramName,baseMap,showLeg=True,Ns=None)
plt.savefig(cfg['outName']+'_{}.pdf'.format(re.sub('\$','',"K")),figure=fig)
# silhouette plot
values = {'DPvMFmeans': Sils[:,0:7].T, 'spkm': Sils[:,7::].T}
fig = plotOverParams(values,'silhouette',paramBase,paramName,baseMap,showLeg=True,Ns=None)
plt.savefig(cfg['outName']+'_{}.pdf'.format(re.sub('\$','',"silhouette")),figure=fig)

colA = colorScheme('labelMap')['orange']
colB = colorScheme('labelMap')['turquoise']
idMax = np.argmax(np.mean(values['DPvMFmeans'],axis=1))

fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
# histogram over the number of clusters for all frames
plt.hist(Ks[:,idMax],bins=np.arange(0,Ks[:,idMax].max()+1)+.5, alpha =0.7,color=colB)
plt.xlim(2,Ks[:,idMax].max()+1)
#plt.plot(paramBase[base],vMeasures[base][:],label=baseMap[base],c=cl[(i+1)*255/I])
#plt.title("histogram over the number of clusters")
plt.xlabel('number of clusters')
plt.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(right=0.6,bottom=0.3)
plt.savefig(cfg['outName']+'_histNClusters.pdf',figure=fig)

plt.show()


fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(111)
ind = np.arange(nFiles)
width = 0.95
# histogram over the number of clusters for all frames
rects = ax.bar(ind, np.mean(Sils,axis=0), width, color=colB, alpha = 0.7)
for rect in rects[7::]:
  rect.set_color(colA)
(_,caps,_) = ax.errorbar(ind+width/2., np.mean(Sils,axis=0), np.std(Sils,axis=0), color=(0,0,0),fmt ='.', capsize=10)
for cap in caps:
  cap.set_markeredgewidth(4)
#plt.plot(paramBase[base],vMeasures[base][:],label=baseMap[base],c=cl[(i+1)*255/I])
ax.set_ylabel('silhouette')
ax.set_xticks(ind+width/2)
#70.,90.,100.,110.,115.,120.,130
ax.set_xticklabels(('DP-vMF-means 70','DP-vMF-means 90','DP-vMF-means 100','DP-vMF-means 110','DP-vMF-means 115','DP-vMF-means 120','DP-vMF-means 130','spkm $K=3$','spkm $K=4$','spkm $K=5$','spkm $K=6$','spkm $K=7$','spkm $K=8$','spkm $K=9$'),rotation=30)
plt.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(right=0.6,bottom=0.3)
plt.savefig(cfg['outName']+'_silhouette.pdf',figure=fig)
