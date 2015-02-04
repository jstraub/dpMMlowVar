import numpy as np
import subprocess as subp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mayavi.mlab as mlab

from matplotlib.patches import Ellipse

import ipdb, re
import os.path
import time, copy

from js.utils.plot.colors import colorScheme
from js.utils.config import Config2String

from vpCluster.manifold.karcherMean import karcherMeanSphere_propper
from vpCluster.manifold.sphere import Sphere

from plots import plotOverParams

def mutualInfo(z,zGt):
  ''' assumes same number of clusters in gt and inferred labels '''
  N = float(z.size)
  Kgt = int(np.max(zGt)+1)
  K = int(np.max(z)+1)
  print Kgt, K
  mi = 0.0
  Ndata = np.bincount(z,minlength=K).astype(np.float)
  NGts = np.bincount(zGt,minlength=Kgt).astype(np.float)
  Nsqs = np.bincount(z*Kgt+zGt,minlength=Kgt*K).astype(np.float)
  for j in range(K):
    for k in range(Kgt):
#      indzj = z==j
#      indzGtk = zGt==k
#      Njk = np.count_nonzero(np.logical_and(indzj,indzGtk))
#      Nj = np.count_nonzero(indzj)
#      Nk = np.count_nonzero(indzGtk)
      Njk = Nsqs[j*Kgt+k]
      Nj = Ndata[j]
      Nk = NGts[k]
      if Njk > 0:
#        print '{} {} {} {} {} -> += {}'.format(N, Njk,Nj,Nk, N*Njk/(Nj*Nk), Njk/N * np.log(N*Njk/(Nj*Nk)))
        mi += Njk/N * np.log(N*Njk/(Nj*Nk))
  return mi
def entropy(z):
  ''' assumes same number of clusters in gt and inferred labels '''
  N = float(z.size)
  K = int(np.max(z)+1)
  H = 0.0
  Ndata = np.bincount(z,minlength=K).astype(np.float)
  for k in range(K):
#    Nk = np.count_nonzero(z==k)
    if Ndata[k] > 0:
#        print '{} {} {} {} {} -> += {}'.format(N, Njk,Nj,Nk, N*Njk/(Nj*Nk), Njk/N * np.log(N*Njk/(Nj*Nk)))
      H -= Ndata[k]/N * np.log(Ndata[k]/N)
  return H

def run(cfg,reRun):
  #args = ['../build/dpSubclusterSphereGMM',
#  args = ['../build/dpStickGMM',
  if cfg['base'] in ['CrpvMF']:
    args = ['../../dpMMshared/build/dpmmSampler', '--alpha 1.0']
    params = np.r_[ cfg['m0'], np.array([cfg['t0'],cfg['a0'],cfg['b0']])]
  else:
    params = np.array([cfg['lambda']])
    args = ['../build/dpMMlowVarCluster',
        '--shuffle',
        '--silhouette']
  args = args + ['--seed {}'.format(int(time.time()*100000) - 100000*int(time.time())),
    '-N {}'.format(N), #TODO: read N,D from file!
    '-D {}'.format(D),
    '-K {}'.format(cfg['K']),
    '-T {}'.format(cfg['T']),
    #'--base NiwSphereUnifNoise',
    '--base '+cfg['base'],
    '-i {}'.format(cfg['rootPath']+cfg['dataPath']),
    '-o {}'.format(cfg['outName']+'.lbl'),
    '--params '+' '.join([str(p) for p in params])]

  if reRun:
    print ' '.join(args)
    print ' --------------------- '
    time.sleep(1)
    err = subp.call(' '.join(args),shell=True)
    if err:
      print 'error when executing'
      raw_input()
  z = np.loadtxt(cfg['outName']+'.lbl',dtype=int,delimiter=' ')
  sil = np.loadtxt(cfg['outName']+'.lbl_measures.csv',delimiter=" ")
  return z,sil

cfg = dict()
cfg['rootPath'] = rootPath = '../results/'
print cfg

dataPath = './rndSphereData.csv';
dataPath = './rndSphereDataIw.csv';
dataPath = './rndSphereDataElipticalCovs.csv';
dataPath = './rndSphereDataElipticalCovs1.csv';
dataPath = './rndSphereDataElipticalCovs2.csv'; # 10k datapoints with 30 classes
dataPath = './rndSphereDataElipticalCovs3.csv'; # 10k datapoints with 30 classes

# for final eval
dataPath = './rndSphereDataElipticalCovs4.csv'; # 10k datapoints with 30 classes less spread
dataPath = './rndSphereDataIwUncertain.csv';

# rebuttal
dataPath = './rndSphereDataNu9D20.csv';
dataPath = './rndSphereNu9D20N30000.csv';
dataPath = './rndSphereDataNu29D20N30000.csv';
dataPath = './rndSphereDataNu25D20N30000.csv';
dataPath = './rndSphereDataNu26D20N30000.csv';

dataPath = './rndSphereDataIwUncertain.csv'; # still works well in showing advantage of DpNiwSphereFull
dataPath = './rndSphereDataElipticalCovs4.csv'; # 10k datapoints with 30 classes less spread


# aistats resubmission
dataPath = './rndSphereDataIwUncertain.csv'; # still works well in showing advantage of DpNiwSphereFull
dataPath = './rndSphereDataNu25D3N30000NonOverLap.csv' # a few anisotropic clusters
dataPath = './rndSphereDataNu10D3N30000NonOverLap.csv' # very isotropic
dataPath = './rndSphereminAngle_15.0-K_30-N_30000-delta_4.0-nu_3.001-D_3.csv'
dataPath = './rndSphereminAngle_15.0-K_30-N_30000-delta_100.0-nu_21.0-D_20.csv'
dataPath = '././rndSphereminAngle_15.0-K_30-N_30000-delta_30.0-nu_21.0-D_20.csv'
dataPath = './rndSphereminAngle_15.0-K_30-N_30000-delta_30.0-nu_21.0-D_20.csv'
dataPath = './rndSphereminAngle_10.0-K_60-N_60000-delta_30.0-nu_21.0-D_20.csv'
dataPath = '././rndSphereminAngle_10.0-K_60-N_60000-delta_25.0-nu_21.0-D_20.csv'

# DP-vMF-means
dataPath = './rndSphereDataElipticalCovs4.csv'; # 10k datapoints with 30 classes less spread

dataPath = './rndSphereDataIwUncertain.csv';
cfg['nParms'] = 50;
paramBase = {'spkm':np.floor(np.linspace(70,10,cfg['nParms'])).astype(int), # 60,2
  'DPvMFmeans':np.array([ang for ang in np.linspace(10.,45.,cfg['nParms'])])}
cfg['T'] = 100
cfg['nRun'] = 50

dataPath = './rndSphereDataNu10D3N30000NonOverLap.csv' # very isotropic
cfg['nParms'] = 50;
paramBase = {'spkm':np.floor(np.linspace(70,10,cfg['nParms'])).astype(int), # 60,2
  'DPvMFmeans':np.array([ang for ang in np.linspace(5.,45.,cfg['nParms'])]),
  'CrpvMF':np.array([5.0])}
cfg['T'] = 100
cfg['nRun'] =  50

cfg['dataPath'] = dataPath

if os.path.isfile(re.sub('.csv','_gt.lbl',rootPath+dataPath)):
  zGt = np.loadtxt(re.sub('.csv','_gt.lbl',rootPath+dataPath),dtype=int,delimiter=' ')
  Kgt = np.max(zGt)+1
else:
  print "groundtruth not found"

# aistats resubmission
bases = ['DPvMFmeans']
# params for the different al5os
bases = ['spkm','DPvMFmeans']
bases = ['spkm']
bases = ['CrpvMF']
bases = ['DPvMFmeans','spkm']

paramName =  {'spkm':"$K$",'DPvMFmeans':"$\phi_\lambda$ [deg]"}
baseMap={'spkm':'spkm','kmeans':'k-means','NiwSphere':'DirSNIW', \
  'DpNiw':'DP-GMM','DpNiwSphere':'DpSNIW opt','DpNiwSphereFull':'DP-TGMM', \
  'DPvMFmeans':'DP-vMF-means'}

print paramBase

x=np.loadtxt(rootPath+dataPath,delimiter=' ')
N = x.shape[1]
D = x.shape[0]

reRun = True
reRun = False

if reRun:
  print cfg
  raw_input("are you sure you want to rerun??")

mis = {'spkm':np.zeros((len(paramBase['spkm']),cfg['nRun'])), 'DPvMFmeans':np.zeros((len(paramBase['DPvMFmeans']),cfg['nRun']))}
nmis = {'spkm':np.zeros((len(paramBase['spkm']),cfg['nRun'])), 'DPvMFmeans':np.zeros((len(paramBase['DPvMFmeans']),cfg['nRun']))}
vMeasures = {'spkm':np.zeros((len(paramBase['spkm']),cfg['nRun'])), 'DPvMFmeans':np.zeros((len(paramBase['DPvMFmeans']),cfg['nRun']))}
Ns = {'spkm':np.zeros((len(paramBase['spkm']),cfg['nRun'])), 'DPvMFmeans':np.zeros((len(paramBase['DPvMFmeans']),cfg['nRun']))}
Sils = {'spkm':np.zeros((len(paramBase['spkm']),cfg['nRun'])), 'DPvMFmeans':np.zeros((len(paramBase['DPvMFmeans']),cfg['nRun']))}

cfg0 = copy.deepcopy(cfg)
for i,base in enumerate(bases):
  cfg = copy.deepcopy(cfg0)
  cfg['base']=base
  cfg['outName'],_ = os.path.splitext(cfg['rootPath']+cfg['dataPath'])
  cfg['outName'] += '_'+Config2String(cfg).toString()
  print cfg['outName']
  if not reRun and os.path.exists('./'+cfg['outName']+'_MI.csv'):
    MI = np.loadtxt(cfg['outName']+'_MI.csv')
    Hgt = np.loadtxt(cfg['outName']+'_Hgt.csv')
    Hz = np.loadtxt(cfg['outName']+'_Hz.csv')
    Ns[base] = np.loadtxt(cfg['outName']+'_Ns.csv')
    Sils[base] = np.loadtxt(cfg['outName']+'_Sil.csv')
    print MI.shape, Hgt.shape, Hz.shape, Ns[base].shape
  else:
    MI = np.zeros((len(paramBase[base]),cfg['nRun']))
    Hz = np.zeros((len(paramBase[base]),cfg['nRun']))
    Hgt = np.zeros((len(paramBase[base]),cfg['nRun']))
    Sils[base] = np.zeros((len(paramBase[base]),cfg['nRun']))
    Ns[base] = np.zeros((len(paramBase[base]),cfg['nRun']))
    for j,param in enumerate(paramBase[cfg['base']]):
      if cfg['base'] == 'spkm':
        cfg['K'] = int(np.floor(param))
      else:
        cfg['K'] = 1
      if cfg['base'] == 'DPvMFmeans':
        cfg['lambda'] = np.cos(param*np.pi/180.0)-1. 
      else:
        cfg['lambda'] = 0.
      if cfg['base'] == 'CrpvMF':
        cfg['m0'] = np.zeros(D); cfg['m0'][0] = 1.
        cfg['t0'] = 0.01
        cfg['a0'] = 3.0
        cfg['b0'] = 2.7
      for t in range(cfg['nRun']):
        cfg['runId'] = t;
        z,measures = run(cfg,reRun)
        # compute MI and entropies - if not already computed and stored 
        Sils[base][j,t] = measures
        MI[j,t] = mutualInfo(z[-1,:],zGt)
        Hz[j,t] = entropy(z[-1,:])
        Hgt[j,t] = entropy(zGt)
        Ns[base][j,t] = int(np.max(z[-1,:])+1)
    np.savetxt(cfg['outName']+'_MI.csv',MI);
    np.savetxt(cfg['outName']+'_Hgt.csv',Hgt);
    np.savetxt(cfg['outName']+'_Hz.csv',Hz);
    np.savetxt(cfg['outName']+'_Sil.csv',Sils[base]);
    np.savetxt(cfg['outName']+'_Ns.csv',Ns[base]);

  mis[base] = MI
#  for t in range(cfg['T']):
#    for j in range(paramBase[base]):
  nmis[base] = MI / np.sqrt(Hz*Hgt)
  vMeasures[base] = 2.* MI / (Hz+Hgt)
  print mis[base].shape, nmis[base].shape, vMeasures[base].shape
#  print nmis[i,t], 2.*MI[t], Hz[t], Hgt[t]

print "done with the runs"

#cl = cm.gnuplot2(np.arange(len(bases)))
cl = cm.hsv(np.arange(255))
cl = cm.brg(np.arange(255))
cl = cm.gist_rainbow(np.arange(255))
cl = cm.gnuplot2(np.arange(255))
cl = cm.gnuplot(np.arange(255))
cl = cm.spectral(np.arange(255))
#print cltlib 
I = len(bases) +1

#fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
#for i,base in enumerate(bases):
#  plt.subplot(1,3,i+1)
#  plt.plot(paramBase[base],nmis[base][:,-1],label=baseMap[base],c=cl[(i+1)*255/I])
#  plt.title(base)
#  plt.xlabel(paramName[base])
#  plt.ylabel('NMI')
#  plt.ylim([0,1])
#  plt.legend(loc='lower right')
#plt.tight_layout()

#ipdb.set_trace()

indSpkm = np.ones(len(paramBase['spkm']),dtype=bool)
indSpkm[Ns['spkm'].mean(axis=1) < Ns['DPvMFmeans'].min()] = False
indSpkm[Ns['spkm'].mean(axis=1) > Ns['DPvMFmeans'].max()] = False

paramBase['spkm'] = paramBase['spkm'][indSpkm]
nmis['spkm'] = nmis['spkm'][indSpkm,:]
mis['spkm'] = mis['spkm'][indSpkm,:]
Ns['spkm'] = Ns['spkm'][indSpkm,:]
Sils['spkm'] = Sils['spkm'][indSpkm,:]

print paramBase

fig = plotOverParams(mis,'MI',paramBase,paramName,baseMap, Ns=Ns,showLeg=False)
plt.savefig(cfg['outName']+'_{}.pdf'.format(re.sub('\$','',"MI")),figure=fig)

fig = plotOverParams(nmis,'NMI',paramBase,paramName,baseMap,Ns=Ns,showLeg=False)
plt.savefig(cfg['outName']+'_{}.pdf'.format(re.sub('\$','',"NMI")),figure=fig)

fig = plotOverParams(Ns,'$K$',paramBase,paramName,baseMap,Ns=Ns,showLeg=True)
plt.savefig(cfg['outName']+'_{}.pdf'.format(re.sub('\$','',"K")),figure=fig)

fig = plotOverParams(Sils,'silhouette',paramBase,paramName,baseMap,Ns=Ns,showLeg=False)
plt.savefig(cfg['outName']+'_{}.pdf'.format(re.sub('\$','',"silhouette")),figure=fig)

plt.show()

#fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
#ax1 = plt.subplot(111)
#base = 'spkm'
#ax1.plot(Sils[base].mean(axis=1),paramBase[base],label=baseMap[base],c=cl[(0+1)*255/I])
#iKtrue = np.argmin(np.abs(Ns[base].mean()-30))
#ax1.plot(Sils[base].mean(axis=1)[iKtrue],paramBase[base][iKtrue],'x',label=baseMap[base]+' $K={}$'.format(Ns[base][iKtrue]),c=(1,0,0))
#ax1.set_ylabel(paramName[base])  
#ax1.invert_yaxis()
#ax1.set_xlabel("Sil")  
#ax1.legend(loc='best')
#ax2 = ax1.twinx()
#base = 'DPvMFmeans'
#ax2.plot(Sils[base].mean(axis=1),paramBase[base],label=baseMap[base],c=cl[(1+1)*255/I])
#iKtrue = np.argmin(np.abs(Ns[base].mean()-30))
#ax2.plot(Sils[base].mean(axis=1)[iKtrue],paramBase[base][iKtrue],'x',label=baseMap[base]+' $K={}$'.format(Ns[base][iKtrue]),c=(1,0,0))
#ax2.set_ylabel(paramName[base])  
#ax2.legend(loc='right')
#plt.tight_layout()
#plt.savefig(cfg['outName']+'_Sil.png',figure=fig)

#fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
#ax1 = plt.subplot(111)
#base = 'spkm'
#ax1.plot(Ns[base].mean(axis=1),paramBase[base],label=baseMap[base],c=cl[(0+1)*255/I])
#ax1.plot([30]*len(paramBase[base]), paramBase[base], label="true number of clusters $K=30$", c=(1,0,0))
#ax1.set_ylabel(paramName[base])  
#ax1.invert_yaxis()
#ax1.set_xlabel("number of clusters")  
#ax1.legend(loc='best')
#ax2 = ax1.twinx()
#base = 'DPvMFmeans'
#ax2.plot(Ns[base].mean(axis=1),paramBase[base],label=baseMap[base],c=cl[(1+1)*255/I])
#ax2.set_ylabel(paramName[base])  
#ax2.legend(loc='right')
#plt.tight_layout()
#plt.savefig(cfg['outName']+'_nClusters.png',figure=fig)


fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
for i,base in enumerate(bases):
  plt.subplot(1,2,i+1)
  plt.plot(paramBase[base],vMeasures[base][:],label=baseMap[base],c=cl[(i+1)*255/I])
  plt.title(base)
  plt.xlabel(paramName[base])
  plt.ylabel('vMeasure')
  plt.ylim([0,1])
  plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(cfg['outName']+'_VMeasure.png',figure=fig)

#fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
#for i,base in enumerate(bases):
#  plt.subplot(1,2,i+1)
#  plt.plot(paramBase[base],mis[base][:,-1],label=baseMap[base],c=cl[(i+1)*255/I])
#  plt.title(base)
#  plt.xlabel(paramName[base])
#  plt.ylabel('MI')
#  plt.legend(loc='lower right')
#plt.tight_layout()

fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
ax1 = plt.subplot(111)
base = 'spkm'
ax1.plot(mis[base][:],paramBase[base],label=baseMap[base],c=cl[(0+1)*255/I])
ax1.set_ylabel(paramName[base])  
ax1.invert_yaxis()
ax1.set_xlabel("MI")  
ax1.legend(loc='best')
ax2 = ax1.twinx()
base = 'DPvMFmeans'
ax2.plot(mis[base][:],paramBase[base],label=baseMap[base],c=cl[(1+1)*255/I])
ax2.set_ylabel(paramName[base])  
ax2.legend(loc='right')
plt.tight_layout()
plt.savefig(cfg['outName']+'_MI.png',figure=fig)

#fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
#for i,base in enumerate(bases):
#  plt.subplot(1,2,i+1)
#  plt.plot(paramBase[base],Ns[base][:,-1],label=baseMap[base],c=cl[(i+1)*255/I])
#  plt.title(base)
#  plt.xlabel(paramName[base])
#  plt.ylabel('number of clusters')
#  plt.legend(loc='best')
#plt.tight_layout()


plt.show()
