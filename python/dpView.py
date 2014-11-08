import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import scipy.io
import subprocess as subp

import os, re, time
import argparse

from vpCluster.rgbd.rgbdframe import RgbdFrame
from vpCluster.manifold.sphere import Sphere
from js.utils.config import Config2String
from js.utils.plot.pyplot import SaveFigureAsImage

def run(cfg,reRun):
  #args = ['../build/dpSubclusterSphereGMM',
#  args = ['../build/dpStickGMM',
  args = ['../build/dpMMlowVarCluster',
    '--seed {}'.format(int(time.time()*100000) - 100000*int(time.time())),
    '-N {}'.format(cfg['N']), #TODO: read N,D from file!
    '-D {}'.format(cfg['D']),
    '-K {}'.format(cfg['K']),
    '-T {}'.format(cfg['T']),
    '--base '+cfg['base'],
    '-i {}'.format(cfg['rootPath']+cfg['dataPath']+'_normals.csv'),
    '-o {}'.format(cfg['outName']+'.lbl'),
    '--shuffle',
    '--silhouette']
  if cfg['base'] == 'DPvMFmeans':
    args.append('--params {}'.format(cfg['lambda']))

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

parser = argparse.ArgumentParser(description = 'DpMM modeling and viewer')
parser.add_argument('-s','--start', type=int, default=0, help='start image Nr')
parser.add_argument('-e','--end', type=int, default=0, help='end image Nr')
parser.add_argument('-K0', type=int, default=1, help='initial number of MFs')
parser.add_argument('-b','--base', default='DPvMFmeans', help='base distribution/algorithm')
parser.add_argument('-nyu', action='store_true', help='switch to process the NYU dataset')
args = parser.parse_args()

cfg=dict()
cfg['rootPath'] = '/home/jstraub/workspace/research/vpCluster/data/nyu2/'
cfg['rootPath'] = '/home/jstraub/workspace/research/vpCluster/data/'
cfg['rootPath'] = '~/workspace/research/vpCluster/data/'
cfg['outputPath'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/'
cfg['rootPath'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2/'
#cfg['base'] = 'DpNiwSphereFull';
#cfg['base'] = 'spkm';
#cfg['base'] = 'DpNiw';
cfg['base'] = args.base;
#cfg['base'] = 'DPvMFmeans';
#cfg['base'] = 'spkm';
cfg['K'] = args.K0
cfg['T'] = 100
cfg['delta'] = 12. #18.
cfg['nu'] =   3 + 10000.0 
if cfg['base'] == 'DPvMFmeans':
  cfg['lambda'] = np.cos(90*np.pi/180.0)-1.
#if cfg['base'] == 'DPvMFmeans':
#  cfg['lambDeg'] = 30.
#  cfg['T'] = 10
#if cfg['base'] == 'spkm':
#  cfg['T'] = 10
#  cfg['K'] = 12

seed = 214522
algo = 'guy' #'sobel'#'guy'
#mode = ['multi']
mode = ['multiFromFile']
mode = ['multi']
mode = ['single','disp']

reRun = False
printCmd = True

if args.nyu:
  mode = ['multiFromFile']

if 'single' in mode:
  cfg['dataPath'] = '2013-09-27.10:33:47' #
  cfg['dataPath'] = '2013-10-01.19:25:00' # my room
  cfg['dataPath'] = 'living_room_0000'
  cfg['dataPath'] = 'study_room_0004_uint16'
  cfg['dataPath'] = 'study_room_0005_uint16'
  cfg['dataPath'] = 'home_office_0001_uint16'
  cfg['dataPath'] = '2boxes_1'
  cfg['dataPath'] = 'kitchen_0004'
  cfg['dataPath'] = 'office_0008_uint16'
  cfg['dataPath'] = '3boxes_moreTilted_0' #segments really well - has far distance!!! [k=4]
  cfg['dataPath'] = 'table_1'
  cfg['dataPath'] = 'kitchen_0016_252'
  names = [cfg['dataPath']]
elif 'multi' in mode:
  names = []
  for root,dirs,files in os.walk(cfg['rootPath']):
    for f in files:
      ff = re.split('_',f)
      if ff[-1] == 'd.png':
        names.append('_'.join(ff[0:-1]))
        print 'adding {}'.format(names[-1])
##  names = ['home_office_0001_358','3boxes_moreTilted_0','couches_0','MIT_hallway_1','stairs_5','office_0008_17','stairs_5','MIT_hallway_0','kitchen_0007_132'] 
  names = ['kitchen_0015_252', 'living_room_0058_1301', 'bedroom_0085_1084', 'kitchen_0033_819', 'conference_room_0002_342', 'kitchen_0048_879']
elif 'multiFromFile' in mode:
  cfg['evalStart'] = args.start
  cfg['evalEnd'] = args.end
  indexPath = '/data/vision/fisher/data1/nyu_depth_v2/index.txt'
  cfg['rootPath'] = '/data/vision/fisher/data1/nyu_depth_v2/extracted/'
  cfg['outputPath'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2/'
  names =[]
  with open(indexPath) as f:
    allNames = f.read().splitlines() #readlines()
  for i in range(len(allNames)):
    if cfg['evalStart'] <= i and i <cfg['evalEnd']:
      names.append(allNames[i])
      print '@{}: {}'.format(len(names)-1,names[-1])
  print names
else:
  print 'no files in list'
  exit(1)

if 'disp' in mode:
  import mayavi.mlab as mlab
#  figm0 = mlab.figure(bgcolor=(1,1,1))
#  figm1 = mlab.figure(bgcolor=(1,1,1))
  fig0 = plt.figure()

rndInds = np.random.permutation(len(names))
for ind in rndInds:
  cfg['dataPath'] = names[ind]

  cfg['outName'] = cfg['outputPath']+cfg['dataPath']+'_'+Config2String(cfg).toString()
  if not reRun and 'multiFromFile' in mode and os.path.isfile(cfg['outName']+'_measures.csv'):
    print 'skipping '+cfg['outName']+' since it is already existing'
    continue;
  if not reRun and os.path.isfile(cfg['outName']+'_measures.csv'):
    measures = np.loadtxt(cfg['outName']+'_measures.csv')
    if not measures.size == 2:
      reRun = True
  if not reRun and not os.path.isfile(cfg['outName']+'_measures.csv'):
    print 'rerun False but '+cfg['outName'] + '_measures.csv not existing => run inference!'
    reRun = True;
  
  print 'processing '+cfg['rootPath']+cfg['dataPath']
  rgbd = RgbdFrame(460.0) # correct: 540
  rgbd.load(cfg['rootPath']+cfg['dataPath'])
  if 'disp' in mode:
    rgbd.showRgbd(fig=fig0)
  rgbd.getPc()
  nAll = rgbd.getNormals(algo=algo)
  n = nAll[rgbd.mask,:].T
  cfg['D'] = n.shape[0]
  cfg['N'] = n.shape[1]
  dataPath = cfg['rootPath']+cfg['dataPath']+'_normals.csv'
  np.savetxt(dataPath,n)

  z, sil = run(cfg,reRun)
  K = (np.max(z[-1,:])+1)
  np.savetxt(cfg['outName']+'_measures.csv',np.array([K,sil]));
  print 'measures for eval saved to {}'.format(cfg['outName']+'_measures.csv')

  figL = plt.figure()
  I = np.zeros(rgbd.mask.shape)
  I[rgbd.mask] = z[-1,:] + 1
  plt.imshow(I,cmap=cm.spectral,figure = figL)
#  plt.imshow(I,cmap=cm.hsv,figure = figL)
  SaveFigureAsImage(cfg['outName']+'lbls.png',figL)

  if 'disp' in mode:
    figL.show()
#    plt.show()
#    figm2 = rgbd.showWeightedNormals(algo=algo)
#    fig = rgbd.showAxialSigma()
#    fig = rgbd.showLateralSigma(theta=30.0)
    #fig = rgbd.bilateralDepthFiltering(theta=30.0)
#    figm0 = rgbd.showPc(showNormals=True,algo=algo)
#    figm1 = rgbd.showNormals()
    figm2 = rgbd.showNormals(as2D=True); figm2.show()
    M = Sphere(2)
    figm1 = mlab.figure()
    M.plotFanzy(figm1,1.0) 
    mlab.show(stop=True)
  elif  'multiFromFile' in mode and 'disp' in mode:
    figm0 = rgbd.showPc(figm=figm0,showNormals=True,algo=algo)
    figm1 = rgbd.showNormals(figm=figm1)
    M = Sphere(2)
    M.plotFanzy(figm1,1.0) 
    mlab.show(stop=True)
    mlab.clf(figm0)
    mlab.clf(figm1)
  
  plt.close(figL)

