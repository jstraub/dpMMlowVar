import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.io
import cv2
import subprocess as subp

import os, re, time
import argparse

from rgbdframe import RgbdFrame
from sphere import Sphere
from pyplot import SaveFigureAsImage

def run(cfg,reRun):
  #args = ['../build/dpSubclusterSphereGMM',
#  args = ['../build/dpStickGMM',
  args = ['../build/bin/dpMMlowVarCluster',
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

def config2Str(cfg):
  use = ['K','base','T','delta','nu','lambda']
  st = use[0]+'_'+str(cfg[use[0]])
  for key in use[1::]:
    st += '-'+key+'_'+str(cfg[key])
  return st

parser = argparse.ArgumentParser(description = 'DpMM modeling and viewer')
parser.add_argument('-s','--start', type=int, default=0, help='start image Nr')
parser.add_argument('-e','--end', type=int, default=0, help='end image Nr')
parser.add_argument('-K0', type=int, default=1, help='initial number of MFs')
parser.add_argument('-l','--lamb', type=float, default=90., help='lambda parameter in degree for DPvMFmeans')
parser.add_argument('-b','--base', default='DPvMFmeans', help='base distribution/algorithm')
parser.add_argument('-nyu', action='store_true', help='switch to process the NYU dataset')
args = parser.parse_args()

cfg=dict()
cfg['rootPath'] = '/home/jstraub/workspace/research/vpCluster/data/nyu2/'
cfg['rootPath'] = '/home/jstraub/workspace/research/vpCluster/data/'
cfg['rootPath'] = '~/workspace/research/vpCluster/data/'
cfg['outputPath'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2/'
cfg['rootPath'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2/'
cfg['rootPath'] = '/data/vision/fisher/data1/nyu_depth_v2/extracted/'



#cfg['base'] = 'DpNiwSphereFull';
#cfg['base'] = 'spkm';
#cfg['base'] = 'DpNiw';
cfg['base'] = args.base;
#cfg['base'] = 'DPvMFmeans';
#cfg['base'] = 'spkm';
cfg['K'] = args.K0
cfg['T'] = 500 # max number of iterations
cfg['delta'] = 12. #18.
cfg['nu'] =   3 + 10000.0 
if cfg['base'] == 'DPvMFmeans':
  cfg['lambda'] = np.cos(args.lamb*np.pi/180.0)-1.
else:
  cfg['lambda'] = 0.0
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
reRun = True
printCmd = True

if args.nyu:
  mode = ['multiFromFile']

if 'single' in mode:
  cfg['outputPath'] = '../data/'
  cfg['rootPath'] = '../data/'
  cfg['dataPath'] = '2013-09-27.10:33:47' #
  cfg['dataPath'] = '2013-10-01.19:25:00' # my room
  cfg['dataPath'] = 'living_room_0000'
  cfg['dataPath'] = 'study_room_0004_uint16'
  cfg['dataPath'] = 'study_room_0005_uint16'
  cfg['dataPath'] = 'home_office_0001_uint16'
  cfg['dataPath'] = '2boxes_1'
  cfg['dataPath'] = 'kitchen_0004'
  cfg['dataPath'] = 'office_0008_uint16'
  cfg['dataPath'] = 'table_1'
  cfg['dataPath'] = 'kitchen_0016_252'
  cfg['lambda'] = np.cos(45.*np.pi/180.0)-1.
  cfg['dataPath'] = '3boxes_moreTilted_0' #segments really well - has far distance!!! [k=4]
  cfg['lambda'] = np.cos(100.*np.pi/180.0)-1.
  cfg['dataPath'] = 'MIT_hallway_1'
  cfg['dataPath'] = 'MIT_hallway_0'
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

rndInds = range(len(names)) # np.random.permutation(len(names))
for ind in rndInds:
  cfg['dataPath'] = names[ind]
  if 'N' in cfg.keys():
#    import ipdb
#    ipdb.set_trace()
    del cfg['N']
    del cfg['D']

  cfg['outName'] = cfg['outputPath']+cfg['dataPath']+'_'+config2Str(cfg)
  if not reRun and 'multiFromFile' in mode and os.path.isfile(cfg['outName']+'_measures.csv'):
    print '  ** skipping '+cfg['outName']+' since it is already existing'
#    continue;
  if not reRun and os.path.isfile(cfg['outName']+'_measures.csv'):
    measures = np.loadtxt(cfg['outName']+'_measures.csv')
    if not measures.size == 2:
      reRun = True
  if not reRun and not os.path.isfile(cfg['outName']+'_measures.csv'):
    print '  ** rerun False but '+cfg['outName'] + '_measures.csv not existing => run inference!'
    reRun = True;

  print 'processing '+cfg['rootPath']+cfg['dataPath']
  rgbd = RgbdFrame(460.0) # correct: 540
  rgbd.load(cfg['rootPath']+cfg['dataPath'])
  if False and 'disp' in mode:
    rgbd.showRgbd(fig=fig0)
  rgbd.getPc()
  nAll = rgbd.getNormals(algo=algo)
  n = nAll[rgbd.mask,:].T
  cfg['D'] = n.shape[0]
  cfg['N'] = n.shape[1]
  print cfg
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
  print "image to {}".format(cfg['outName']+'lbls.png')
  SaveFigureAsImage(cfg['outName']+'lbls.png',figL)

  # compute blended image gray with overlayed segmentation
  figLrgb = plt.figure()
  Iz = np.zeros(rgbd.mask.shape)
  Iz[rgbd.mask] = z[-1,:]
  Iz = np.floor(Iz * 255./Iz.max())
  Iz = Iz.astype(np.uint8)
  Iz = cv2.applyColorMap(Iz,cv2.COLORMAP_JET)
  gray = np.zeros((rgbd.mask.shape[0],rgbd.mask.shape[1],3), dtype=np.uint8)
  for i in range(3): 
    Iz[:,:,i][np.logical_not(rgbd.mask)] = 255
    gray[:,:,i] = cv2.cvtColor( rgbd.rgb, cv2.COLOR_BGR2GRAY)
#  cv2.imshow("Iz",Iz)
#  cv2.waitKey(0)
  Iout = cv2.addWeighted(gray , 0.7, Iz,0.3,0.0)
  plt.imshow(Iout,figure = figLrgb)
  print "image to {}".format(cfg['outName']+'lblsRgbBlend.png')
  SaveFigureAsImage(cfg['outName']+'lblsRgbBlend.png',figLrgb)

  if 'disp' in mode:
#    figL.show()
    figLrgb.show()
#    figm1 = rgbd.showNormals(as2D=True); 
#    figm1.show()
#    plt.show()

#    plt.show()
#    figm2 = rgbd.showWeightedNormals(algo=algo)
#    fig = rgbd.showAxialSigma()
#    fig = rgbd.showLateralSigma(theta=30.0)
    #fig = rgbd.bilateralDepthFiltering(theta=30.0)
#    figm0 = rgbd.showPc(showNormals=True,algo=algo)
#    figm1 = rgbd.showNormals()


    # show raw normals
    figm1 = mlab.figure(bgcolor=(1,1,1))
    M = Sphere(2)
    M.plotFanzy(figm1,1.0) 
    from js.utils.plot.colors import colorScheme
    rgbd.n[2,:] *= -1.
    figm1=rgbd.showNormals(figm=figm1,color=colorScheme("labelMap")["orange"])
    # show clustered normals
    figm3 = mlab.figure(bgcolor=(1,1,1))
    M = Sphere(2)
    M.plotFanzy(figm1,1.0) 
    mlab.points3d(n[0,:],n[1,:],n[2,:],-z[-1,:],colormap='jet',
            mode='point')
#    for k in range(K):
#      ids = z[-1,:]==k
#      if np.count_nonzero(ids) > 0:
#        mlab.points3d(n[0,ids],n[1,ids],n[2,ids],color=((float(k)/K),0,0),
#            mode='point')
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


