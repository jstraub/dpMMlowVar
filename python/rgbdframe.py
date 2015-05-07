# Licensed for research purposes only. See the license file LICENSE.txt
#
# If this code is used, the following should be cited:
# Straub, Julian, Guy Rosman, Oren Freifeld, John Leonard, and John Fisher. "A
# mixture of Manhattan frames: Beyond the Manhattan world." In Proceedings of
# the IEEE Conference on Computer Vision and Pattern Recognition, pp.
# 3770-3777. 2013.

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2 
from scipy.linalg import norm

import scipy.io as scio
import os,re,pdb

from buffered import BufferedResult

import mayavi.mlab as mlab

from sphere import Sphere

def normalize(I):
  return (I-np.min(I))/(np.max(I)-np.min(I))

class RgbdFrame(object):
  '''
  handles RGB-D frames. 
  - loading from .png files
  - loading from .mat files
  - computation of the normals
  - computation of gradients of gray scale image
  - display of point cloud and normals
  - display of rgb image
  '''
  def __init__(s,f_rgb,f_d=None):
    s.f_rgb = f_rgb
    if f_d is not None:
      s.f_d = f_d
    else:
      s.f_d = f_rgb
    s.__reset()
  def __reset(s):
    '''
    called once a new frame is loaded to invalidate 
    all extracted information from the previous frame
    '''
    s.rgb_dx = np.zeros(1)
    s.pc = np.zeros(1)
    s.n = np.zeros(1)
  def load(s,path):
    path = re.sub("_rgb","",path)
    s.path = path
    print s.path
    if os.path.isfile(path+'.mat'):
      print 'loading from '+path+'.mat'
      mat = scio.loadmat(path+'.mat')
      print mat.keys()
      s.d = mat['d']
      cv2.imwrite(path+'_d.png',s.d.astype(np.uint16))
      s.rgb= mat['rgb']
      print 'rgb shape {}, type={}'.format(s.rgb.shape,s.rgb.dtype)

      s.rgb[:,:,0] = mat['rgb'][:,:,0]
      s.rgb[:,:,1] = mat['rgb'][:,:,1]
      s.rgb[:,:,2] = mat['rgb'][:,:,2]
      cv2.imwrite(path+'_0_rgb.png',s.rgb)
      s.rgb= mat['rgb']
      s.rgb[:,:,0] = mat['rgb'][:,:,1]
      s.rgb[:,:,1] = mat['rgb'][:,:,0]
      s.rgb[:,:,2] = mat['rgb'][:,:,2]
      cv2.imwrite(path+'_1_rgb.png',s.rgb)
      s.rgb= mat['rgb']
      s.rgb[:,:,0] = mat['rgb'][:,:,0]
      s.rgb[:,:,1] = mat['rgb'][:,:,2]
      s.rgb[:,:,2] = mat['rgb'][:,:,1]
      cv2.imwrite(path+'_2_rgb.png',s.rgb)
      s.rgb= mat['rgb']
      s.rgb[:,:,0] = mat['rgb'][:,:,2]
      s.rgb[:,:,1] = mat['rgb'][:,:,1]
      s.rgb[:,:,2] = mat['rgb'][:,:,0]
      cv2.imwrite(path+'_rgb.png',s.rgb)
    if os.path.isfile(path+'_d.png'):
      print 'loading from '+path+'_d.png'
      s.d = cv2.imread(path+'_d.png',cv2.CV_LOAD_IMAGE_UNCHANGED)
    else:
      print 'error reading depth file from '+path+'_d.png'
    if os.path.isfile(path+'_rgb.png'):  
      print 'loading from '+path+'_rgb.png'
      s.rgb = cv2.imread(path+'_rgb.png')
    else:
      print 'error reading rgb file from '+path+'_rgb.png'
    print s.rgb.shape
    s.gray = cv2.cvtColor(s.rgb, cv2.COLOR_BGR2GRAY)
    print s.gray.shape
    s.mask = s.d > 0.0
    s.mask3d = s.mask.copy()
    s.mask3d = np.resize(s.mask3d,(s.rgb.shape))
    s.__reset()
  def getPc(s):
    # 3d points
    if s.pc.size == 1:
      s.pc = np.zeros((s.d.shape[0],s.d.shape[1],3))
      U,V = np.meshgrid(np.arange(-s.d.shape[1]/2.,s.d.shape[1]/2.),np.arange(-s.d.shape[0]/2.,s.d.shape[0]/2.)) 
      U += 0.5
      V += 0.5
      s.pc[:,:,2] = s.d*0.001
      s.pc[:,:,0] = s.pc[:,:,2]*U/s.f_d
      s.pc[:,:,1] = s.pc[:,:,2]*V/s.f_d
    return s.pc
  def getRgbGrad(s,kernelSize=9):
    if s.rgb_dx.size == 1:
      # gradient extraction form gray image
      s.rgb_dx = cv2.Sobel(s.gray,cv2.CV_32F,dx=1,dy=0,ksize=kernelSize)
      s.rgb_dy = cv2.Sobel(s.gray,cv2.CV_32F,dx=0,dy=1,ksize=kernelSize)
      s.rgb_E = np.sqrt(s.rgb_dx**2+s.rgb_dy**2) 
      s.rgb_phi = np.arctan2(s.rgb_dy,s.rgb_dx)
    return s.rgb_dx, s.rgb_dy, s.rgb_E, s.rgb_phi
  def getNormals(s,renormalize=False,algo=None,path=None):
    if algo is None:
      algo = 'sobel'
    if s.n.size == 1:
      if path is None:
        path = s.path
      head,tail=os.path.split(path)
      inp = {'name':tail , 'type':'normals', 'd':s.d, 'f':s.f_d, 'method':algo, 'path':path}
      normals = BufferedResult(head+'/',_getNormals,False)
      if 'n' in normals.get(inp).keys():
        s.n = normals.get(inp)['n']
      elif 'u_registered' in normals.get(inp).keys():
        s.n = normals.get(inp)['u_registered']
      else:
        print normals.get(inp).keys()
    if renormalize:
      norm = np.sqrt(s.n[:,:,0]**2 + s.n[:,:,1]**2 + s.n[:,:,2]**2)
      s.n[:,:,0] /= norm
      s.n[:,:,1] /= norm
      s.n[:,:,2] /= norm
    return s.n
  def showRgbd(s,fig=None):
    cv2.imshow('rgb',s.rgb)
    if fig is None:
      fig = plt.figure()
    plt.imshow(s.d,interpolation='nearest',cmap = cm.jet)
    fig.show()
  def showRgbGrad(s):
    s.getRgbGrad()
    cv2.imshow('dx and dy; absolute value of gradient, angle of gradient', \
      np.r_[np.c_[normalize(s.rgb_dx),normalize(s.rgb_dy)],
            np.c_[normalize(s.rgb_E),normalize(s.rgb_phi)]])
  def showPc(s,figm=None,showNormals=False,algo='sobel',color=(1,0,0)):
    s.getPc()
    if figm is None:
      figm = mlab.figure(bgcolor=(1,1,1))
    mlab.points3d(s.pc[s.mask,0],s.pc[s.mask,1],s.pc[s.mask,2],
        s.gray[s.mask],colormap='gray',scale_factor=0.01,
        figure=figm,mode='point',mask_points=1)
    if showNormals:
      s.getNormals(algo)
      mlab.quiver3d(s.pc[s.mask,0],s.pc[s.mask,1],s.pc[s.mask,2],
          s.n[s.mask,0],s.n[s.mask,1],s.n[s.mask,2], 
          figure=figm, mode='2darrow',line_width=1.0, 
          color=color, scale_factor=0.1,mask_points=50)
    return figm
  def showNormals(s,figm=None,algo='sobel',color=(1,0,0),as2D=False):
    s.getNormals(algo)
    if as2D:
      if figm is None:
        figm = plt.figure()
      I = (s.n.copy()+1.0)*0.5 # make non negative
      I[np.logical_not(s.mask),0] = 0.;
      I[np.logical_not(s.mask),1] = 0.;
      I[np.logical_not(s.mask),2] = 0.;
      plt.imshow(I)
    else:
      if figm is None:
        figm = mlab.figure(bgcolor=(1,1,1)) 
      M = Sphere(2)
      M.plotFanzy(figm,1.0,linewidth=1)
      mlab.points3d(s.n[s.mask,0],s.n[s.mask,1],s.n[s.mask,2],
          scale_factor=0.01, color=color, figure=figm, 
          mode='point',mask_points=1)
    return figm
  def showWeightedNormals(s,theta=30.0,figm=None,algo='sobel'):
    # according to http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6375037
    # weighting is 1.0/\sigma_z
    s.sigma_z = 0.0012 + 0.019*(s.d*0.001-0.4)**2
    ang = theta/180.0*np.pi
    s.sigma_l_px = 0.8 + 0.035*ang/(np.pi/2.0-ang)
    s.sigma_l = s.sigma_l_px * s.d*0.001/s.f_d
    s.w = 1.0/(s.sigma_z**2+2*s.sigma_l**2)
    s.getNormals(algo)
    if figm is None:
      figm = mlab.figure(bgcolor=(0,0,0)) 
    mlab.points3d(s.n[s.mask,0],s.n[s.mask,1],s.n[s.mask,2],s.w[s.mask],
        scale_factor=0.01, figure=figm, mode='point',mask_points=1)
    mlab.colorbar(orientation='vertical')
    
    return figm
  #def showWeightedNormals(s,figm=None,algo='sobel'):
  def showAxialSigma(s,fig=None):
    # according to http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6375037
    s.sigma_z = 0.0012 + 0.019*(s.d*0.001-0.4)**2
    if fig is None:
      fig = plt.figure()
    plt.title('axial sigma')
    plt.imshow(s.sigma_z)
    plt.colorbar()
    fig.show()
    return fig
  def showLateralSigma(s,theta=30.0,fig=None):
    # according to http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6375037
    ang = theta/180.0*np.pi
    s.sigma_l_px = 0.8 + 0.035*ang/(np.pi/2.0-ang)
    s.sigma_l = s.sigma_l_px * s.d*0.001/s.f_d
    if fig is None:
      fig = plt.figure()
    plt.title('lateral sigma')
    plt.imshow(s.sigma_l)
    plt.colorbar()
    fig.show()
    return fig
  def bilateralDepthFiltering(s,theta=30.0,fig=None):
    # according to http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6375037
    s.z = s.d*0.001
    s.z[s.d==0] = np.nan
    s.z[:,0] = np.nan
    s.sigma_z = 0.0012 + 0.019*(s.z-0.4)**2
    ang = theta/180.0*np.pi
    s.sigma_l_px = 0.8 + 0.035*ang/(np.pi/2.0-ang)
    s.sigma_l = s.sigma_l_px * s.z/s.f_d
    s.z_latFilt = np.zeros_like(s.z)
    thresh = 0.1
    for u in range(2,s.d.shape[1]-2):
      for v in range(2,s.d.shape[0]-2):
        if s.z[v,u] == 0.0:
          continue
        zw = 0.0
        w = 0.0
        for u_n in range(-1,2):
          for v_n in range(-1,2):
            #if u_n == 0 and v_n ==0:
            #  continue
            dp = np.sqrt(u_n**2 + v_n**2)
            dz = abs(s.z[v,u]-s.z[v+v_n,u+u_n])
#            if dz>0.0:
#              print 'dp={}; dz={}'.format(dp,dz)
            if dz < thresh:
              #print '{} {}'.format(dp**2/(s.sigma_l_px**2), dz**2/(s.sigma_z[v+v_n,u+u_n]**2))
              dw = np.exp(-0.5*(dp**2/(s.sigma_l_px**2)+dz**2/(s.sigma_z[v+v_n,u+u_n]**2)))
              w += dw
              zw += s.z[v+v_n,u+u_n]*dw
        if w > 0:
          s.z_latFilt[v,u] = zw/w
    if fig is None:
      fig = plt.figure()
    plt.imshow(s.z_latFilt)
    plt.title('lateral filtered img')
    plt.colorbar()
    fig1 = plt.figure()
    plt.imshow(np.abs(s.z_latFilt-s.z),cmap=cm.hot)
    fig.show()
    fig1.show()
def _getNormals(inp):
  '''
  fct will compute the normals of a given depth image
  this is ment to be used with BufferedResult
  example input: 
  inp = {'d':s.d, 'f':s.f_d, 'method':'guy', 'path':path}
  returns a dict with only the normals
  '''
  algo = inp['method']
  d = inp['d']
  f = inp['f']
  path = inp['path']
  if algo == 'sobel' or algo == '':
    d_dx = cv2.Sobel(d,cv2.CV_32F,dx=1,dy=0,ksize=5)
    d_dy = cv2.Sobel(d,cv2.CV_32F,dx=0,dy=1,ksize=5)
    dx = np.zeros((d.shape[0]*d.shape[1],3))
    dx[:,0] = np.ones(dx.shape[0])*f
    dx[:,2] = d_dx.ravel()
    dy = np.zeros((d.shape[0]*d.shape[1],3))
    dy[:,1] = np.ones(dy.shape[0])*f
    dy[:,2] = d_dy.ravel()
  
    n=np.cross(dx,dy)
    for i in xrange(0,d.shape[0]*d.shape[1]):
      n[i,:] /= norm(n[i,:],2)
      #n[i,:] *= n[i,2]
    #n = n.T
    n = np.reshape(n,(d.shape[0],d.shape[1],3))
    #TODO: this will not give a normal image but a list of normals
    #pickle.dump(n,open(path+'_normals_sobel.pickle','w'))
  elif algo == 'guy':
    os.system('matlab -nodesktop -nodisplay -r "addpath(genpath(\'../matlab\')); compute_normals_from_file(\''+path+'_d.png\'); exit"')
    mat=scio.loadmat(path+'_d.png.normals.mat')
    n = mat['regularized_u']
    print 'n.shape={}'.format(n.shape)
    #pickle.dump(n,open(path+'_normals_guy.pickle','w'))

  # make sure the normals are norm one before handing outside 
  for v in range(n.shape[0]):
    for u in range(n.shape[1]):
      n[v,u,:] /= norm(n[v,u,:])
  return {'n':n}


