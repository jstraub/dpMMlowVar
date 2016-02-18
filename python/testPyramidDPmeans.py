
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from js.data.rgbd.rgbdframe import RgbdFrame
from dpMeansWeighted import DPmeansWeighted
import ipdb 
import mayavi.mlab as mlab

rgbd = RgbdFrame(540.)
rgbd.load("../data/MIT_hallway_0_d.png")
#rgbd.showRgbd()
#plt.show()

pc = rgbd.getPc()
H, W, _ = pc.shape
pc = np.concatenate((pc,np.ones((H,W,1))), axis=2)
print pc.shape
lamb0 = 0.4

runComparison = False
if runComparison:
  dpmeans = DPmeansWeighted(lamb0, verbose=True)
  t0 = time.time()
  dpmeans.compute(np.reshape(pc, (H*W,4)).T, Tmax=1000)
  print "dt={}s".format(t0 - time.time())
  raw_input()

x = [[] for i in range(6)]
C = np.zeros(6)
t0 = time.time()
for pyr in range(5,-1,-1):
  h, w = int(H*2**(-pyr)), int(W*2**(-pyr))
  lamb = lamb0*2**(-pyr)
  print pyr, h, w, lamb
  if pyr == 5:
    for v in range(0, H, h):
      for u in range(0, W, w):
        x_uv = np.reshape(pc[v:v+h,u:u+w], (h*w,4)).T
        x[pyr].append(x_uv)
  # cluster at the current pyr level
  
  for v in range(0, H/h):
    for u in range(0, W/w):
      x_uv = x[pyr][v*(W/w)+u]
#      print x_uv[3,:]
      dpmeans = DPmeansWeighted(lamb, verbose=pyr<2)
      cost = dpmeans.compute(x_uv, Tmax=30)
      C[pyr] += cost - dpmeans.K*lamb + dpmeans.K*lamb0
      x[pyr][v*(W/w)+u] = dpmeans.mu
      print u,v, lamb, cost, dpmeans.K
#      print x_uv.shape
#      print dpmeans.N_
  if pyr > 0:
    # merge 2x2 neighborhoods for input into next pyr level
    print "merging outputs of level {} into level {}".format(pyr, pyr-1)
    h, w = int(H*2**(-pyr+1)), int(W*2**(-pyr+1))
    for v in range(0, H/h):
      for u in range(0, W/w):
        x[pyr-1].append(np.concatenate((
          x[pyr][v*2*(2*W/w)+u*2],
          x[pyr][(v*2+1)*(2*W/w)+u*2],
          x[pyr][v*2*(2*W/w)+u*2+1],
          x[pyr][(v*2+1)*(2*W/w)+u*2+1]
          ), axis=1))
#       print u,v, x[pyr-1][-1]
  else:
    mus = x[0][0]
print "dt={}s".format(time.time() - t0)
print "costs ", C

D = np.zeros((mus.shape[1], mus.shape[1]))
for i,muA in enumerate(mus.T):
  for j,muB in enumerate(mus.T):
    if i==j:
      D[i,j] = np.nan
    else:
      D[i,j] = np.sqrt(((muA[:3] - muB[:3])**2).sum())
print "D min - max: {} - {}".format(np.min(D[np.logical_not(np.isnan(D))]),
    np.max(D[np.logical_not(np.isnan(D))]))
fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(D, interpolation="nearest")
plt.colorbar()
plt.subplot(1,2,2)
plt.hist(D[np.logical_not(np.isnan(D))])
plt.show()

print mus.shape
rgbd.showPc()
mlab.points3d(mus[0,:], mus[1,:], mus[2,:], scale_factor=0.05,
    color=(0,1,0), mode="sphere", opacity=0.5)
mlab.show()
dpmeans = DPmeansWeighted(lamb0)
dpmeans.compute(mus, Tmax=30)
mus = dpmeans.mu
print mus.shape
mlab.points3d(mus[0,:], mus[1,:], mus[2,:], scale_factor=0.05,
    color=(1,0,0), mode="sphere", opacity=0.5)
mlab.show()



