
import numpy as np
from numpy import sin,cos
from scipy.linalg import  norm, det
#import ipdb

import mayavi.mlab as mlab

class Sphere:
  def __init__(s,D):
    s.D = D
    s.north = np.zeros(D)
    s.north[-1] = 1.0;

  def verify_an_arrow_belongs_to_TpM(self,p,vec):
    return
    # TODO: for more than one vec
    if p.shape[0] != self.N+1:
      raise ValueError(p.shape[0],self.N+1)
    if vec.shape[0] != self.N+1:
      raise ValueError(vec.shape,self.N+1)
    if not np.allclose(np.dot(p.ravel(),vec.ravel()),0,atol=1e-01):
      raise ValueError(np.dot(p.ravel(),vec.ravel()))

  def verify_pt_belongs_to_the_sphere(self,p):
    return
    if len(p) != self.N+1:
      raise ValueError(p)
    if not np.allclose(norm(p),1,atol=1e-01):
      raise ValueError(norm(p))

  def Exp_p(s,p,vecs_in_Tp):
    '''
    maps vectors in the tangent space of p onto the sphere
    '''
    s.verify_pt_belongs_to_the_sphere(p)
    s.verify_an_arrow_belongs_to_TpM(p,vecs_in_Tp)
  
    theta = np.sqrt((vecs_in_Tp**2).sum(axis=0))
    if theta == 0: # sinc case
      return p + vecs_in_Tp
    else:
      return p*cos(theta) + vecs_in_Tp/theta * sin(theta)

  def Log_p(self,p,q):
    """
    maps a point q on the sphere into the tangent space at p
    This assumes p and q are on the sphere.
    """
    # TODO: put this onto GPU
    if q.shape[0] != p.shape[0]:
      q = q.T
    # need to do it this way because of row mayor
    P = np.resize(p,(q.shape[1],p.shape[0])).T
    # dot product
    a = np.dot(p,q)
    # angles
    theta = np.arccos(a)
    # compute sinc
    sinc = theta/np.sin(theta)
    # make sure that the division 0.0/0.0 = 1.0 (due to sinc)
    sinc[np.isnan(sinc)] = 1.0
    return (q-P*a)*sinc

  def LogTo2D(s,p,q):
    '''
     map points on sphere onto the tangent space and 
     express them as a 2D vector there.
     rotates all points into the xy plane!
    '''
    return s.Log_p_north(p,q)

  def Log_p_north(s,p,q):
    '''
     map points on sphere onto the tangent space and 
     express them as a 2D vector there.
     rotates all points into the xy plane!
    '''
#    if q.shape[1] ==0:
#      return np.zeros((3,0))

    # north = R \dot p
    # R will rotate points into the xy plane
    northRTpS = rotationFromAtoB(p,s.north)
    # map points on sphere onto the tangent space and 
    # express them as a 2D vector there
    xp = np.dot(northRTpS, s.Log_p(p,q))
    return xp[0:-1,:]

  def rotateToP(s,p,x):
    '''
    from a 2D vector x in the tangent space around p get the eucledian vector
    '''
    north = np.array([0.0,0.0,1.0])
    # north = R \dot p
    # R will rotate points into the xy plane
    R = rotFromTwoVectors(p,north)
    xx = np.zeros(3)
    xx[0:2] = x
    return R.T.dot(xx)

#  def geodesic_distance(self,p,q):
#    return np.arccos(max(-1.0,min(1.0,np.dot(p.ravel(),q.ravel()))))

  def plotCov(s,fig,S,mu,scale=1.0):
    V,D = np.linalg.eig(S)
    D = D[:,np.argsort(V)]
    V = np.sort(V)
#    ipdb.set_trace()
    if S.shape[0] == s.D:
      D = D[1::,1::] # just take the largest eigen vectors
      V = V[1::]
    D = np.r_[D,np.zeros((1,2))] # add z=0 coordinate
  
    north = np.array([0.0,0.0,1.0])
    # north = R \dot mu
    # R will rotate points into the xy plane
    R = rotFromTwoVectors(north,mu)
    # rotate D to mu
    D = np.dot(R,D)
    pts = np.zeros((3,4))
    for i in range(0,2):
      pts[:,i*2]   = mu - scale*np.sqrt(V[i])*D[:,i]
      pts[:,i*2+1] = mu + scale*np.sqrt(V[i])*D[:,i] 
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=(1.0,0.0,0.0))
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=(0.0,1.0,0.0))

  def mesh(s,r):
    # Create a sphere
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:45j, 0:2*pi:45j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    return (x,y,z)

  def plot(s,figm,r,color=(0.5,0.5,0.5),N=30,linewidth=3):
    X,Y,Z = s.mesh(r)
    mlab.mesh(X,Y,Z,color=color, opacity=1.0, figure=figm)
#    s.plotFanzy(figm,r,N,linewidth)
#    X,Y,Z = s.mesh(r)
#    mlab.mesh(X,Y,Z,color=(0.5,0.5,0.5), opacity=0.5, figure=figm, representation='wireframe' )

  def plotFanzy(s,figm,r,N=30,linewidth=3):
    # adapted from http://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html
    # The number of points per line
    #N = 30
    # The scalar parameter for each line
    az = np.linspace(0.0, 2.0*np.pi, N)
    el = np.linspace(0.0, np.pi, N)
    # We create a list of positions and connections, each describing a line.
    # We will collapse them in one array before plotting.
    x = list()
    y = list()
    z = list()
    connections = list()
    # The index of the current point in the total amount of points
    index = 0
    # now plot longitudes
    for i in range(N):
        x.append(r*np.sin(el)*np.cos(az[i]))
        y.append(r*np.sin(el)*np.sin(az[i]))
        z.append(r*np.cos(el))
        # This is the tricky part: in a line, each point is connected
        # to the one following it. We have to express this with the indices
        # of the final set of points once all lines have been combined
        # together, this is why we need to keep track of the total number of
        # points already created (index)
        connections.append(np.vstack(
                           [np.arange(index,   index + N - 1.5),
                            np.arange(index+1, index + N - .5)]
                                ).T)
        index += N
    # now plot latitudes
    for i in range(N):
        x.append(r*np.sin(el[i])*np.cos(az))
        y.append(r*np.sin(el[i])*np.sin(az))
        z.append(r*np.cos(el[i])*np.ones(az.shape) )
        # This is the tricky part: in a line, each point is connected
        # to the one following it. We have to express this with the indices
        # of the final set of points once all lines have been combined
        # together, this is why we need to keep track of the total number of
        # points already created (index)
        connections.append(np.vstack(
                           [np.arange(index,   index + N - 1.5),
                            np.arange(index+1, index + N - .5)]
                                ).T)
        index += N
    
    # Now collapse all positions, scalars and connections in big arrays
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    connections = np.vstack(connections)
    
    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z)
    
    # Connect them
    src.mlab_source.dataset.lines = connections
    
    # The stripper filter cleans up connected lines
    lines = mlab.pipeline.stripper(src)
    
    # Finally, display the set of lines
    mlab.pipeline.surface(lines, color=(0.5,0.5,0.5), opacity=0.5, 
        line_width=linewidth,figure=figm)

class Quaternion(object):
  def __init__(s,r=None):
    if r is None:
      s.q = np.array([1.0,0.0,0.0,0.0])
    else:
      if r.shape[0] == 4:
        s.q = r
      elif r.shape[0] ==3 and r.shape[1]==3:
        s.q = s.__rot2Quat(r)
      else:
        s.q = np.array([1.0,0.0,0.0,0.0])

  def __rot2Quat(s,R):
    '''
    Not time to do that
    '''
    return np.array([1.0,0.0,0.0,0.0])
 
  def angleTo(s,b):
    ''' 
    angle to another quaternion
    '''
    return np.arccos(2.0*(s.q.dot(b.q))**2-1.0)

  def toRot(s):
    '''
    quat2Rot: convert from a quaternion representation q of a rotation into a 
    rotation matrix representation R (S03); q has to be a column vector; 
    '''
    a = s.q[0]
    b = s.q[1]
    c = s.q[2]
    d = s.q[3]
    R = np.array([[ a**2+b**2-c**2-d**2 , 2*b*c-2*a*d ,2*b*d+2*a*c ]  ,
         [2*b*c+2*a*d        , a**2-b**2+c**2-d**2 , 2*c*d-2*a*b        ],
         [2*b*d-2*a*c        , 2*c*d+2*a*b        , a**2-b**2-c**2+d**2]]) 
#    R=np.array([[1-2*q[1]**2-2*q[2]**2, 2*q[0]*q[1]-2*q[2]*q[3], 2*q[0]*q[2]+2*q[1]*q[3]],
#       [2*q[0]*q[1]+2*q[2]*q[3], 1-2*q[0]**2-2*q[2]**2, 2*q[1]*q[2]-2*q[0]*q[3]], 
#       [2*q[0]*q[2]-2*q[1]*q[3], 2*q[1]*q[2]+2*q[0]*q[3], 1-2*q[0]**2-2*q[1]**2]]); 
    return R

  def toAxisAngle(s):
    #NOTE: right-hand-rule rotation!
    theta=-2.*np.arccos(s.q[0]);
    k=s.q[1:4]/np.sin(theta/2.)
    return k,theta

  def sample(s,dtheta,sigma):
    # radial symmetry
    q = np.random.randn(4)
     # step 2: A point on S^3 defines a quaternion. 
    q /= norm(q)
    # sample rotation angle
    theta = (np.random.randn(1)*sigma**2+dtheta)*np.pi/180.0
    theta =((theta+np.pi)%(2.0*np.pi))-np.pi
    
    q[0] = np.cos(theta/2.0)
    q[1:4] /= norm(q[1:4])#(q[1:3]**2).sum()
    q[1:4] *= np.sin(theta/2.0)

#    print 'q={}'.format(q)
#    print 'norm(q) = {}'.format(norm(q))
#    print theta
#    print np.arccos(q[0])*2
#    print np.arcsin(norm(q[1:4]))*2
#    print (q**2).sum()
    s.q = q
    return s.q

  def sampleUnif(s,dtheta=2.0*np.pi):
    # radial symmetry
    q = np.random.randn(4)
     # step 2: A point on S^3 defines a quaternion. 
    q /= norm(q)
    # sample rotation angle
    theta = (np.random.rand(1)-0.5)*dtheta #(np.random.randn(1)*sigma**2+dtheta)*np.pi/180.0
    #theta =((theta+np.pi)%(2.0*np.pi))-np.pi
    
    q[0] = np.cos(theta/2.0)
    q[1:4] /= norm(q[1:4])#(q[1:3]**2).sum()
    q[1:4] *= np.sin(theta/2.0)
    s.q = q
    return s.q

  def sample1stQuad(s,dtheta=0.5*np.pi):
    # radial symmetry
    q = np.random.randn(4)
     # step 2: A point on S^3 defines a quaternion. 
    q /= norm(q)
    q = np.abs(q)
    # sample rotation angle
    theta = (np.random.rand(1)-0.5)*dtheta #(np.random.randn(1)*sigma**2+dtheta)*np.pi/180.0
    #theta =((theta+np.pi)%(2.0*np.pi))-np.pi
    
    q[0] = np.cos(theta/2.0)
    q[1:4] /= norm(q[1:4])#(q[1:3]**2).sum()
    q[1:4] *= np.sin(theta/2.0)
    s.q = q
    return s.q

def testQuaternion():
  dtheta = 30.0
  quat = Quaternion()
  print quat.q
  print quat.toRot()
  print det(quat.toRot())

  figm = mlab.figure(bgcolor=(1,1,1))
  for i in range(100):
    print quat.sampleUnif(0.5*np.pi)
    k,theta = quat.toAxisAngle()
    print theta*180.0/np.pi
    plotCosy(figm, quat.toRot())

  figm = mlab.figure(bgcolor=(1,1,1))
  for i in range(100):
    print quat.sample(dtheta)
    k,theta = quat.toAxisAngle()
    print theta*180.0/np.pi
    plotCosy(figm, quat.toRot())

  figm1 = mlab.figure(bgcolor=(1,1,0.0))
  for i in range(100):
    # sample rotation axis
    k = np.random.rand(3)-0.5
    # sample uiniformly from +- 5 degrees
    theta =  (np.asscalar(np.random.rand(1)) + dtheta - 0.5) *np.pi/180.0 # (np.a      sscalar(np.random.rand(1))-0.5)*np.pi/(180.0/(2.0*dtheta))
    print 'perturbation: {} theta={}'.format(k/norm(k),theta*180.0/np.pi)
    dR = RodriguesRotation(k/norm(k),theta)
    plotCosy(figm1, dR)
    
  mlab.show()

def veeInv(k):
  return np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
def vee(R):
  # according to appendix of http://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0CC0QFjAA&url=http%3A%2F%2Fwww.researchgate.net%2Fpublication%2F220146411_Metrics_for_3D_Rotations_Comparison_and_Analysis%2Ffile%2F60b7d51f39f278a9d2.pdf&ei=1OPWUpTFNYLksATwroJw&usg=AFQjCNEAFpqLFkYqyJP7y_F0WYOyA0eHzQ&sig2=_JOcbCFKvBSPGkySlq-i9w&bvm=bv.59378465,d.cWc&cad=rja

  theta = np.arccos(max(-1.0,min(1.0,(np.trace(R)-1.0)/2.0)))
  if np.sin(theta) < 1e-10:
    S=np.eye(3)
  else:
    S = (R-R.T)/(2.0*np.sin(theta))
  axis = np.array([-S[1,2],S[0,2],-S[0,1]])
  return axis*theta

def RodriguesRotation(k_,theta=None):
  '''
  obtain rotation matrix using Rodrigues Formula
  input is either a single k vector (axis of rotation) with norm(k) = theta
  or k with norm(k) = 1.0 and an angle theta
  '''
  k = np.copy(k_)
  if theta is None:
    theta = norm(k)
    k /= theta
  k_cross = veeInv(k)
  return np.eye(3) + np.sin(theta)* k_cross + (1.0-np.cos(theta))*np.dot(k_cross,k_cross)

def testRodrigues():
  dRs = []
  dRs.append(RodriguesRotation(np.array([1.0,0.0,0.0]),np.pi/2.0))
  dRs.append(RodriguesRotation(np.array([1.0,0.0,0.0]),-np.pi/2.0))
  dRs.append(RodriguesRotation(np.array([1.0,0.0,0.0]),np.pi))
  dRs.append(RodriguesRotation(np.array([0.0,1.0,0.0]),np.pi/2.0))
  dRs.append(RodriguesRotation(np.array([0.0,1.0,0.0]),-np.pi/2.0))
  dRs.append(RodriguesRotation(np.array([0.0,1.0,0.0]),np.pi))
  dRs.append(RodriguesRotation(np.array([0.0,0.0,1.0]),np.pi/2.0))
  dRs.append(RodriguesRotation(np.array([0.0,0.0,1.0]),-np.pi/2.0))
  dRs.append(RodriguesRotation(np.array([0.0,0.0,1.0]),np.pi))
  fig1 = mlab.figure()
  for dR in dRs:
    print '-------'
    print dR
    print det(dR)
    plotCosy(fig1,dR)
  mlab.show()

def rotationFromAtoB(a, b):
#  ipdb.set_trace()
  assert(b.size == a.size);
  D = b.size;
  bRa = np.eye(D);
   
  dot = b.dot(a);
#  print  "dot={}".format(dot);
  if(np.fabs(dot -1.) < 1e-6):
    bRa = np.eye(D);
  elif(np.fabs(dot +1.) <1e-6):
    bRa = -np.eye(D);
  else:
    alpha = np.arccos(min(1.0,max(-1.0,dot)));
    c = a - b*dot;
    c /= norm(c);
    bRa = np.eye(D) + np.sin(alpha)*(np.outer(b,c) - np.outer(c,b)) +  \
      (np.cos(alpha)-1.)*(np.outer(b,b) + np.outer(c,c))
  return bRa

def rotFromTwoVectors(a,b):
  '''
  find rotation R such that b=R*a
  '''
  theta = np.arccos(max(-1.0,min(1.0,np.dot(a.ravel(),b.ravel())/(norm(a)*norm(b)))))
  if abs(theta) < 1e-6:
    print 'theta very small !!! returning identity matrix!'
    return np.eye(3)
  elif abs(theta-np.pi) <1e-6:
    print 'theta very close to pi !!! returning aribtrary 180deg rotation matrix!'
    # sample randomm point
    c = np.random.randn(3)
    # make sure that c is not in 0 or 180 degree to a
    while abs(np.dot(c,a)) == 1.0:
      c = np.random.randn(3)
    # get arbitrary rotation axis
    k = np.cross(a.ravel(),c.ravel())
    k /= norm(k)
    # and rotate by theta
    return RodriguesRotation(k,theta)
  else:
    k = np.cross(a.ravel(),b.ravel())
    k /= norm(k)
    return RodriguesRotation(k,theta)

def plotCov3D(fig,S,mu,scale=1.0):
  V,D = np.linalg.eig(S)
#  n = 100
#  # Compute the points on the surface of the ellipse.
#  t = np.linspace(0, 2*np.pi, n);

  pts = np.zeros((3,6))
  for i in range(0,3):
    pts[:,i*2]   = mu - scale*np.sqrt(V[i])*D[:,i]
    pts[:,i*2+1] = mu + scale*np.sqrt(V[i])*D[:,i] 
  mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=(1.0,0.0,0.0))
  mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=(0.0,1.0,0.0))
  mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=(0.0,0.0,1.0))
  
def plotCosy(fig,R,col=None):
  pts = np.zeros((3,6))
  for i in range(0,3):
    pts[:,i*2]  = np.zeros(3)
    pts[:,i*2+1] = R[:,i] 
  if col is None:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=(1.0,0.0,0.0))
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=(0.0,1.0,0.0))
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=(0.0,0.0,1.0))
  else:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=col)
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=col)
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=col)

def plotMF(fig,R,col=None):
  mfColor = []
  mfColor.append((232/255.0,65/255.0,32/255.0)) # red
  mfColor.append((32/255.0,232/255.0,59/255.0)) # green
  mfColor.append((32/255.0,182/255.0,232/255.0)) # tuerkis
  pts = np.zeros((3,6))
  for i in range(0,3):
    pts[:,i*2]  = -R[:,i] 
    pts[:,i*2+1] = R[:,i] 
  if col is None:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=mfColor[0])
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=mfColor[1])
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=mfColor[2])
  else:
    mlab.plot3d(pts[0,0:2],pts[1,0:2],pts[2,0:2],figure=fig,color=col)
    mlab.plot3d(pts[0,2:4],pts[1,2:4],pts[2,2:4],figure=fig,color=col)
    mlab.plot3d(pts[0,4:6],pts[1,4:6],pts[2,4:6],figure=fig,color=col)


if __name__=="__main__":
  print '----- test of vee() function - theta should be equal to x'
  R0 = np.eye(3)
  for x in np.linspace(0.0,3.14,100):
    R1 = np.array([[np.cos(x),-np.sin(x),0],
                   [np.sin(x),np.cos(x),0],
                   [0,0,1]])
    theta = norm(vee(R0.dot(R1.T)),2)
    print 'x={}; angle={}'.format(x,theta)
  
    figm = mlab.figure()
    plotCosy(figm,np.eye(3))
    for i in range(10):
      plotCosy(figm,Quaternion(Quaternion().sample(0.,4)).toRot())
    mlab.show(stop=True)
