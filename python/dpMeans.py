# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import numpy as np

def normed(x):
  # return unit L2 length vectors
  return x / np.sqrt((x**2).sum(axis=0)) 

class DPmeans(object):
  def __init__(self, lamb):
    self.lamb = lamb
  def removeCluster(self,k):
    self.mu = np.concatenate((self.mu[:,:k],self.mu[:,k+1::]),axis=1)
    self.N_ = np.concatenate((self.N_[:k],self.N_[k+1::]),axis=1)
    self.z[self.z>k] -= 1
    self.K -= 1

  def labelAssign(self,i,x_i):
    z_i = np.argmin(np.r_[np.sqrt(((x_i - self.mu)**2).sum(axis=0)),np.array([self.lamb])])
    # check if this was the last datapoint in a cluster if so do not
    # assign to it again
    if self.N_[self.z[i]] == 0 and z_i == self.z[i]:
      self.removeCluster(self.z[i])
      z_i = np.argmin(np.r_[np.sqrt(((x_i - self.mu)**2).sum(axis=0)),np.array([self.lamb])])
    # creata a new cluster if required
    if z_i == self.K:
      self.mu = np.concatenate((self.mu,x_i),axis=1)
      self.N_ = np.concatenate((self.N_,np.array([0])),axis=0)
      self.K += 1
    return z_i

  def compute(self,x,Tmax=100):
    # init stuff
    N = x.shape[1]
    self.K = 1
    self.z = np.zeros(N,dtype=np.int) # labels for each data point
    self.mu = x[:,0][:,np.newaxis] # the first data point always creates a cluster
    self.N_ = np.bincount(self.z,minlength=self.K) # counts per cluster
    self.C = np.zeros(Tmax) # cost function value
    self.C[0] = 1e6
    for t in range(1,Tmax):
      # label assignment
      for i in range(N):
        self.N_[self.z[i]] -= 1 
        self.z[i] = self.labelAssign(i,x[:,i][:,np.newaxis])
        self.N_[self.z[i]] += 1
      # centroid update
      for k in range(self.K-1,-1,-1):
        if self.N_[k] > 0:
          self.mu[:,k] = (x[:,self.z==k].sum(axis=1))/self.N_[k]
        else:
          self.removeCluster(k)
      # eval cost function
      self.C[t] = np.array(\
          [np.sqrt(((x[:,i][:,np.newaxis]-self.mu[:,z_i][:,np.newaxis])**2).sum(axis=0)) for i,z_i in enumerate(self.z)]).sum() \
          + self.K*self.lamb
      print 'iteration {}:\tcost={};\tcounts={}'.format(t,self.C[t], self.N_)
      if self.C[t] >= self.C[t-1]:
        break;

# generate two noisy clusters
N = 10
x = np.concatenate(\
    ((np.random.randn(3,N/2).T*0.1+np.array([1,0,0])).T,\
     (np.random.randn(3,N/2).T*0.1+np.array([0,1,0])).T),axis=1)
print x.shape
# instantiate DP-means algorithm object
dpmeans = DPmeans(lamb = 0.4)
# compute clustering (maximum of 30 iterations)
dpmeans.compute(x,Tmax=30)

