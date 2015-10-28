# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import os
import pickle

from config import Config2String


class BufferedResult(object):
  '''
    buffers the result of a computation of function fct with data (dict) 
    in a local path (if nfs = False)
    in the global nfs path (if nfs = True)
  '''
  def __init__(s,path,fct,nfs=True):
    if nfs:
      s.p_wosp = os.getenv('WORKSPACE_HOME', '/home/jstraub/workspace/')
      s.p_scratch = os.getenv('SCRATCH_HOME', '/scratch/')
      s.p_data = os.getenv('DATA_HOME', './data/')
      s.p_results = os.getenv('RESULTS_HOME', './results/')
      s.path = s.p_results + path
    s.path = path
    s.fct = fct
  def get(s,inp,recompute=False):
    path = s.path + Config2String(inp).toString() + '.pickle'
    print 'BufferedResult:: get from '+path
    if os.path.isfile(path) and not recompute:
      val = pickle.load(open(path,'r'))
    else:
      print 'recomputing'
      val = s.fct(inp)
      pickle.dump(val,open(path,'w'))
    return val

class BufferedData(object):
  '''
    base class for large data objects that should
    be buffered on the hard disc since computing
    them everytime takes too long
    path: path relative to scratch space
  '''
  def __init__(s,path):
    s.path = os.getenv('SCRATCH_HOME', '.') + '/'+ path
    s.__load_or_create()
  def __load_or_create(s):
    if os.path.isfile(s.path):
      print 'BufferedData:: loading pickle from '+s.path
      s.X = pickle.load(open(s.path,'r'))
    else:
      s.X = s.create()
      print 'BufferedData:: dumping pickle to '+s.path
      pickle.dump(s.X, open(s.path,'w+'))
  def create(s):
    print "BufferedData:: overwrite this function"
    return None
  def recompute(s):
    s.X = s.create()
    pickle.dump(s.X, open(s.path,'w+'))
  def get(s):
    return s.X;

def loadCfg(cfg):
  '''
  loads a pickle file from path
  cfg has to contain a key 'path' that tells the function where to load the data from
  returns None if the file does not exist

  '''
  if 'path' in cfg:
    fullPath = cfg['path']+Config2String(cfg).toString()+'.pickle'
    if os.path.isfile(fullPath):
      print 'loading data from {}'.format(fullPath)
      X = pickle.load(open(fullPath,'r'))
      print 'data loaded from '+fullPath
      return X
    else:
      print 'no file found under '+fullPath
      return None
  else:
    print 'no key \'path\' in cfg'
    return None

def dumpCfg(X,cfg):
  '''
  dump X to a pickle file
  cfg has to contain a key 'path' that tells where to dump to
  '''
  if 'path' in cfg:
    fullPath = cfg['path']+Config2String(cfg).toString()+'.pickle'
    pickle.dump(X,open(fullPath,'w'))
    print 'data saved to '+fullPath
    return True
  else:
    print 'no key \'path\' in cfg'
    return False

