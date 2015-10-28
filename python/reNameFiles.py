# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import os.path
import re
import ipdb
import fnmatch
import subprocess as subp

cfg = dict()
cfg['path'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2/'

for file in os.listdir(cfg['path']):
  if not re.search('-N_[0-9]+',file) is None:
#  if True or fnmatch.fnmatch(file, '*[0-9]_measures.csv'):
    foutName = re.sub('-N_\d+','',file)
#    print foutName

    args=['mv ',cfg['path']+file,cfg['path']+foutName]
    print ' '.join(args)
#    ipdb.set_trace()

    subp.call(' '.join(args),shell=True)
