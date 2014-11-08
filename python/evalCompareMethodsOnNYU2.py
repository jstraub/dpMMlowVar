import numpy as np
import cv2
import os, copy
import fnmatch

cfg=dict()
cfg['path'] = '/data/vision/scratch/fisher/jstraub/dpMMlowVar/nyu2'
cfg['base'] = ['DpNiw' , 'DpNiwSphereFull', 'spkm'];
cfg['base'] = ['DPvMFmeans','spkm']; 
#cfg['base'] += [ 'K_{}-base_spkm'.format(k) for k in range(6,7) ]

baseKs = {'spkm':[k for k in range(6,7) ], 'DPvMFmeans':[1]}

#cfg['base'] += [ 'K_{}-base_spkm'.format(k) for k in range(4,8) ]
cfg['T'] = 100

reIndex = True;

nFiles = 0
for base in cfg['base']:
  nFiles += len(baseKs[base])
# --------------------------- final cost function value -------------------- 
if reIndex:
  cfctFiles = []
  index = open('/data/vision/fisher/data1/nyu_depth_v2/index.txt')
  for i,name in enumerate(index):
    name =name[:-1]
    found = []; #[None for base in cfg['base']]
    for file in os.listdir(cfg['path']):
      if fnmatch.fnmatch(file, '*[0-9]_measures.csv'):
        for j,base in enumerate(cfg['base']):
          for k,K in enumerate(baseKs[base]):
            if fnmatch.fnmatch(file, '{}*K_{}*{}*T_{}*[0-9]_measures.csv'.format(name,K,base,cfg['T'])):
              found.append(file)
#            print file
    if len(found) == nFiles : #found[0] is None and not found[1] is None:
      print found
      cfctFiles.append(found)
  with open('./cfctFiles.txt','w') as f:
    for cfctFile in cfctFiles:
      f.writelines(cfctFile)
else:
  with open('./cfctFiles.txt','r') as f:
    cfctFiles =[]
    cfctFile = []
    while 42:
      cfctFile = []
      for base in cfg['base']:
        fil = f.readline()[:-1]
        if fil == '': 
          break
        else: 
          cfctFile.append(fil)
      if fil == "": 
        break
      cfctFiles.append(copy.deepcopy(cfctFile))

Sils = np.zeros((len(cfctFiles),len(cfg['base'])))
Ks = np.zeros((len(cfctFiles),len(cfg['base'])))
for i,cfctFile in enumerate(cfctFiles):
   for j,f in enumerate(cfctFile):
     print f
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
print "Sils eval"
print np.mean(Sils,axis=0)
print np.std(Sils,axis=0)

print "Ks eval"
print np.mean(Ks,axis=0)
print np.std(Ks,axis=0)

exit(0);
# --------------------------- images -------------------------------------
for i,name in enumerate(index):
#  print name
  name =name[:-1]
  found = [None for base in cfg['base']]
  for file in os.listdir(cfg['path']):
    if not fnmatch.fnmatch(file, '*lbls.png'):
      continue
#    print file
    for j,base in enumerate(cfg['base']):
#      print  '{}*{}*.png'.format(name,base)
      if fnmatch.fnmatch(file, '{}*{}-*.png'.format(name,base)):
        found[j] = file
#        print file
  if not any(f is None for f in found): #found[0] is None and not found[1] is None:
    print found
    I = cv2.imread(os.path.join(cfg['path'],found[0]))
    for f in found[1::]:
      I = np.r_[I,cv2.imread(os.path.join(cfg['path'],f))]
#    print found
#    I0 = cv2.imread(os.path.join(cfg['path'],found[0]))
#    I1 = cv2.imread(os.path.join(cfg['path'],found[1]))
#    print I0.shape
#    print np.c_[I0,I1].shape
#    print np.r_[I0,I1].shape
    cv2.imshow(' vs. '.join(cfg['base']),I)
    cv2.waitKey()
