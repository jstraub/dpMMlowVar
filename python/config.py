import re
import json
import pickle
import copy


class Config2String:
  ''' Allows to convert dicts into strings and back
      This can be used to generate unique names for files from
      config dict
  '''
  def __init__(s,config):
    s.config = config

  def toString(s,nMax=9999):
    ''' 
        inside dicts are ignored 
        lists of strings work

        @param nMax: string values that are longer than nMax 
          are not put into the string
    '''
    cfg = copy.deepcopy(s.config) # make a deep copy
    for key, val in cfg.items():
      if not (isinstance(val,int) or isinstance(val,float) or isinstance(val,basestring) or isinstance(val,list)):
        cfg.pop(key,None)
        continue
      if isinstance(val,basestring):
        if len(val) > nMax or re.search('/',val) is not None:
          cfg.pop(key,None)
       
      if isinstance(val,list):
        valNew = ''
        for elem in val:
          valNew += str(elem) + '#'
        cfg[key] = valNew[0:-1]
    name = json.dumps(cfg)
    name = re.sub('{\"','',name)
    name = re.sub('}','',name)
    name = re.sub('\": ','_',name)
    name = re.sub(', \"','-',name)
    name = re.sub('\"','',name)
    return name

  def fromString(s,name):
    keyVals = re.split('-',name)
    for keyVal in keyVals:
      kv = re.split('_',keyVal)
      if len(kv) > 2:
        print "ERROR while parsing filename! keyVal={}\nkv={}".format(keyVal,kv)
      else:
        if kv[1] == 'true':
          s.config[kv[0]] = True
        elif kv[1] == 'false':
          s.config[kv[0]] = False
        elif re.match('^[1-9][0-9]*$',kv[1]): #ints
          s.config[kv[0]] = int(kv[1])
        elif re.match('^[0-9][.0-9]*$',kv[1]): #floats
          s.config[kv[0]] = float(kv[1])
        elif re.search('#',kv[1]): #lists
          s.config[kv[0]] = re.split('#',kv[1])
        else:
          s.config[kv[0]] = kv[1]
    return s.config

  def dump(s,path):
    with open(path, 'w') as outfile:
      json.dump(s.config, outfile)

