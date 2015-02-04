
from js.utils.plot.colors import colorScheme
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl                                                        
#poster
mpl.rc('font',size=35) 
mpl.rc('lines',linewidth=4.)
figSize = (12, 14)

#paper
mpl.rc('font',size=40) 
mpl.rc('lines',linewidth=4.)
figSize = (14, 5.5)
figSize = (14, 10)
legendSize = 35

mpl.rc('figure', autolayout=True)

def plotOverParams(values,name,paramBase,paramName,baseMap,Ns=None,showLeg=None):
  colA = colorScheme('labelMap')['orange']
  colB = colorScheme('labelMap')['turquoise']

  fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
  ax1 = plt.subplot(111)
  base = 'spkm'
  valMean = values[base].mean(axis=1)
  valStd = values[base].std(axis=1)
#  print name,base,valMean
#  print name,base,valStd
  leg1 = ax1.plot(paramBase[base],valMean,label=baseMap[base],c=colA)
  ax1.plot(paramBase[base],valMean-valStd,'--',label=baseMap[base],c=colA,lw=2,alpha=0.7)
  ax1.plot(paramBase[base],valMean+valStd,'--',label=baseMap[base],c=colA,lw=2,alpha=0.7)
  ax1.fill_between(paramBase[base],valMean-valStd , valMean+valStd, color=colA, alpha=0.3)
  if name == '$K$':
    leg1 += ax1.plot( paramBase[base],[30]*len(paramBase[base]), '--', label="$K_{T} = 30$", c=colorScheme('labelMap')['red'])
  ax1.set_xlabel(paramName[base],color=colA)  
  ax1.xaxis.set_label_coords(0.5,-0.08)  
  ax1.set_xlim(paramBase[base].min(),paramBase[base].max())
  ax1.invert_xaxis()
  for  tl in ax1.get_xticklabels():
    tl.set_color(colA)
  tiks = ax1.get_xticks()
  tikLbl = ["{}".format(int(tik)) for tik in tiks[:-1]] + [''] 
  ax1.set_xticklabels(tikLbl)
#  ax1.legend(loc='best')
  ax1.set_ylabel(name)  
  if False and name == 'NMI':
    tiks = ax1.get_yticks()
    tikLbl = []
    for tik in tiks[::2]:
      tikLbl += [str(tik),'']
    del tikLbl[-1]
    ax1.set_yticklabels(tikLbl)

  ax2 = ax1.twiny()
  base = 'DPvMFmeans'
  valMean = values[base].mean(axis=1)
  valStd = values[base].std(axis=1)
#  print name,base,valMean
#  print name,base,valStd
  leg2 = ax2.plot(paramBase[base],values[base].mean(axis=1),label=baseMap[base],c=colB)
  ax2.plot(paramBase[base],valMean-valStd,'--',label=baseMap[base],c=colB,lw=2,alpha=0.7)
  ax2.plot(paramBase[base],valMean+valStd,'--',label=baseMap[base],c=colB,lw=2,alpha=0.7)
  ax2.fill_between(paramBase[base],valMean-valStd , valMean+valStd, color=colB, alpha=0.3)
  ax2.set_xlabel(paramName[base],color=colB)  
  ax2.xaxis.set_label_coords(0.5,1.12)  
  for  tl in ax2.get_xticklabels():
    tl.set_color(colB)
  tiks = ax2.get_xticks()
  tikLbl = ax2.get_xticklabels()
  tikLbl = [''] + [str(int(tik)) for tik in tiks[1:]] 
  ax2.set_xticklabels(tikLbl)
  ax2.set_xlim(paramBase[base].min(),paramBase[base].max())
  if False and name == 'NMI':
    tiks = ax2.get_yticks()
    tikLbl = []
    for tik in tiks[::2]:
      tikLbl += [str(tik),'']
    del tikLbl[-1]
    ax2.set_yticklabels(tikLbl)
  if not showLeg is None and showLeg:
    legs = leg2+leg1
    labs = [leg.get_label() for leg in legs]
    ax2.legend(legs,labs,loc='best',prop={'size':legendSize})
  plt.tight_layout()
  plt.subplots_adjust(right=0.85,bottom=0.3)
  return fig

def plotOverParamsYaxes(values,name,paramBase,paramName,baseMap,Ns=None,showLeg=None):
  colA = colorScheme('labelMap')['orange']
  colB = colorScheme('labelMap')['turquoise']

  fig = plt.figure(figsize=figSize, dpi=80, facecolor='w', edgecolor='k')
  ax1 = plt.subplot(111)
  base = 'spkm'
  valMean = values[base].mean(axis=1)
  valStd = values[base].std(axis=1)
#  print name,base,valMean
#  print name,base,valStd
  leg1 = ax1.plot(valMean,paramBase[base],label=baseMap[base],c=colA)
  ax1.plot(valMean-valStd,paramBase[base],'--',label=baseMap[base],c=colA,lw=2,alpha=0.7)
  ax1.plot(valMean+valStd,paramBase[base],'--',label=baseMap[base],c=colA,lw=2,alpha=0.7)
  ax1.fill_betweenx(paramBase[base],valMean-valStd , valMean+valStd, color=colA, alpha=0.3)
  if name == '$K$':
    leg1 += ax1.plot([30]*len(paramBase[base]), paramBase[base], '--', label="$K_{T} = 30$", c=colorScheme('labelMap')['red'])
  ax1.set_ylabel(paramName[base],color=colA)  
  ax1.set_ylim(paramBase[base].min(),paramBase[base].max())
  ax1.invert_yaxis()
  for  tl in ax1.get_yticklabels():
    tl.set_color(colA)
  tiks = ax1.get_yticks()
  tikLbl = [str(tik) for tik in tiks[:-1]] 
  tikLbl += [''] 
  ax1.set_yticklabels(tikLbl)
#  ax1.legend(loc='best')
  ax1.set_xlabel(name)  
  if False and name == 'NMI':
    tiks = ax1.get_xticks()
    tikLbl = []
    for tik in tiks[::2]:
      tikLbl += [str(tik),'']
    del tikLbl[-1]
    ax1.set_xticklabels(tikLbl)

  ax2 = ax1.twinx()
  base = 'DPvMFmeans'
  valMean = values[base].mean(axis=1)
  valStd = values[base].std(axis=1)
#  print name,base,valMean
#  print name,base,valStd
  leg2 = ax2.plot(values[base].mean(axis=1),paramBase[base],label=baseMap[base],c=colB)
  ax2.plot(valMean-valStd,paramBase[base],'--',label=baseMap[base],c=colB,lw=2,alpha=0.7)
  ax2.plot(valMean+valStd,paramBase[base],'--',label=baseMap[base],c=colB,lw=2,alpha=0.7)
  ax2.fill_betweenx(paramBase[base],valMean-valStd , valMean+valStd, color=colB, alpha=0.3)
  if not Ns is None: # and not name == '$K$':
    xlim = ax2.get_xlim()
    # first to spkm
    base = 'spkm'
    Nmean = Ns[base].mean(axis=1)
#    iKtrue = np.where(np.abs(Nmean-30)<2)
#    ax1.plot(values[base].mean(axis=1)[iKtrue],paramBase[base][iKtrue],'x',mew=4,ms=15,label=baseMap[base]+' $K={}$'.format(Nmean[iKtrue]),c=colorScheme('labelMap')['red'])
    iKtrue = np.argmin(np.abs(Nmean-30))
    ax1.plot([xlim[0],values[base].mean(axis=1)[iKtrue]], [paramBase[base][iKtrue], paramBase[base][iKtrue]],'-',mew=4,ms=15,label=baseMap[base]+' \
        $K={}$'.format(Nmean[iKtrue]),c=colA, alpha=0.7)
    ax1.plot([xlim[0],values[base].mean(axis=1)[iKtrue]], [paramBase[base][iKtrue], paramBase[base][iKtrue]],'--',mew=4,ms=15,label=baseMap[base]+' \
        $K={}$'.format(Nmean[iKtrue]),c=colorScheme('labelMap')['red'], alpha=1.)
    # then do DPvMFmeans
    base = 'DPvMFmeans'
    Nmean = Ns[base].mean(axis=1)
#    iKtrue = np.where(np.abs(Nmean-30)<2)
#    ax2.plot(values[base].mean(axis=1)[iKtrue],paramBase[base][iKtrue],'x',mew=4,ms=15,label=baseMap[base]+' $K={}$'.format(Nmean[iKtrue]),c=colorScheme('labelMap')['red'])
    iKtrue = np.argmin(np.abs(Nmean-30))
    ax2.plot([values[base].mean(axis=1)[iKtrue],xlim[1]], [paramBase[base][iKtrue], paramBase[base][iKtrue]],'-',mew=4,ms=15,label=baseMap[base]+' \
        $K={}$'.format(Nmean[iKtrue]),c=colB, alpha = 0.7)
    ax2.plot([values[base].mean(axis=1)[iKtrue],xlim[1]], [paramBase[base][iKtrue], paramBase[base][iKtrue]],'--',mew=4,ms=15,label=baseMap[base]+' \
        $K={}$'.format(Nmean[iKtrue]),c=colorScheme('labelMap')['red'], alpha = 1.)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
  ax2.set_ylabel(paramName[base],color=colB)  
  for  tl in ax2.get_yticklabels():
    tl.set_color(colB)
  tiks = ax2.get_yticks()
  tikLbl = ax2.get_yticklabels()
  tikLbl = [''] 
  tikLbl += [str(tik) for tik in tiks[1:]] 
  ax2.set_yticklabels(tikLbl)
  ax2.set_ylim(paramBase[base].min(),paramBase[base].max())
  if False and name == 'NMI':
    tiks = ax2.get_xticks()
    tikLbl = []
    for tik in tiks[::2]:
      tikLbl += [str(tik),'']
    del tikLbl[-1]
    ax2.set_xticklabels(tikLbl)
  if not showLeg is None and showLeg:
    legs = leg2+leg1
    labs = [leg.get_label() for leg in legs]
    ax2.legend(legs,labs,loc='best',prop={'size':legendSize})
  plt.tight_layout()
  plt.subplots_adjust(right=0.85,bottom=0.3)
  return fig
