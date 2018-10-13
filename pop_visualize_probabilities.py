import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pypianoroll as ppr
from pypianoroll import Multitrack, Track

data = np.load("props_2018-13-10.npz")
props = data["props"]
print(props.shape)
#props = np.round(props)
print(np.max(props))
print(np.min(props))
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
ax1.pcolormesh((props))
#locs, labels = plt.yticks()
# set last y-tick to 88
#locs[-1] = 88
# find labels in frequency bins
#plt.yticks(locs, np.append(freq_bins.round(decimals=1)[::10], freq_bins.round(decimals=1)[-1]))
#plt.title(title)
#ax1.colorbar(format='%+2.0f dB')
#plt.tight_layout()
#plt.show()


data = np.load("MAPS_MUS-alb_se3_AkPnBcht_25050.npz")
props = np.transpose(data["labels"])
print(props.shape)

#pianoroll = ppr.parse("/Users/Jaedicke/Desktop/MAPS_MUS-alb_se3_AkPnBcht.mid", beat_resolution=24,
#          name='MAPS_MUS-alb_se3_AkPnBcht')
pianoroll = Multitrack('/Users/Jaedicke/Desktop/MAPS_MUS-alb_se3_AkPnBcht.mid')
#f,ax2 = pianoroll.plot()
ppr.plot(pianoroll, filename=None, mode='separate', track_label='name', preset='frame', cmaps=None, xtick='auto', ytick='octave', xticklabel=True, yticklabel='auto', tick_loc=None, tick_direction='in', label='both', grid='off', grid_linestyle=':', grid_linewidth=0.5)

#ax2.pcolormesh(np.flipud(props))
#locs, labels = plt.yticks()
# set last y-tick to 88
#locs[-1] = 88
# find labels in frequency bins
#plt.yticks(locs, np.append(freq_bins.round(decimals=1)[::10], freq_bins.round(decimals=1)[-1]))
#plt.title(title)
#ax2.colorbar(format='%+2.0f dB')
#plt.tight_layout()
plt.show()
