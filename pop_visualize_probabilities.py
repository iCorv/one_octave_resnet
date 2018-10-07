import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

data = np.load("notes_2018-10-05-17:51:17.npz")
props = data["props"]
print(props.shape)

print(np.max(props))
print(np.min(props))
plt.figure()
plt.pcolormesh(np.flipud(props))
#locs, labels = plt.yticks()
# set last y-tick to 88
#locs[-1] = 88
# find labels in frequency bins
#plt.yticks(locs, np.append(freq_bins.round(decimals=1)[::10], freq_bins.round(decimals=1)[-1]))
#plt.title(title)
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
