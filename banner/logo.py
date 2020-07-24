import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

figsize = (8.4, 1.6)

plt.close(0)
plt.figure(0, figsize=figsize)
ax = plt.subplot(111, frameon=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.text(0,0,'PMOIRED', ha='center', va='center',
        fontsize=100*figsize[1],
        fontfamily='eurofurence', fontweight='bold')


plt.xlim(-1,1)
plt.ylim(-.8,1.2)
plt.savefig('logo_.png')
img = mpimg.imread('logo_.png')
print(img.shape)
x = np.arange(img.shape[0])
y = np.arange(img.shape[1])

r1 = np.sqrt((x[:,None]+5)**2 + (y[None,:])**2)
z1 = np.cos(r1/2.45)
r2 = np.sqrt((x[:,None]-5)**2 + (y[None,:])**2)
z2 = np.cos(r2/2.55)

#z = np.logical_or(z1>0, z2>0)
z = np.maximum((z1>0)*(1+z1), (z2>0)*(1+z2))

RGB = np.zeros(img.shape)
RGB[:,:,0] = z
RGB[:,:,1] = z
RGB[:,:,2] = z

print(RGB.shape)

plt.close(1)
plt.figure(1, figsize=figsize)
ax = plt.subplot(111, frameon=False)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
plt.imshow(1-((1-img)*RGB)[:,:,:3])
plt.savefig('logo.png')
