#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import h5py

# load the image
data = h5py.File("img.h5", "r")

mask = data["mask"][:]
tau = data["tau"][:]
img = data["img"][:]

nvel = img.shape[0]
ncolpix = img.shape[1]
nrowpix = img.shape[2]

print("Max value in tau", np.max(tau))
print()

print("Max value in image", np.max(img))
print("Min value in image", np.min(img))


fig, ax = plt.subplots(ncols=nvel, nrows=3, figsize=(1.0 * nvel, 3.0))
for j in range(nvel):
    ax[0,j].imshow(mask[j,:,:].T, interpolation="none")
    ax[1,j].imshow(tau[j,:,:].T, interpolation="none")
    ax[2,j].imshow(img[j,:,:].T, interpolation="none")

fig.savefig("image.png", dpi=120)
