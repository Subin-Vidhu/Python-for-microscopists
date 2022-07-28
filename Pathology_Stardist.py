from __future__ import print_function, unicode_literals, absolute_import,  division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

#Random color map labels
np.random.seed(42)
lbl_cmap=random_label_cmap()

#Read input image and corresponding mask names
X = sorted(glob('D:/Pathology_23_5_22/Anil project/Path_Data_160_256/images/*.tif'))
Y = sorted(glob('D:/Pathology_23_5_22/Anil project/Path_Data_160_256/masks/*.tif'))

#Read images and masks using their names.
#We are using tifffile library to read images as we have tif images.
X = list(map(imread,X))
Y = list(map(imread,Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1] #If no third dim.then number of channels=1. Otherwise get the num channels

#Normalize input images and fill holes in masks
axis_norm = (0,1) #normalize channels independently
#axis_norm=(0,1,2) #normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels%s."%('jointly'if axis_norm is None or 2 in axis_norm else'independently'))
    sys.stdout.flush()
X = [normalize(x,1,99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

#Split to train and val
#You can use any method to split.Iam following the method used in StarDist documentation example
assert len(X) > 1,"not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1,int(round(0.15*len(ind))))
ind_train,ind_val=ind[:-n_val],ind[-n_val:]
X_val,Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
X_trn,Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
print('number of images: %3d' %len(X))
print(' - training:      %3d' %len(X_trn))
print(' - validation:    %3d' %len(X_val))

#PLot image and label for some images-sanity check
def plot_img_label(img, lbl, img_title="image", lbl_title="label", ** kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img,cmap='gray',clim=(0,1))
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap = lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()


i = min(9, len(X)-1)

img, lbl = X[i], Y[i]
assert img.ndim in (2,3)
img = img if (img.ndim ==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl)

#Check config to get an idea about all parameters
print(Config2D.__doc__)

#Define the config by setting some parameter values
#32 isagood default choice(see 1_data.ipynb)
n_rays=32 #Number of radial directions for the star-convex polygon.

#Use OpenCL-based computations for data generator during training(requires'gputools')
use_gpu = False and gputools_available()
#Predict on subsampled grid for increased efficiency and larger field of view
grid=(2,2)

conf = Config2D(
    n_rays = n_rays,
    grid =  grid,
    use_gpu = use_gpu,
    n_channel_in =n_channel,
)
print(conf)
vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    #adjust as necessary:limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)
    #alternatively,try this:
    #limit_gpu_memory(None,allow_growth=True)
    

#Save model to the specified directory
model = StarDist2D(conf, name='stardist_tutorial', basedir='D:/Pathology_23_5_22/Anil project/Path_Data_160_256/models')

#Define the network field of view to size larger than the median object size
median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:  {median_size}")
print(f"network field of view: {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")
    
    
#Defineafew augmentation methods
def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim,img.ndim)))
    mask = mask.transpose(perm)
                                     
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img,axis=ax)
            mask = np.flip(mask,axis-ax)
    return img,mask

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x,y):
    """Augmentation ofasingle input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """

    x,y = random_fliprot(x, y)
    x = random_intensity_change(x)
    #add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

#plot some augmented examples
img, lbl = X[9],Y[9]
plot_img_label(img, lbl)
for _ in range(3):
    img_aug,lbl_aug = augmenter(img, lbl)
    plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")
    
    
model.train(X_trn, Y_trn, validation_data = (X_val,Y_val), augmenter = augmenter, epochs = 150, steps_per_epoch = 100)

#Optimize the thresholds using the trained model
model.optimize_thresholds(X_val, Y_val)

#Load saved model
my_model = StarDist2D(None, name = 'stardist', basedir = '')

model = my_model

#Prediction on validation images
Y_val_pred = [model.predict_instances(x, n_tiles = model._guess_n_tiles(x), show_title_progress = False)[0]
             for x in tqdm(X_val)]

#Plot original labels and predictions
plot_img_label(X_val[0], Y_val[0], lbl_title="label GT")
plot_img_label(X_val[0], Y_val_pred[0], lbl_title="label Pred")

taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
stats = [matching_dataset(Y_val, Y_val_pred, thresh = t, show_progress = False) for t in tqdm(taus)]

stats[taus.index(0.5)]

#Plot key metrics.
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
for m in('precision','recall','accuracy','f1','mean_true_score','mean_matched_score','panoptic_quality'):
    ax1.plot(taus,[s._asdict()[m] for s in stats],'.-',lw=2,label=m)
ax1.set_xlabel(r'IoU threshold$\tau$')
ax1.set_ylabel('Metric value')
ax1.grid()
ax1.legend()

for m   in('fp','tp','fn'):
    ax2.plot(taus,[s._asdict()[m] for s in stats],'.-', lw=2, label=m)
ax2.set_xlabel(r'IoU threshold$\tau$')
ax2.set_ylabel('Number#')
ax2.grid()
ax2.legend();

