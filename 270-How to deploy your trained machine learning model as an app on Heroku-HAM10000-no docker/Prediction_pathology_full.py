#### Create folder for Patchifying Large image #####################
# Python program to explain os.mkdir() method
  
# importing os module
import os
  
# Directory
#directory = "RC"
  
# Parent Directory path
parent_dir_img = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/Patchify/"
#parent_dir_mask = "D:/Pathology_23_5_22/single_msk_patchify_new/"

# Path
#path = os.path.join(parent_dir, directory)
  
# Create the directory
# 'GeeksForGeeks' in
# '/home / User / Documents'
#os.mkdir(path)
#print("Directory '% s' created" % directory)
  

directory_img = "A02B74X11_d_img"  #### replace the name of the image 
#directory_mask = "A02B74X11_d_mask" #### replace the name of the mask
path1 = os.path.join(parent_dir_img, directory_img)
os.mkdir(path1)
path2 = os.path.join(parent_dir_mask, directory_mask)
os.mkdir(path2)
print("Directory '% s' created" % directory_img)
print("Directory '% s' created" % directory_mask)


########################### Patchify the large image and mask #############################

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import glob
import cv2
import os

from matplotlib import pyplot as plt
import keras 
large_image = tiff.imread('D:/Pathology_23_5_22/A02B74X11 d_img.tif')   ### rename ###
large_image.shape
large_image = cv2.resize(large_image, (1792, 768))
plt.imshow(large_image )

    count = 0
    patches_img = patchify(large_image, (256, 256,3), step=256)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            count = count +1
            tiff.imwrite(path1 + '/image(' + str(count)+ ").tif", single_patch_img)
            print(count)
        
large_mask = tiff.imread('D:/Pathology_23_5_22/A02B74X11_d_mask.tif')   ### rename ###
large_mask.shape
large_mask = cv2.resize(large_mask, (1792, 768))
plt.imshow(large_mask)

count = 0
patches_img = patchify(large_mask, (256, 256), step=256)
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i,j,:,:]
        count = count +1
        tiff.imwrite(path2 + '/mask(' + str(count)+ ").tif", single_patch_img)
        print(count)
            
            
            
###################### Read Images and Mask from the Patches folder ######        
                 
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
import glob
import tifffile as tiff
#Path
path_RC1 = path2

path_RC6 = path1
#path_RC2 = "D:/ARAMIS_RENAL_PROJECT/ARAMIS_RENAL/comparison_on_RC145/images_png_for_interpolation/"


train_images = []
train_masks = [] 
SIZE = 256

for i in range(1, 22):
    print(i)
    #plt.figure(figsize=(30, 25))
    
    #images
    img1 = cv2.imread(path_RC1+"/mask("+str(i)+").tif", 0)
    mask = img1
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)  
    train_masks.append(mask)


    img3 = cv2.imread(path_RC6+"/image("+str(i)+").tif", 1)
    img = img3
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    #img = cv2.flip(img, 0)
    train_images.append(img)
    
"""if(not np.all(mask == 0)):
        train_masks.append(mask)
        train_images.append(img)"""



train_images = np.array(train_images)
train_masks = np.array(train_masks)

train_masks = np.expand_dims(train_masks, axis = 3)

train_images.shape

#Sanity check, view few mages
import random

image_number = random.randint(0, len(train_images)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(train_images[image_number,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(train_masks[image_number,:,:,0], cmap='gray')
plt.show()

#### Load the model #################

from keras.models import load_model
model1 = load_model('D:/Pathology_23_5_22/models/0&1class _only_resnet34_pathology_114_tif_img_entire_data_test_size_0.2_epochs_100_finding_random_state_42.hdf5', compile=False)



###Model 1
import segmentation_models as sm
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)

#### Predict the model and save it to a folder

####### To check Prediction #########
import random
test_img_number = random.randint(0, len(train_images)-1)
test_img = train_images[test_img_number]
ground_truth=train_masks[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input1 = preprocess_input1(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1, cmap='jet')
plt.show()


### Reading images to predict and saving it to folder 

path_RC6 = "D:/Pathology_23_5_22/large_image_patches_prediction_new/"
image = "A02B74X11_d"   #### Change the image name ################
path3 = os.path.join(path_RC6, image)
os.mkdir(path3)


for i in range(0, 22):
    test_img_number = i
    test_img = train_images[test_img_number]
    ground_truth=train_masks[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)

    test_img_input1 = preprocess_input1(test_img_input)

    test_pred1 = model1.predict(test_img_input1)
    test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


    plt.figure(figsize=(20, 15))
    ''' plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')'''
    plt.axis("off")
    plt.imshow(test_prediction1, cmap='jet')
    plt.savefig(path3+"/"+"image("+str(i+1)+").png",bbox_inches='tight', pad_inches=0.0)
    plt.show()

############################ Unpatchify  ####################################

import cv2
import glob
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
import numpy as np



#### Check
'''
path_RC6 = "C:/Users/Chippy-PC/Downloads/single_mask_patchify1/"
for img in glob.glob("C:/Users/Chippy-PC/Downloads/single_mask_patchify/*.tif"):
    print(img)
    cv_img = cv2.imread(img)
    plt.axis("off")
    count = count+1
    print(count)
    plt.title(count)
    plt.imshow(cv_img)
    plt.savefig(path_RC6+str(count)+".png",bbox_inches='tight')
    #plt.savefig("C:/Users/Chippy-PC/Downloads/single_mask_patchify1/",single_mask_patchify1)
    plt.show()
'''

#Apply a trained model on large image, patch by patch
path_RC = path3 + "/"
predicted_patches = []

for k in range(1, 22):
    print(k)
    #plt.figure(figsize=(30, 25))

    img3 = cv2.imread(path_RC+"image("+str(k)+").png", 0)
    img = img3
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    #img = cv2.flip(img, 0)
    
    predicted_patches.append(img)

predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (3, 7, 256,256) )


predicted_patches_reshaped.shape


plt.figure(figsize=(30, 25))
square = 6
ix = 1
for i in range(3):
    for j in range(7):
        # specify subplot and turn of axis
        ax = plt.subplot(3, 7, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot 
        plt.imshow(predicted_patches_reshaped[i, j, :, :], cmap="gray")
        ix += 1
# show the figure
plt.show()


path_op = "D:/Pathology_23_5_22/large_image_patches_prediction_results_unpatchified/"
path_op = os.path.join(path_op,image)
os.mkdir(path_op)
reconstructed_image = unpatchify(predicted_patches_reshaped, (768,1792))
plt.figure(figsize=(30, 25))
plt.axis("off")
plt.imshow(reconstructed_image, cmap='gray')
plt.savefig(path_op+"/new"+".png",bbox_inches='tight',pad_inches=0.0)
plt.show()

###### Comparison ##########

path = "D:/Pathology_23_5_22/large_image_patches_prediction_results_unpatchified_comparison/"

 
plt.figure(figsize=(30,25)) 
plt.subplot(131)
plt.title('IMAGE')
plt.axis("off")
plt.imshow(cv2.cvtColor(large_image,cv2.COLOR_BGR2RGB),cmap = "jet")

plt.subplot(132)
plt.title('Ground Truth')
plt.axis("off")
plt.imshow(large_mask,cmap = "gray")

plt.subplot(133)
plt.title('Prediction_Resnet34')
plt.axis("off")
plt.imshow(cv2.cvtColor(reconstructed_image,cv2.COLOR_BGR2RGB))

plt.savefig(path+str(image)+".png",bbox_inches='tight')
plt.show()

########################################
#Predict on large image

#Apply a trained model on large image

from patchify import patchify, unpatchify

large_image = cv2.imread('D:/Pathology_23_5_22/A22S86Y21 a_tif.tif')
large_image = cv2.resize(large_image, (768, 1792))
#This will split the image into small images of shape [3,3]
patches = patchify(large_image, (256, 256, 3), step=256)  #Step=256 for 256 patches means no overlap

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i,j,3] 
        #print(single_patch)
        test_img = single_patch
        test_img = np.expand_dims(test_img, 0)

        test_img_input1 = preprocess_input1(test_img)
        
        test_pred1 = model1.predict(test_img_input1)
        single_patch_predicted_img = np.argmax(test_pred1, axis=3)[0,:,:]
        plt.imshow(single_patch_predicted_img)
        '''single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
        single_patch_input=np.expand_dims(single_patch_norm, 0)
        single_patch_prediction = (model.predict(single_patch))
        single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]'''

        predicted_patches.append(single_patch_predicted_img)
        plt.show()
predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
#predicted_patches_reshaped = np.expand_dims(predicted_patches_reshaped, 4)

reconstructed_image = unpatchify(predicted_patches_reshaped, (768,1792))
plt.imshow(reconstructed_image, cmap='gray')
#plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')

plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

# final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
# plt.imshow(final_prediction)

plt.figure(figsize=(10, 15))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='jet')
plt.show()





########################## glo.glob to reaad files 
import cv2
import glob
from matplotlib import pyplot as plt
for img in glob.glob("D:/Pathology_23_5_22/single_img_patchify/*.tif"):
    print(img)