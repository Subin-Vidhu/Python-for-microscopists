import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io

#STEP1 - Read image and define pixel size
img = cv2.imread("D:/Pathology_23_5_22/Anil project/Path_Data_160_256/images/image_60_16.tif", 0)

pixels_to_um = 0.5 # (1 px = 500 nm)

#cropped_img = img[0:450, :]   #Crop the scalebar region

#Step 2: Denoising, if required and threshold image

#No need for any denoising or smoothing as the image looks good.
#Otherwise, try Median or NLM
#plt.hist(img.flat, bins=100, range=(0,255))

#Change the grey image to binary by thresholding. 
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#print(ret)  #Gives 157 on grains2.jpg. OTSU determined this to be the best threshold. 

#View the thresh image. Some boundaries are ambiguous / faint.
#Some pixles in the middle. 
#Need to perform morphological operations to enhance.

#Step 3: Clean up image, if needed (erode, etc.) and create a mask for grains

kernel = np.ones((3,3),np.uint8) 
eroded = cv2.erode(thresh,kernel,iterations = 1)
dilated = cv2.dilate(eroded,kernel,iterations = 1)

# Now, we need to apply threshold, meaning convert uint8 image to boolean.
mask = dilated == 255  #Sets TRUE for all 255 valued pixels and FALSE for 0
#print(mask)   #Just to confirm the image is not inverted. 

#from skimage.segmentation import clear_border
#mask = clear_border(mask)   #Removes edge touching grains. 

io.imshow(mask)  #cv2.imshow() not working on boolean arrays so using io
#io.imshow(mask[250:280, 250:280])   #Zoom in to see pixelated binary image

#Step 4: Label grains in the masked image

#Now we have well separated grains and background. Each grain is like an object.
#The scipy ndimage package has a function 'label' that will number each object with a unique ID.

#The 'structure' parameter defines the connectivity for the labeling. 
#This specifies when to consider a pixel to be connected to another nearby pixel, 
#i.e. to be part of the same object.

#use 8-connectivity, diagonal pixels will be included as part of a structure
#this is ImageJ default but we have to specify this for Python, or 4-connectivity will be used
# 4 connectivity would be [[0,1,0],[1,1,1],[0,1,0]]
s = [[1,1,1],[1,1,1],[1,1,1]]
#label_im, nb_labels = ndimage.label(mask)
labeled_mask, num_labels = ndimage.label(mask, structure=s)

#The function outputs a new image that contains a different integer label 
#for each object, and also the number of objects found.


#Let's color the labels to see the effect
img2 = color.label2rgb(labeled_mask, bg_label=0)

cv2.imshow('Colored Grains', img2)
cv2.waitKey(0)