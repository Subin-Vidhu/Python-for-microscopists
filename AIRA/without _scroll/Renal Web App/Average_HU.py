# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:25:35 2022

@author: Subin-PC
"""

#Libraries
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2

#Reading Image
ct=nib.load("C:/Users/Subin-PC/Desktop/28-10/Renal Web App/static/images/2/img.nii")
#ct_mask = nib.load("D:/ARAMIS/ARAMIS_RENAL/ARAMIS_RENAL_FULL_DATASET/RC175/mask.nii")
Pred = nib.load("C:/Users/Subin-PC/Desktop/28-10/Renal Web App/static/Results/2/mask.nii")
#dir_for_img = "C:/Users/Subin-PC/Desktop/"
print(ct.dataobj.slope, ct.dataobj.inter)
    

ct_data = ct.get_fdata()
ct_data.max()
ct_data.min()

Pred_data = Pred.get_fdata()
Pred_data.max()
Pred_data.min()

np.unique(Pred_data)

#Resize and flip image so that cordinates need not be changed
original = []
img_size = 512
Pred_data = Pred_data.transpose(2,1,0)
for i in range(0, len(Pred_data)):  
    test_image_gt_one1 = Pred_data[i,:,:]
    #original.append(test_image_gt_one1)
    test_image_gt_one_reshape1 = cv2.resize(test_image_gt_one1,(img_size,img_size), cv2.COLOR_BGR2RGB)
    original.append(test_image_gt_one_reshape1)
original = np.array(original)
original = original.transpose(2,1,0)

original.max()
original.min()

##Cordinates from Prediction
#calculi_cord_img = np.where(ct_data[:,:,:] == 1350)

calculi_cord = np.where(original[:,:,(original.shape[2]-123)] == 1)
#Left_kidney_cord = np.where(ct_data_mask[:,:,408] == 2)
#Right_kidney_cord_img = np.where(ct_data_mask[:,:,408] == 3)

#For loop for going through the entire slices
Sum_Calculi = 0
Count_pixels = 0
Calculi_Values = []
x=[]
y=[]
z=[]
length = original.shape[2]
for slice_num in range(1,original.shape[2]+1):
    print(slice_num)
    calculi_cord = np.where(original[:,:,(length-slice_num)] == 1)
    x1=list(calculi_cord[0])
    y1=list(calculi_cord[1])
    z1= [(length-slice_num)] * len(x1)
    for i in range(0, len(x1)):
        Value = ct_data[x1[i],y1[i],(length-slice_num)]
        if(Value<0):
            Value=0 #ignore if the value is negative
        else:
            Value = Value
        print(x1[i])
        print(y1[i])
        print((original.shape[2]-slice_num))
        print(Value)
        Calculi_Values.extend([Value])
        Sum_Calculi += Value
        Count_pixels += 1
    x.extend(x1)
    y.extend(y1)
    z.extend(z1)
        

if(Count_pixels == 0):
    Avg_HU = 0 
else:
    Avg_HU = Sum_Calculi/Count_pixels
print(Avg_HU)


#For saving the file as csv file for checking with ImageJ
coord = []
for a, b, c in zip( x, y, z ):
    coord.append( [ a, ((original.shape[1]-b)-1), ((original.shape[2]-(c))-1) ] )
    

dictionary = list(zip(coord, Calculi_Values))
print(dictionary)
np.savetxt("C:/Users/Subin-PC/Desktop/HU.csv", dictionary, delimiter =",", fmt ='% s')


import pandas as pd 

data = pd.read_csv("C:/Users/Subin-PC/Desktop/HU.csv", names=["x", "y", "z", "Value"])
print("The Avg HU value from the csv file is: ", data["Value"].mean())
print("Some findings from our csv file is: ", data.describe())

"""    
l1 = [ 2, 4, 6, 8 ]
l2 = [ 1, 3, 5, 7 ]
l3 = [ 1, 2, 3, 4]

coord = []

for a, b, c in zip( x, y, z ):
    coord.append( [ a, b, c ] )

print( coord )

value_check = [100, 200, 300, 400]

dictionary = list(zip(coord, value_check))
print(dictionary)

"""



