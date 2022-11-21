from flask import Flask, render_template, request, redirect, flash,session
import glob
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from PIL import Image
from tensorflow.keras.models import load_model
import segmentation_models as sm
sm.set_framework('tf.keras')
import copy
import time
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import sys

def getPrediction1(filename,slice_START, slice_END):
    print("Hi, I am inside Prediction function!")
    print("LIMIT RECURSIVE1",sys.getrecursionlimit())
    sys.setrecursionlimit(3000)
    print("LIMIT RECURSIVE2",sys.getrecursionlimit())
    userid = session['userid']
    userid=str(userid)
    
    BACKBONE1 = 'resnet101'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)
    print("Backbone Loaded")
    #filename = "A31P15X11 a.tif"
    #from keras.models import load_model
    model1 = load_model('model/weights-improvement-150-0.68.hdf5', compile=False)

    print("model loaded")
    #Reading Images RC1
    print("filename", filename)
    path = "static/Image_Copy/"+userid+"/"
    test_image_gt_RC1__=nib.load(path+filename)
    test_image_gt_RC1_ = test_image_gt_RC1__.get_fdata()
    print("Read .nii File")
    test_image_gt_RC1 = test_image_gt_RC1_.transpose(2,1,0)
    test_image_gt_ = copy.deepcopy(test_image_gt_RC1)
    test_image_gt_RC1 = np.clip(test_image_gt_RC1,-135,215)
    test_image_gt_RC1=scaler.fit_transform(test_image_gt_RC1.reshape(-1, test_image_gt_RC1.shape[-1])).reshape(test_image_gt_RC1.shape)
    test_image_gt_RC1.shape
    for i in range(0, len(test_image_gt_RC1)):
        test_image_gt_RC1[i,:,:] = cv2.flip(test_image_gt_RC1[i,:,:], 0)
        test_image_gt_[i,:,:] = cv2.flip(test_image_gt_[i,:,:], 0)
        
    ground_truth_array_test = []
    ground_truth_array_test_without_threshold = []
    img_size = 256
    for pointer in range(test_image_gt_RC1.shape[0]):
        image_groundtruth_test = test_image_gt_RC1[pointer,:,:]
        image_groundtruth_test_ = test_image_gt_[pointer,:,:]
        #image_groundtruth_test = cv2.resize(test_image_gt_RC1[pointer,:,:], (img_size,img_size), cv2.IMREAD_COLOR)
        ground_truth_array_test.append(image_groundtruth_test)
        ground_truth_array_test_without_threshold.append(image_groundtruth_test_)


    ground_truth_array_test = np.array(ground_truth_array_test)
    ground_truth_array_test_without_threshold = np.array(ground_truth_array_test_without_threshold)
    #ground_truth_array_test = np.expand_dims(ground_truth_array_test, 3)
    print("ground_truth_array_test_dimensions", ground_truth_array_test.shape)    


    original = []
    preprocessed = []
    #append slices

    ###### for testing the performance of the model
    if slice_END > len(ground_truth_array_test):
        slice_END = len(ground_truth_array_test)

    slice_START = len(ground_truth_array_test) - slice_START
    slice_END = len(ground_truth_array_test) - slice_END


    for i in range(slice_END, slice_START):
        
        test_image_gt_one1 = ground_truth_array_test[i,:,:]
        test_image_gt_one1_ = ground_truth_array_test_without_threshold[i,:,:]
        original.append(test_image_gt_one1_)
        test_image_gt_one_reshape1 = cv2.resize(test_image_gt_one1,(img_size,img_size), cv2.COLOR_BGR2RGB)

        test_img_input2 = preprocess_input1(test_image_gt_one_reshape1)
        test_img_input2 = np.expand_dims(test_img_input2, 0)
        test_pred2 = model1.predict(test_img_input2)
        test_prediction2 = np.argmax(test_pred2, axis=3)[0,:,:]
        preprocessed.append(test_prediction2)

        arr = np.unique(test_prediction2)
        print("Predicting...")
        print(i)
        
        #plt.figure(figsize=(10,10))
        #plt.imshow(test_prediction2, cmap='jet')
        #plt.axis("off")
        #plt.title("%d" %i + str(arr))
        #plt.savefig(path_RC6+str(i)+".png",bbox_inches='tight')
        #plt.show()
    
    #ORIGINAL
    x, y, z = test_image_gt_RC1__.header.get_zooms()
    #x, y = x * 2, y * 2

    
    original= np.array(original)
    print("Original_dimensions", original.shape)
    for i in range(0, len(original)):
        original[i,:,:] = cv2.flip(original[i,:,:], -1)
    original = np.transpose(original, [2, 1, 0])
    nifty = nib.Nifti1Image(original, np.diagflat([x, y, z, 1]), dtype=np.int16)
    files = glob.glob('static/images/'+userid+'/*')
    for f in files:
        f = f.replace("\\","/")
        os.remove(f)
    dir_for_img = "static/images/"+userid
    print("Original image removed so that new image can be added")
    nib.save(nifty, os.path.join(dir_for_img, 'img.nii'))
    
    #PREDICTION
    x, y, z = test_image_gt_RC1__.header.get_zooms()
    x, y = x * 2, y * 2

    preprocessed= np.array(preprocessed)
    for i in range(0, len(preprocessed)):
        preprocessed[i,:,:] = cv2.flip(preprocessed[i,:,:], -1)
    preprocessed = np.transpose(preprocessed, [2, 1, 0])
    nifty = nib.Nifti1Image(preprocessed, np.diagflat([x, y, z, 1]), dtype=np.int16)
    print("Try deleting mask if any")
    files = glob.glob('static/Result_Copy/'+userid+'/*')
    for f in files:
        f = f.replace("\\","/")
        os.remove(f)
    print("Deleted mask if any")
    dir_for_img = "static/Result_Copy/"+userid+"/"
    dir_for_img_ = "static/Results/"+userid+"/"
    if not os.path.exists(dir_for_img):
        os.mkdir(dir_for_img)
    if not os.path.exists(dir_for_img_):
        os.mkdir(dir_for_img_)
    print("Original mask removed so that new image can be added")
    nib.save(nifty, os.path.join(dir_for_img, 'mask.nii'))
    nib.save(nifty, os.path.join(dir_for_img_, 'mask.nii'))
    
    pred_path = "static/Result_Copy/"+userid+"/"
    mask_pred  = nib.load(pred_path+"mask.nii")

    ct=nib.load("static/images/"+userid+"/img.nii")
    ct_data = ct.get_fdata()
    x1,y1,z1 = mask_pred.header.get_zooms()
    #x1,y1,z1 = x1/2,y1/2,z1
    voxel_volume1 = np.prod(x1*y1*z1)
    data1 = np.asarray(mask_pred.dataobj)
    count1 =  np.unique(data1,  return_counts=True) 
    print("Count", count1)
    keys = count1[0]
    values = count1[1]
    values = [x * voxel_volume1 for x in values]
    volume1 = {}
    for i in range(len(keys)):
        volume1[keys[i]] = values[i]
    
    # voxel_volume = np.prod(img.header.get_zooms())
    # data = np.asarray(img.dataobj)
    # count =  np.unique(data,  return_counts=True) 
    # print("COunt", count)
    # volume = {k: count[1][k] * voxel_volume for k in range(0,len(count[0]))}

    print("Volume", volume1)

    ################     Cranio - Caudal Length of Kidneys     ########

    initial=time.perf_counter()

    Right_Calculi = 0
    Left_Calculi = 0
    Right_Calculi_cord=[]
    Left_Calculi_cord=[]
    var1=-1
    var2=0
    c=[[]]
    var3=-1
    var4=0
    d=[[]]
    cranio_caudal_Left_min=[]
    cranio_caudal_Right_min=[]
    cranio_caudal_Left_max=[]
    cranio_caudal_Right_max=[]
    #Divide into two and add the total number of calculi slices
    #Axial
    for z in range(0,data1.shape[2]):
        if(2 in np.unique(data1[:,:,z]) or 3 in np.unique(data1[:,:,z])):
            for x in range(0,data1.shape[0]):
                if(2 in np.unique(data1[x,:,:]) or 3 in np.unique(data1[x,:,:])):
                    for y in range(0,data1.shape[1]):
                        #print((x,y,z))
                        if(data1[x,y,z] == 2 and (cranio_caudal_Left_min==[])):
                            cranio_caudal_Left_min = [x,y,z]
                        elif(data1[x,y,z] == 3 and cranio_caudal_Right_min==[]):
                            cranio_caudal_Right_min = [x,y,z]
                        if(data1[data1.shape[0]-x-1,data1.shape[1]-y-1,data1.shape[2]-z-1] == 2 and cranio_caudal_Left_max==[]):
                            cranio_caudal_Left_max = [data1.shape[0]-x-1,data1.shape[1]-y-1,data1.shape[2]-z-1]
                        elif(data1[data1.shape[0]-x-1,data1.shape[1]-y-1,data1.shape[2]-z-1] == 3 and cranio_caudal_Right_max==[]):
                            cranio_caudal_Right_max = [data1.shape[0]-x-1,data1.shape[1]-y-1,data1.shape[2]-z-1]
                        if(cranio_caudal_Left_min!=[] and cranio_caudal_Right_min!=[] and cranio_caudal_Left_max!=[] and cranio_caudal_Right_max!=[]):
                            break
                    if(cranio_caudal_Left_min!=[] and cranio_caudal_Right_min!=[] and cranio_caudal_Left_max!=[] and cranio_caudal_Right_max!=[]):
                        break
            if(cranio_caudal_Left_min!=[] and cranio_caudal_Right_min!=[] and cranio_caudal_Left_max!=[] and cranio_caudal_Right_max!=[]):
                break


    #Left Kidney
    print("cranio_caudal_Left_min",cranio_caudal_Left_min)
    print("cranio_caudal_Left_max",cranio_caudal_Left_max)
    if(cranio_caudal_Left_min == []):
        cranio_caudal_Left_min = 0
    else:
        cranio_caudal_Left_min = [cranio_caudal_Left_min[0]*x1,cranio_caudal_Left_min[1]*y1,cranio_caudal_Left_min[2]*z1]
    if(cranio_caudal_Left_max == []):
        cranio_caudal_Left_max = 0
    else:
        cranio_caudal_Left_max = [cranio_caudal_Left_max[0]*x1,cranio_caudal_Left_max[1]*y1,cranio_caudal_Left_max[2]*z1]

    # Calculate Euclidean distance
    import math
    if((cranio_caudal_Left_min == 0) and (cranio_caudal_Left_max == 0)):
        distance_left = 0
    else:
        distance_left = math.dist(cranio_caudal_Left_min, cranio_caudal_Left_max)
    print("Left_cranio_caudal_length",(round(distance_left, 2)),"mm")

    #Right Kidney
    #cranio_caudal_Right_min = [cranio_caudal_Right_min[0]*x1,cranio_caudal_Right_min[1]*y1,cranio_caudal_Right_min[2]*z1]
    #cranio_caudal_Right_max = [cranio_caudal_Right_max[0]*x1,cranio_caudal_Right_max[1]*y1,cranio_caudal_Right_max[2]*z1]

    if(cranio_caudal_Right_min == []):
        cranio_caudal_Right_min = 0
    else:
        cranio_caudal_Right_min = [cranio_caudal_Right_min[0]*x1,cranio_caudal_Right_min[1]*y1,cranio_caudal_Right_min[2]*z1]
    if(cranio_caudal_Right_max == []):
        cranio_caudal_Right_max = 0
    else:
        cranio_caudal_Right_max = [cranio_caudal_Right_max[0]*x1,cranio_caudal_Right_max[1]*y1,cranio_caudal_Right_max[2]*z1]

    #test1=[header_x, header_y,header_z]
    #test2=[0,0,0]
    # Calculate Euclidean distance
    #import math
    if((cranio_caudal_Right_min == 0) and (cranio_caudal_Right_max == 0)):
        distance_right = 0
    else:
        distance_right = math.dist(cranio_caudal_Right_min, cranio_caudal_Right_max)
    print("Right_cranio_caudal__length",(round(distance_right, 2)),"mm")   
    

    final=time.perf_counter()
    time_duration=final-initial
    print("Time taken for cranio_caudal__length:",time_duration)     


    #################################   END  Cranio - Caudal Length of Kidneys     ###################
        ###################### Separate Calculis ###################

    def solve_plane(tlt, brt, p) :
        if (p[0] > tlt[0] and p[0] < brt[0] and p[1] > tlt[1] and p[1] < brt[1]) :
            return True
        else :
            return False    


    #voxel_volume1 = np.prod(header_x*header_y*header_z) 
    Right_Calculi = 0
    Left_Calculi = 0
    Right_Calculi_cord=[]
    Left_Calculi_cord=[]
    var1=-1
    var2=0
    c=[[]]
    var3=-1
    var4=0
    d=[[]]
    #Divide into two and add the total number of calculi slices
    #Axial

    initial=time.perf_counter()
    for z in range(0,data1.shape[2]):
        if(1 in np.unique(data1[:,:,z])):
            for x in range(0,data1.shape[0]):
                for y in range(0,data1.shape[1]):
                    #print((x,y,z))
                    if((data1[x,y,z] == 1) and (solve_plane((128,0), (256,256), (x,y)))):
                        Right_Calculi = Right_Calculi+1
                        Right_Calculi_cord.append((x,y,z))
                        if(z-var1>=2 and var1!=-1):
                            var2=var2+1
                            c=c+[[(x,y,z)]]
                        elif(z-var1<=1 or var1==-1):
                            c[var2].append((x,y,z))
                        var1=z
                    elif((data1[x,y,z] == 1) and (solve_plane((0,0), (128,256), (x,y)))):
                        Left_Calculi = Left_Calculi+1
                        Left_Calculi_cord.append((x,y,z))
                        if(z-var3>=2 and var3!=-1):
                            var4=var4+1
                            d=d+[[(x,y,z)]]
                        elif(z-var3<=1 or var3==-1):
                            d[var4].append((x,y,z))
                        var3=z

    #final=time.perf_counter()
    #time_duration=final-initial
    #print(time_duration)



    #initial=time.perf_counter()
    Right_Calculi = 0
    Left_Calculi = 0
    Right_Calculi_cord=[]
    Left_Calculi_cord=[]
    var1=-1
    var2=0
    c_sagittal=[[]]
    var3=-1
    var4=0
    d_sagittal=[[]]
    #Sagittal
    for x in range(0,data1.shape[0]):
        if(1 in np.unique(data1[x,:,:])):
            for y in range(0,data1.shape[1]):
                for z in range(0,data1.shape[2]):
                    if((data1[x,y,z] == 1) and (solve_plane((128,0), (256,256), (x,y)))):
                        Right_Calculi = Right_Calculi+1
                        Right_Calculi_cord.append((x,y,z))
                        if(x-var1>=2 and var1!=-1):
                            var2=var2+1
                            c_sagittal=c_sagittal+[[(x,y,z)]]
                        elif(x-var1<=1 or var1==-1):
                            c_sagittal[var2].append((x,y,z))
                        var1=x
                    elif((data1[x,y,z] == 1) and (solve_plane((0,0), (128,256), (x,y)))):
                        Left_Calculi = Left_Calculi+1
                        Left_Calculi_cord.append((x,y,z))
                        if(x-var3>=2 and var3!=-1):
                            var4=var4+1
                            d_sagittal=d_sagittal+[[(x,y,z)]]
                        elif(x-var3<=1 or var3==-1):
                            d_sagittal[var4].append((x,y,z))
                        var3=x


    #final=time.perf_counter()
    #time_duration=final-initial
    #print(time_duration)

    #initial=time.perf_counter()
    #Coronal
    Right_Calculi = 0
    Left_Calculi = 0
    Right_Calculi_cord=[]
    Left_Calculi_cord=[]
    var1=-1
    var2=0
    c_coronal=[[]]
    var3=-1
    var4=0
    d_coronal=[[]]
    for y in range(0,data1.shape[1]):
        if(1 in np.unique(data1[:,y,:])):
            for x in range(0,data1.shape[0]):
                for z in range(0,data1.shape[2]):
                    if((data1[x,y,z] == 1) and (solve_plane((128,0), (256,256), (x,y)))):
                        Right_Calculi = Right_Calculi+1
                        Right_Calculi_cord.append((x,y,z))
                        if(y-var1>=2 and var1!=-1):
                            var2=var2+1
                            c_coronal=c_coronal+[[(x,y,z)]]
                        elif(y-var1<=1 or var1==-1):
                            c_coronal[var2].append((x,y,z))
                        var1=y
                    elif((data1[x,y,z] == 1) and (solve_plane((0,0), (128,256), (x,y)))):
                        Left_Calculi = Left_Calculi+1
                        Left_Calculi_cord.append((x,y,z))
                        if(y-var3>=2 and var3==-1):
                            var4=var4+1
                            d_coronal=d_coronal+[[(x,y,z)]]
                        elif(y-var3<=1 or var3==-1):
                            d_coronal[var4].append((x,y,z))
                        var3=y
                    
    final=time.perf_counter()
    time_duration=final-initial
    print("Time taken for Separating Calculi: ",time_duration)
                    

    ###################### END Separate Calculis ###################

    '''
    Right_Calculi_Volume = Right_Calculi * voxel_volume
    Left_Calculi_Volume = Left_Calculi * voxel_volume

    print("Right Calculi Volume is: ", Right_Calculi_Volume)
    print("Left Calculi Volume is: ", Left_Calculi_Volume)
    print("TOtal Volume is :",round(((Right_Calculi_Volume+Left_Calculi_Volume)/1000), 2))
    '''

    ############# Store Left and Right Calculis ########################

    initial=time.perf_counter()
    c_coronal = list(filter(None, c_coronal))
    c_sagittal = list(filter(None, c_sagittal))
    c = list(filter(None, c))

    index_calculis_R=0
    Calculis_R=[[]]
    for i in range(len(c)):
        for j in c[i]:
            Calculis_R[index_calculis_R].append(j)
            for k1 in range(len(c_sagittal)):
                if(j in c_sagittal[k1]):
                    c_sagittal[k1].remove(j)
                if(c_sagittal.__contains__([])):
                    c_sagittal = list(filter(None, c_sagittal))
                    index_calculis_R+=1
                    Calculis_R+=[[]]
                    break
            for k2 in range(len(c_coronal)):
                if(j in c_coronal[k2]):
                    c_coronal[k2].remove(j)
                if(c_coronal.__contains__([])):
                    c_coronal = list(filter(None, c_coronal))
                    index_calculis_R+=1
                    Calculis_R+=[[]]
                    break
            
        index_calculis_R+=1
        Calculis_R+=[[]]        
    
    Calculis_R = list(filter(None, Calculis_R))    
    #final=time.perf_counter()
    #time_duration=final-initial
    #print(time_duration)


    ###############################

    #initial=time.perf_counter()
    d_coronal = list(filter(None, d_coronal))
    d_sagittal = list(filter(None, d_sagittal))
    d = list(filter(None, d))

    index_calculis_L=0
    Calculis_L=[[]]
    for i in range(len(d)):
        for j in d[i]:
            Calculis_L[index_calculis_L].append(j)
            for k1 in range(len(d_sagittal)):
                if(j in d_sagittal[k1]):
                    d_sagittal[k1].remove(j)
                if(d_sagittal.__contains__([])):
                    d_sagittal = list(filter(None, d_sagittal))
                    index_calculis_L+=1
                    Calculis_L+=[[]]
                    break
            for k2 in range(len(d_coronal)):
                if(j in d_coronal[k2]):
                    d_coronal[k2].remove(j)
                if(d_coronal.__contains__([])):
                    d_coronal = list(filter(None, d_coronal))
                    index_calculis_L+=1
                    Calculis_L+=[[]]
                    break
            
        index_calculis_L+=1
        Calculis_L+=[[]]        
            
    Calculis_L = list(filter(None, Calculis_L))
    final=time.perf_counter()
    time_duration=final-initial
    print("Time taken for Storing Calculi: ",time_duration)
    ############################################

    #Calculi Further Separated from last one
    #Calculi_L_all

    #Calculi_L_all

    Calculi_Left_OfAll=[]
    for i in Calculis_L:
        for j in i:
            Calculi_Left_OfAll.append(j)

    Calculi_Left_OfAll_check=[]
    def remove_Left_Calculi_cords(pxyz):
        Calculi_Left_OfAll_check.append(pxyz)
        if((((pxyz[0]-1),pxyz[1],pxyz[2]) in Calculi_Left_OfAll) and (((pxyz[0]-1),pxyz[1],pxyz[2]) not in Calculi_Left_OfAll_check)):
            remove_Left_Calculi_cords(((pxyz[0]-1),pxyz[1],pxyz[2]))
            Calculi_Left_OfAll.remove(((pxyz[0]-1),pxyz[1],pxyz[2]))
        if(((pxyz[0],(pxyz[1]-1),pxyz[2]) in Calculi_Left_OfAll) and ((pxyz[0],(pxyz[1]-1),pxyz[2]) not in Calculi_Left_OfAll_check)):
            remove_Left_Calculi_cords((pxyz[0],(pxyz[1]-1),pxyz[2]))
            Calculi_Left_OfAll.remove((pxyz[0],(pxyz[1]-1),pxyz[2]))
        if(((pxyz[0],pxyz[1],(pxyz[2]-1)) in Calculi_Left_OfAll) and ((pxyz[0],pxyz[1],(pxyz[2]-1)) not in Calculi_Left_OfAll_check)):
            remove_Left_Calculi_cords((pxyz[0],pxyz[1],(pxyz[2]-1)))
            Calculi_Left_OfAll.remove((pxyz[0],pxyz[1],(pxyz[2]-1)))
        if((((pxyz[0]+1),pxyz[1],pxyz[2]) in Calculi_Left_OfAll) and (((pxyz[0]+1),pxyz[1],pxyz[2]) not in Calculi_Left_OfAll_check)):
            remove_Left_Calculi_cords(((pxyz[0]+1),pxyz[1],pxyz[2]))
            Calculi_Left_OfAll.remove(((pxyz[0]+1),pxyz[1],pxyz[2]))
        if(((pxyz[0],(pxyz[1]+1),pxyz[2]) in Calculi_Left_OfAll) and ((pxyz[0],(pxyz[1]+1),pxyz[2]) not in Calculi_Left_OfAll_check)):
            remove_Left_Calculi_cords((pxyz[0],(pxyz[1]+1),pxyz[2]))
            Calculi_Left_OfAll.remove((pxyz[0],(pxyz[1]+1),pxyz[2]))
        if(((pxyz[0],pxyz[1],(pxyz[2]+1)) in Calculi_Left_OfAll) and ((pxyz[0],pxyz[1],(pxyz[2]+1)) not in Calculi_Left_OfAll_check)):
            remove_Left_Calculi_cords((pxyz[0],pxyz[1],(pxyz[2]+1)))
            Calculi_Left_OfAll.remove((pxyz[0],pxyz[1],(pxyz[2]+1)))

    Calculis_L=[]    
    while(Calculi_Left_OfAll!=[]):
        Calculi_Left_OfAll_check=[]
        remove_Left_Calculi_cords(Calculi_Left_OfAll[0])
        Calculi_Left_OfAll.remove(Calculi_Left_OfAll[0])
        Calculis_L.append(Calculi_Left_OfAll_check)
        

    Calculis_L = list(filter(None, Calculis_L))

    #import sys
    #print(sys.getrecursionlimit())
    #sys.setrecursionlimit(3200)
    ########################
    #Calculi_R_all
    Calculi_Right_OfAll=[]
    for i in Calculis_R:
        for j in i:
            Calculi_Right_OfAll.append(j)

    Calculi_Right_OfAll_check=[]
    def remove_Right_Calculi_cords(pxyz):
        Calculi_Right_OfAll_check.append(pxyz)
        if((((pxyz[0]-1),pxyz[1],pxyz[2]) in Calculi_Right_OfAll) and (((pxyz[0]-1),pxyz[1],pxyz[2]) not in Calculi_Right_OfAll_check)):
            remove_Right_Calculi_cords(((pxyz[0]-1),pxyz[1],pxyz[2]))
            Calculi_Right_OfAll.remove(((pxyz[0]-1),pxyz[1],pxyz[2]))
        if(((pxyz[0],(pxyz[1]-1),pxyz[2]) in Calculi_Right_OfAll) and ((pxyz[0],(pxyz[1]-1),pxyz[2]) not in Calculi_Right_OfAll_check)):
            remove_Right_Calculi_cords((pxyz[0],(pxyz[1]-1),pxyz[2]))
            Calculi_Right_OfAll.remove((pxyz[0],(pxyz[1]-1),pxyz[2]))
        if(((pxyz[0],pxyz[1],(pxyz[2]-1)) in Calculi_Right_OfAll) and ((pxyz[0],pxyz[1],(pxyz[2]-1)) not in Calculi_Right_OfAll_check)):
            remove_Right_Calculi_cords((pxyz[0],pxyz[1],(pxyz[2]-1)))
            Calculi_Right_OfAll.remove((pxyz[0],pxyz[1],(pxyz[2]-1)))
        if((((pxyz[0]+1),pxyz[1],pxyz[2]) in Calculi_Right_OfAll) and (((pxyz[0]+1),pxyz[1],pxyz[2]) not in Calculi_Right_OfAll_check)):
            remove_Right_Calculi_cords(((pxyz[0]+1),pxyz[1],pxyz[2]))
            Calculi_Right_OfAll.remove(((pxyz[0]+1),pxyz[1],pxyz[2]))
        if(((pxyz[0],(pxyz[1]+1),pxyz[2]) in Calculi_Right_OfAll) and ((pxyz[0],(pxyz[1]+1),pxyz[2]) not in Calculi_Right_OfAll_check)):
            remove_Right_Calculi_cords((pxyz[0],(pxyz[1]+1),pxyz[2]))
            Calculi_Right_OfAll.remove((pxyz[0],(pxyz[1]+1),pxyz[2]))
        if(((pxyz[0],pxyz[1],(pxyz[2]+1)) in Calculi_Right_OfAll) and ((pxyz[0],pxyz[1],(pxyz[2]+1)) not in Calculi_Right_OfAll_check)):
            remove_Right_Calculi_cords((pxyz[0],pxyz[1],(pxyz[2]+1)))
            Calculi_Right_OfAll.remove((pxyz[0],pxyz[1],(pxyz[2]+1)))

    Calculis_R=[]    
    while(Calculi_Right_OfAll!=[]):
        Calculi_Right_OfAll_check=[]
        remove_Right_Calculi_cords(Calculi_Right_OfAll[0])
        Calculi_Right_OfAll.remove(Calculi_Right_OfAll[0])
        Calculis_R.append(Calculi_Right_OfAll_check)
        

    Calculis_R = list(filter(None, Calculis_R))
    #Slice Coordinates - Calculi
    sliding_min_max_L=[]
    for i in Calculis_L:
        sliding_min=max(data1.shape[0],data1.shape[1],data1.shape[2])
        sliding_max=-1
        for j in i:
            if(sliding_min > j[2]):
                sliding_min = j[2]
            if(sliding_max < j[2]):
                sliding_max = j[2]
        sliding_min_max_L.append([data1.shape[2]-1-sliding_min,data1.shape[2]-1-sliding_max])
    sliding_min_max_L.reverse()

    sliding_min_max_R=[]
    for i in Calculis_R:
        sliding_min=max(data1.shape[0],data1.shape[1],data1.shape[2])
        sliding_max=-1
        for j in i:
            if(sliding_min > j[2]):
                sliding_min = j[2]
            if(sliding_max < j[2]):
                sliding_max = j[2]
        sliding_min_max_R.append([data1.shape[2]-1-sliding_min,data1.shape[2]-1-sliding_max])
    sliding_min_max_R.reverse()

    ##### Calculi Volume #####

    initial=time.perf_counter()
    Calculi_VOLUMES_Right = []
    for i in range(len(Calculis_R)):
        Calculi_VOLUMES_Right.append(round((len(Calculis_R[i])*voxel_volume1)/1000,5))
    print("Calculi_VOLUMES_Right",Calculi_VOLUMES_Right)

    Calculi_VOLUMES_Left = []
    for i in range(len(Calculis_L)):
        Calculi_VOLUMES_Left.append(round((len(Calculis_L[i])*voxel_volume1)/1000,5))
    print("Calculi_VOLUMES_Left", Calculi_VOLUMES_Left)
    final=time.perf_counter()
    time_duration=final-initial
    print("Time taken for Calculating Individual Calculi Volumes: ",time_duration)


    ##### Average HU #####

    initial=time.perf_counter()
    threshold = 45
    Calculi_HU_Right = []
    for i in range(len(Calculis_R)):
        Calculi_HU_Right_Value = 0
        threshold_count=0
        for j in (Calculis_R[i]):
            Calculi_HU_Right_Value+= (0 if (ct_data[2*j[0], 2*j[1], j[2]]<threshold) else ct_data[2*j[0], 2*j[1], j[2]])+(0 if (ct_data[2*j[0]+1, 2*j[1], j[2]]<threshold) else ct_data[2*j[0]+1, 2*j[1], j[2]])+(0 if (ct_data[2*j[0], 2*j[1]+1, j[2]]<threshold) else ct_data[2*j[0], 2*j[1]+1, j[2]])+(0 if (ct_data[2*j[0]+1, 2*j[1]+1, j[2]]<threshold) else ct_data[2*j[0]+1, 2*j[1]+1, j[2]])
            threshold_count+= (0 if (ct_data[2*j[0], 2*j[1], j[2]]<threshold) else 1)+(0 if (ct_data[2*j[0]+1, 2*j[1], j[2]]<threshold) else 1)+(0 if (ct_data[2*j[0], 2*j[1]+1, j[2]]<threshold) else 1)+(0 if (ct_data[2*j[0]+1, 2*j[1]+1, j[2]]<threshold) else 1)
        #print("threshold_count",threshold_count)
        Calculi_HU_Right.append((Calculi_HU_Right_Value)/(threshold_count))
    print("Calculi_HU_Right", Calculi_HU_Right)
        
    Calculi_HU_Left = []
    for i in range(len(Calculis_L)):
        Calculi_HU_Left_Value = 0
        threshold_count=0
        for j in (Calculis_L[i]):
            Calculi_HU_Left_Value+= (0 if (ct_data[2*j[0], 2*j[1], j[2]]<threshold) else ct_data[2*j[0], 2*j[1], j[2]])+(0 if (ct_data[2*j[0]+1, 2*j[1], j[2]]<threshold) else ct_data[2*j[0]+1, 2*j[1], j[2]])+(0 if (ct_data[2*j[0], 2*j[1]+1, j[2]]<threshold) else ct_data[2*j[0], 2*j[1]+1, j[2]])+(0 if (ct_data[2*j[0]+1, 2*j[1]+1, j[2]]<threshold) else ct_data[2*j[0]+1, 2*j[1]+1, j[2]])
            threshold_count+= (0 if (ct_data[2*j[0], 2*j[1], j[2]]<threshold) else 1)+(0 if (ct_data[2*j[0]+1, 2*j[1], j[2]]<threshold) else 1)+(0 if (ct_data[2*j[0], 2*j[1]+1, j[2]]<threshold) else 1)+(0 if (ct_data[2*j[0]+1, 2*j[1]+1, j[2]]<threshold) else 1)
        #print("threshold_count",threshold_count)
        Calculi_HU_Left.append((Calculi_HU_Left_Value)/(threshold_count))  
    print("Calculi_HU_Left",Calculi_HU_Left)
    final=time.perf_counter()
    time_duration=final-initial
    print("Time taken for Calculating Individual Calculi HU: ",time_duration)
        
    ####### Max Dimension ########


    initial=time.perf_counter()

    #Cal_R_Min = ()
    #Cal_R_Max = ()
    Cal_Max_dim_R = []  
    for count in range(len(Calculis_R)):
        Max_R = -1         
        for i in Calculis_R[count]:
            for j in Calculis_R[count]:
                distance_right_ = math.dist([i[0]*x1,i[1]*y1,i[2]*z1], [j[0]*x1,j[1]*y1,j[2]*z1])
                if(Max_R<distance_right_):
                    Max_R = distance_right_
                    #Cal_R_Min = (i[0]*x1,i[1]*y1,i[2]*z1)
                    #Cal_R_Max = (j[0]*x1,j[1]*y1,j[2]*z1)
        Cal_Max_dim_R.append(Max_R)
        #Cal_Max_dim_R.append(Cal_R_Min)
        #Cal_Max_dim_R.append(Cal_R_Max)
    print("Cal_Max_dim_R",Cal_Max_dim_R)           


    #####
    #Cal_L_Min = ()
    #Cal_L_Max = ()
    Cal_Max_dim_L = []   
    for count in range(len(Calculis_L)):
        Max_L = -1         
        for i in Calculis_L[count]:
            for j in Calculis_L[count]:
                distance_left_ = math.dist([i[0]*x1,i[1]*y1,i[2]*z1], [j[0]*x1,j[1]*y1,j[2]*z1])
                if(Max_L<distance_left_):
                    Max_L = distance_left_
                    #Cal_L_Min = (i[0]*x1,i[1]*y1,i[2]*z1)
                    #Cal_L_Max = (j[0]*x1,j[1]*y1,j[2]*z1)
        Cal_Max_dim_L.append(Max_L)
        #Cal_Max_dim_R.append(Cal_L_Min)
        #Cal_Max_dim_R.append(Cal_L_Max)
    print("Cal_Max_dim_R",Cal_Max_dim_L)           
                
    final=time.perf_counter()
    time_duration=final-initial
    print("Time taken for Calculating Individual Calculi Max Dimension: ",time_duration)   

    # final_finding=time.perf_counter()
    # time_duration_finding=final_finding-initial_finding
    # print("Total Time taken for findings:",time_duration_finding) 
    
    
    
    
    return volume1,distance_left,distance_right,Calculi_VOLUMES_Right,Calculi_VOLUMES_Left,Calculi_HU_Right,Calculi_HU_Left,Cal_Max_dim_R,Cal_Max_dim_L,sliding_min_max_L,sliding_min_max_R

    #Volume = volume(img)


    # def getPrediction(filename):
    #     print("Hi, I am inside Prediction function!")
    #     classes = ['Actinic keratoses', 'Basal cell carcinoma', 
    #             'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
    #             'Melanocytic nevi', 'Vascular lesions']
    #     le = LabelEncoder()
    #     le.fit(classes)
    #     le.inverse_transform([2])
        
    
    # #Load model
    # my_model=load_model("model/HAM10000_100epochs.h5 ")
    # print("Hi, model loaded!")
    # SIZE = 32 #Resize to same size as training images
    # img_path = 'static/images/'+filename
    # print("Image_Path", img_path)
    # img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    # img = img/255.      #Scale pixel values
    
    # img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    # print("Here")
    # pred = my_model.predict(img) #Predict                    
    # print("After Prediction")
    # print("pred", pred)
    # #Convert prediction to class name
    # pred_class = le.inverse_transform([np.argmax(pred)])[0]
    # print("Diagnosis is:", pred_class)
    # return pred_class


#test_prediction = getPrediction('Capture.JPG')
