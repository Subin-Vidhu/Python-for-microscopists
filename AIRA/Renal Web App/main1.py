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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def getPrediction1(filename,slice_START, slice_END):
    print("Hi, I am inside Prediction function!")
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
    test_image_gt_RC1 = np.clip(test_image_gt_RC1,-135,215)
    test_image_gt_RC1=scaler.fit_transform(test_image_gt_RC1.reshape(-1, test_image_gt_RC1.shape[-1])).reshape(test_image_gt_RC1.shape)
    test_image_gt_RC1.shape
    for i in range(0, len(test_image_gt_RC1)):
        test_image_gt_RC1[i,:,:] = cv2.flip(test_image_gt_RC1[i,:,:], 0)
        
        
    ground_truth_array_test = []
    img_size = 256
    for pointer in range(test_image_gt_RC1.shape[0]):
        image_groundtruth_test = test_image_gt_RC1[pointer,:,:]
        #image_groundtruth_test = cv2.resize(test_image_gt_RC1[pointer,:,:], (img_size,img_size), cv2.IMREAD_COLOR)
        ground_truth_array_test.append(image_groundtruth_test)
        
    ground_truth_array_test = np.array(ground_truth_array_test)
    #ground_truth_array_test = np.expand_dims(ground_truth_array_test, 3)
    print("ground_truth_array_test_dimensions", ground_truth_array_test.shape)    


    original = []
    preprocessed = []
    #append slices

    ###### for testing the performance of the model
    if slice_END > len(ground_truth_array_test):
        slice_END = len(ground_truth_array_test)

    slice_START = len(ground_truth_array_test)- slice_START
    slice_END = len(ground_truth_array_test) - slice_END


    for i in range(slice_END, slice_START):
        
        test_image_gt_one1 = ground_truth_array_test[i,:,:]
        original.append(test_image_gt_one1)
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
        
    return volume1

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
