# https://youtu.be/pI0wQbJwIIs

"""
For training, watch videos (202 and 203): 
    https://youtu.be/qB6h5CohLbs
    https://youtu.be/fyZ9Rxpoz2I

The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)

"""
import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import glob
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from matplotlib import pyplot as plt
import keras 

def getPrediction1(filename):
    print("Hi, I am inside Prediction function!")



    # importing os module
    import os
    
    # Directory
    #directory = "RC"
    
    # Parent Directory path
    parent_dir_img = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/Patchify/"
    #parent_dir_mask = "D:/Pathology_23_5_22/single_msk_patchify_new/"

    #Remove directories to start as 0
    """path = os.path.join(parent_dir_img + "*")
    files = glob.glob(path)
    for f in files:
        f = f.replace("\\","/")
        print(f, "f")
        os.remove(f)"""
    # Path
    #path = os.path.join(parent_dir, directory)
    
    # Create the directory
    # 'GeeksForGeeks' in
    # '/home / User / Documents'
    #os.mkdir(path)
    #print("Directory '% s' created" % directory)
    
    img_path = parent_dir_img + filename #Parent dir defined above and filename the name of the input image
    print("Image_Path", img_path)
    #directory_mask = "A02B74X11_d_mask" #### replace the name of the mask
    img_path = os.path.join(img_path)
    #os.mkdir(img_path)
    print("path1", img_path)
    path1 = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/images/" + filename
    tiff_image = tiff.imread(path1)
    print("tiff_image", tiff_image)

    print("large_image.shape", tiff_image.shape)
    large_image = cv2.resize(tiff_image, (1792, 768))
    print("resized_image.shape", large_image.shape)
    path_pred = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/Predicted/" + filename.split(".")[0] +".png"
    print("path_pred", path_pred)

    cv2.imwrite(path_pred, large_image)

    

def getPrediction(filename):
    print("Hi, I am inside Prediction function!")
    classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    
    #Load model
    my_model=load_model("model/HAM10000_100epochs.h5 ")
    print("Hi, model loaded!")
    SIZE = 32 #Resize to same size as training images
    img_path = 'static/images/'+filename
    print("Image_Path", img_path)
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    print("Here")
    pred = my_model.predict(img) #Predict                    
    print("After Prediction")
    print("pred", pred)
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is:", pred_class)
    return pred_class


#test_prediction = getPrediction('Capture.JPG')


