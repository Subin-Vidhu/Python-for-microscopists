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

from patchify import patchify, unpatchify
from matplotlib import pyplot as plt
import keras 

def getPrediction1(filename):
    print("Hi, I am inside Prediction function!")

    #from keras.models import load_model
    model1 = load_model('D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/model/binary_pathology_resnet_34_160_images_annotated_only.hdf5', compile=False)


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
        f = f.replace("//","/")
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
    os.mkdir(img_path)
    print("path1", img_path)
    path1 = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/images/" + filename
    tiff_image = tiff.imread(path1)
    print("tiff_image", tiff_image)

    print("large_image.shape", tiff_image.shape)
    large_image = cv2.resize(tiff_image, (1792, 768))
    print("resized_image.shape", large_image.shape)
    path_pred = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/Predicted/" + filename.split(".")[0] +".png"
    print("path_pred", path_pred)

    cv2.imwrite(path_pred, large_image)

    count = 0
    patches_img = patchify(large_image, (256, 256,3), step=256)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]
            count = count +1
            tiff.imwrite(img_path + '/image(' + str(count)+ ").tif", single_patch_img)
            print(count)
    
    import segmentation_models as sm
    BACKBONE1 = 'resnet34'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)

    ###################### Read Images and Mask from the Patches folder ######       

    #path_RC1 = path2

    path_RC6 = img_path

    train_images = []
    #train_masks = [] 
    SIZE = 256

    for i in range(1, 22):
        print(i)
        #plt.figure(figsize=(30, 25))
        
        #images
        #img1 = cv2.imread(path_RC1+"/mask("+str(i)+").tif", 0)
        #mask = img1
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #mask = cv2.resize(mask, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)  
        #train_masks.append(mask)


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
    #train_masks = np.array(train_masks)



    ### Reading images to predict and saving it to folder 
    path_RC6 = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/Predicted/"
    image = filename.split(".")[0]   #### Change the image name ################
    path3 = os.path.join(path_RC6, image)
    os.mkdir(path3)


    for i in range(0, 21):
        test_img_number = i
        test_img = train_images[test_img_number]
        #ground_truth=train_masks[test_img_number]
        test_img_input=np.expand_dims(test_img, 0)

        test_img_input1 = preprocess_input1(test_img_input)

        test_pred1 = model1.predict(test_img_input1)
        test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


        #plt.figure(figsize=(20, 15))
        ''' plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_img[:,:,0], cmap='gray')
        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(ground_truth[:,:,0], cmap='jet')
        plt.subplot(233)
        plt.title('Prediction on test image')'''
        #plt.axis("off")
        #plt.imshow(test_prediction1, cmap='jet')
        cv2.imwrite(path3+"/"+"image("+str(i+1)+").png",test_prediction1)
        #plt.show()

############################ Unpatchify  ####################################

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


    path_op = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/Results/" 
    #path_op = os.path.join(path_op,image)
    #os.mkdir(path_op)
    reconstructed_image = unpatchify(predicted_patches_reshaped, (768,1792))
    #plt.figure(figsize=(30, 25))
    #plt.axis("off")
    #plt.imshow(reconstructed_image, cmap='gray')
    cv2.imwrite(path_op + filename.split(".")[0] +".png",reconstructed_image)
    #plt.show()



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


