# https://www.youtube.com/watch?v=uDNqNv2N-pY&t=

###########################
#Reading D:/DS_python/Python-for-microscopists/images
from PIL import Image 
  
img = Image.open("D:/DS_python/Python-for-microscopists/images/test_image.jpg") #Not a numpy array
print(type(img))
# show D:/DS_python/Python-for-microscopists/images on external default viewer. This can be paint or photo viewer on Windows
img.show() 
 
# prints format of image 
print(img.format) 
  
# prints mode of image RGB or CMYK
print(img.mode) 

print(img. size)  #prints the size of image (wodth, height)

# Resize D:/DS_python/Python-for-microscopists/images
small_img = img.resize((200, 300))
small_img.save("D:/DS_python/Python-for-microscopists/images/test_image_small.jpg")  #squished image

# resize() method resizes D:/DS_python/Python-for-microscopists/images to exact value whether it makes sense or not.
#aspect ratio is not maintained so D:/DS_python/Python-for-microscopists/images are squished.
#if you want to keep the aspect ration then use thumbnai() method

img.thumbnail((200, 200))
img.save("D:/DS_python/Python-for-microscopists/images/test_image_small_new.jpg")

print(img.size)

img.thumbnail((1200, 1200))  #doesn't blow up the image, only reduces the size if original is larger.
img.save("D:/DS_python/Python-for-microscopists/images/test_image_small_new1.jpg")  

large_img = img.resize((1200, 1300))
large_img.save("D:/DS_python/Python-for-microscopists/images/test_image_large.jpg")  #enlarged image. 
print(large_img.size)

#Cropping D:/DS_python/Python-for-microscopists/images
from PIL import Image 
img = Image.open("D:/DS_python/Python-for-microscopists/images/test_image.jpg")
cropped_img = img.crop((0, 0, 300, 300))  #crops from (0,0) to (300,300)
cropped_img.save("D:/DS_python/Python-for-microscopists/images/cropped_img.jpg")

# We can paste image on another image
#this involves copying original image and pasting a second image on it
from PIL import Image 
img1 = Image.open("D:/DS_python/Python-for-microscopists/images/test_image.jpg")
print(img1.size)
img2 = Image.open("D:/DS_python/Python-for-microscopists/images/monkey.jpg")
print(img2.size)
img2.thumbnail((200, 200))  #Resize in case the image is very large. 

img1_copy = img1.copy()   #Create a copy of the large image
img1_copy.paste(img2, (50, 50))  #Paset the smaller imager image at specified location
img1_copy.save("D:/DS_python/Python-for-microscopists/images/pasted_image.jpg")

# Rotating D:/DS_python/Python-for-microscopists/images
from PIL import Image 
img = Image.open("D:/DS_python/Python-for-microscopists/images/test_image.jpg")

img_90_rot = img.rotate(90)
img_90_rot.save("D:/DS_python/Python-for-microscopists/images/rotated90.jpg")  #keeps original aspect ratio and dimensions

img_45_rot = img.rotate(45)
img_45_rot.save("D:/DS_python/Python-for-microscopists/images/rotated45.jpg")  #keeps original aspect ratio and dimensions

img_45_rot = img.rotate(45, expand=True)  #Dimensions are expanded to fir the entire image
img_45_rot.save("D:/DS_python/Python-for-microscopists/images/rotated45.jpg")  


#Flipping or transposing D:/DS_python/Python-for-microscopists/images

from PIL import Image 
img = Image.open("D:/DS_python/Python-for-microscopists/images/monkey.jpg")  #easy to see that the image is flipped

img_flipLR = img.transpose(Image.FLIP_LEFT_RIGHT)
img_flipLR.save("D:/DS_python/Python-for-microscopists/images/flippedLR.jpg")

img_flipTB = img.transpose(Image.FLIP_TOP_BOTTOM)
img_flipTB.save("D:/DS_python/Python-for-microscopists/images/flippedTB.jpg")

# Color transforms, convert D:/DS_python/Python-for-microscopists/images between L (greyscale), RGB and CMYK
from PIL import Image 
img = Image.open("D:/DS_python/Python-for-microscopists/images/test_image.jpg")

grey_img = img.convert('L')  #L is for grey scale
grey_img.save("D:/DS_python/Python-for-microscopists/images/grey_img.jpg")

# Many other tasks can be performed. Here is full documentation.
# https://pillow.readthedocs.io/en/stable/reference/Image.html


#Here is a way to automate image processing for multiple D:/DS_python/Python-for-microscopists/images.

from PIL import Image 
import glob

path = "D:/DS_python/Python-for-microscopists/images/test_D:/DS_python/Python-for-microscopists/images/aeroplane/*.*"
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    a= Image.open(file)  #now, we can read each file since we have the full path
    
    rotated45 = a.rotate(45, expand=True)
    rotated45.save(file+"_rotated45.png", "PNG")   













