# https://youtu.be/pI0wQbJwIIs

"""
Werkzeug provides a bunch of utilities for developing WSGI-compliant applications. 
These utilities do things like parsing headers, sending and receiving cookies, 
providing access to form data, generating redirects, generating error pages when 
there's an exception, even providing an interactive debugger that runs in the browser. 
Flask then builds upon this foundation to provide a complete web framework.
"""

from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from main import getPrediction, getPrediction1
import os
import glob


#Save images to the 'static' folder as Flask serves images from this directory
UPLOAD_FOLDER = 'static/images/'
UPLOAD_FOLDER_PRED = 'static/Predicted/'
UPLOAD_FOLDER_RESULT = 'static/Unpatchified_Result/'
#Create an app object using the Flask class. 
app = Flask(__name__, static_folder="static")



#Add reference fingerprint. 
#Cookies travel with a signature that they claim to be legit. 
#Legitimacy here means that the signature was issued by the owner of the cookie.
#Others cannot change this cookie as it needs the secret key. 
#It's used as the key to encrypt the session - which can be stored in a cookie.
#Cookies should be encrypted if they contain potentially sensitive information.
app.secret_key = "secret key"

#Define the upload folder to save images uploaded by the user. 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['Predicted'] = UPLOAD_FOLDER_PRED
app.config['RESULT'] = UPLOAD_FOLDER_RESULT
#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, index function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 
@app.route('/')
def index():
    return render_template('index.html')

#Add Post method to the decorator to allow for form submission. 
@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            print("file inserted", file)
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            print("secure file inserted", filename)
            #Remove files inside these folders
            print("Removing files inside these folders")
            files = glob.glob('D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/images/*')
            for f in files:
                os.remove(f)
            files = glob.glob('D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/Predicted/*')
            for f in files:
                f = f.replace("\\","/")
                os.remove(f)
            files = glob.glob('D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/Unpatchified_Result/*')
            for f in files:
                f = f.replace("\\","/")
                os.remove(f)

            #Removes files (files in subdirectories) from Patchify and Results folders
            mydir = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker//static/Results"
            filelist = [ f for f in os.listdir(mydir) ]
            for f in filelist:
                print(f)
                path = os.path.join(mydir, f)
                print(path)
                path = path.replace("\\","/")
                print(path)
                files = glob.glob(path+'/*')
                for f in files:
                    print(f)
                    f = f.replace("\\","/")
                    print(f)
                    os.remove(f)     

            mydir = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/Patchify"
            filelist = [ f for f in os.listdir(mydir) ]
            for f in filelist:
                print(f)
                path = os.path.join(mydir, f)
                print(path)
                path = path.replace("\\","/")
                print(path)
                files = glob.glob(path+'/*')
                for f in files:
                    print(f)
                    f = f.replace("\\","/")
                    print(f)
                    os.remove(f)     
            #Remove folders inside these folders
            print("Removing folders inside these folders")
            mydir = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/static/Results"
            filelist = [ f for f in os.listdir(mydir) ]
            for f in filelist:
                #print("f", f)
                path = os.path.join(mydir, f)
                path = path.replace("\\", "/")
                #print("path", path)
                os.rmdir(path)
            mydir = "D:/DS_python/Python-for-microscopists/270-How to deploy your trained machine learning model as an app on Heroku-HAM10000-no docker/Patchify"
            filelist = [ f for f in os.listdir(mydir) ]
            for f in filelist:
                #print("f", f)
                path = os.path.join(mydir, f)
                path = path.replace("\\", "/")
                #print("path", path)
                os.rmdir(path)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            #getPrediction(filename)
            label = getPrediction1(filename)
            print("label", label)
            flash(label)
            filename_pred = filename.split(".")[0] +".png"
            image = os.path.join(app.config['Predicted'], filename_pred)
            flash(image)
            full_filename = os.path.join(app.config['RESULT'], filename_pred)
            print("full filename: ", full_filename)
            flash(full_filename)
            return redirect('/')


if __name__ == "__main__":
    app.run()