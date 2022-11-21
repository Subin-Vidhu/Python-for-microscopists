from flask import Flask, render_template, request, redirect, flash,session
from werkzeug.utils import secure_filename
from main import getPrediction1
import os
import glob
import shutil
import numpy as np
import dicom2nifti
import patoolib
import zipfile
import MySQLdb
 

#Save images to the 'static' folder as Flask serves images from this directory
UPLOADED_IMAGE = 'static/images/'
UPLOADED_IMAGE_COPY = 'static/Image_Copy/'
RESULT_FILE = 'static/Results/'
RESULT_COPY = 'static/Result_Copy/'
UPLOAD_GZ= 'static/Gz_or_Rar/'
RAR_EXTRACT= 'static/rar/'
GZ_EXTRACT= 'static/Zip_gz/'
UPLOAD_FOLDER_DICOM = 'static/DICOM/'
UPLOAD_FOLDER_Zip_Extract = 'static/Zip_Extract/'
UPLOAD_FOLDER_Zip_Zip = 'static/Zip_zip/'

#Create an app object using the Flask class. 
app = Flask(__name__, static_folder="static")
db= MySQLdb.connect("deltadb.cia0oqklelkq.ap-south-1.rds.amazonaws.com","admin","delta#pass1","protos")



#Add reference fingerprint. 
#Cookies travel with a signature that they claim to be legit. 
#Legitimacy here means that the signature was issued by the owner of the cookie.
#Others cannot change this cookie as it needs the secret key. 
#It's used as the key to encrypt the session - which can be stored in a cookie.
#Cookies should be encrypted if they contain potentially sensitive information.
app.secret_key = "secret key"

#Define the upload folder to save images uploaded by the user. 
app.config['UPLOADED_IMAGE'] = UPLOADED_IMAGE
app.config['UPLOADED_IMAGE_COPY'] = UPLOADED_IMAGE_COPY
app.config['RESULT_COPY'] = RESULT_COPY
app.config['RESULT_FILE'] = RESULT_FILE
app.config['UPLOAD_GZ'] = UPLOAD_GZ
app.config['GZ_EXTRACT'] = GZ_EXTRACT
app.config['RAR_EXTRACT'] = RAR_EXTRACT
app.config['Zip_Extract'] = UPLOAD_FOLDER_Zip_Extract
app.config['Zip_zip'] = UPLOAD_FOLDER_Zip_Zip
app.config['Dicom'] = UPLOAD_FOLDER_DICOM
#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, index function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def index0():
    if not session.get("userid"):
        return redirect("/login")
    return render_template('home.html')
@app.route('/viewfile')
def index1():
    if not session.get("userid"):
        return redirect("/login")
    return render_template('view.html')
@app.route('/result')
def index2():
    if not session.get("userid"):
        return redirect("/login")
    return render_template('result.html')
@app.route('/volume_render')
def index5():
    if not session.get("userid"):
        return redirect("/login")
    return render_template('volumerender.html')
@app.route('/login')
def index6():
    return render_template('login.html')
@app.route('/userlogin')
def index8():
    return render_template('userlogin.html')
@app.route('/interpolation')
def index9():
    if not session.get("userid"):
        return redirect("/login")
    return render_template('interpolation.html')
@app.route('/expandview')
def index10():
    if not session.get("userid"):
        return redirect("/login")
    return render_template('expandable.html')
@app.route('/signup', methods=['POST'])
def submit_file0():
    if request.method == 'POST':
        try:
            features = list(request.form.values()) 
            uid=0
            email = features[0]
            username = features[1]
            password = features[2]
            cursor= db.cursor()
            cursor.execute("SELECT user_id FROM user where user_name='"+username+"' or user_email='"+email+"'")
            rows = cursor.fetchall()
            for row in rows:
                uid=row[0]
            if(uid==0):
                cursor1= db.cursor()
                cursor1.execute("insert into user(user_name, user_email,user_password) value(%s, %s, %s)", (username,email, password))
                db.commit()
            else:
                flash("User email or Username already exists")
                return redirect('/login')
            return render_template('login.html')
        except:
          db.rollback()
          db.close() 
@app.route('/login', methods=['POST'])
def submit_file02():
    if request.method == 'POST':
        try:
            lid=0
            features = list(request.form.values()) 
            username = features[0]
            password = features[1]
            cursor= db.cursor()
            cursor.execute("SELECT user_id FROM user where user_name='"+username+"' and user_password='"+password+"'")
            rows = cursor.fetchall()
            for row in rows:
                lid=row[0]
            db.commit()
            if(lid!=0):
                session['userid']=lid
                session['username']=username
                return redirect('/home')
            else:
                flash("Incorrect Username or Password")
                return redirect('/login')
        except:
          db.rollback()
          db.close()
@app.route('/user_login', methods=['POST'])
def submit_file03():
    if request.method == 'POST':
        try:
            lid=0
            features = list(request.form.values()) 
            username = features[0]
            password = features[1]
            cursor= db.cursor()
            cursor.execute("SELECT user_id FROM user where user_name='"+username+"' and user_password='"+password+"'")
            rows = cursor.fetchall()
            for row in rows:
                lid=row[0]
            db.commit()
            if(lid!=0):
                session['userid']=lid
                session['username']=username
                return redirect('/')
            else:
                flash("Incorrect Username or Password")
                return redirect('/userlogin')
        except:
          db.rollback()
          db.close()
@app.route('/logout')
def index3():
    userid = session['userid']
    userid=str(userid)
    mydir=app.config['UPLOADED_IMAGE_COPY']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir=app.config['RESULT_COPY']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir=app.config['UPLOAD_GZ']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir=app.config['RESULT_FILE']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir=app.config['Dicom']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir=app.config['GZ_EXTRACT']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir=app.config['Zip_zip']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir = app.config['RAR_EXTRACT']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir = app.config['Zip_Extract']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    mydir=app.config['UPLOADED_IMAGE']+userid+'/'
    if os.path.exists(mydir):
        shutil.rmtree(mydir)
    session["userid"] = None
    session["username"] = None
    return render_template('login.html')
@app.route('/signout')
def index7():
    session["userid"] = None
    session["username"] = None
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def submit_file1():
    if request.method == 'POST':
        int_features = [(x) for x in request.form.values()] #Convert string inputs to float.
        features = list(request.form.values()) 
        slice_START = int(features[0])
        slice_END = int(features[1])
        userid = session['userid']
        userid=str(userid)
        # print("Features, slice_START, slice_END", list(request.form.values()), slice_START, type(slice_START), slice_END, type(slice_END))
        #getPrediction(filename)
        # print("Got inside save function")
        mydir = app.config['UPLOADED_IMAGE']+userid
        file_dir = [ f for f in os.listdir(mydir) ]
        print("filename", file_dir[0])
        Volume,Maximum_Cranio_caudal_length_left,Maximum_Cranio_caudal_length_right,Calculi_VOLUMES_Right,Calculi_VOLUMES_Left,Calculi_HU_Right,Calculi_HU_Left,Cal_Max_dim_R,Cal_Max_dim_L,scrolling_min_max_L,scrolling_min_max_R = getPrediction1(file_dir[0],slice_START, slice_END)
        print("Volume", Volume)
        print("Left_cranio_caudal_length",Maximum_Cranio_caudal_length_left)
        print("Right_cranio_caudal__length", Maximum_Cranio_caudal_length_right)
        print("Calculi_VOLUMES_Right", Calculi_VOLUMES_Right)
        print("Calculi_VOLUMES_Left", Calculi_VOLUMES_Left)
        print("Calculi_HU_Right", Calculi_HU_Right)
        print("Calculi_HU_Left", Calculi_HU_Left)
        print("Cal_Max_dim_R", Cal_Max_dim_R)
        print("Cal_Max_dim_L", Cal_Max_dim_L)
        print("scrolling_min_max_L", scrolling_min_max_L)
        print("scrolling_min_max_R", scrolling_min_max_R)
        # Calculi_VOLUMES_Right = [0.02157, 0.55414, 0.14022, 0.13887, 0.00674]
        # Calculi_VOLUMES_Left  = [0.23729, 0.22381, 0.01213]
        # Calculi_HU_Right = [225.5492306182221, 572.070318683767, 310.9268948068248, 425.1437435042339, 197.37156745046377]
        # Calculi_HU_Left = [253.06830924641167, 336.16608351730974, 177.50528266653419]
        # Cal_Max_dim_R = [8.245796751224226, 17.411158419243677, 23.933277205859213, 9.619722579159964, 1.928659265526184]
        # Cal_Max_dim_L = [11.55307893215051, 27.780054811186027, 2.424242794152434]
        flash((round(Volume[1]/1000, 2)) if (1 in Volume.keys()) else 0 )
        flash(round(Volume[2]/1000, 2) if (2 in Volume.keys()) else 0)
        flash(round(Volume[3]/1000, 2) if (3 in Volume.keys()) else 0)
        flash("Results/"+userid+"/mask.nii")   
        flash((round(Maximum_Cranio_caudal_length_right, 2))) 
        flash((round(Maximum_Cranio_caudal_length_left, 2)))  
        flash([round(item, 3) for item in Cal_Max_dim_R])
        flash([round(item, 3) for item in Cal_Max_dim_L])
        flash([round(item, 3) for item in Calculi_HU_Right])
        flash([round(item, 3) for item in Calculi_HU_Left])
        flash([round(item, 3) for item in Calculi_VOLUMES_Right]) 
        flash([round(item, 3) for item in Calculi_VOLUMES_Left])
        flash(scrolling_min_max_R)
        flash(scrolling_min_max_L)
        return redirect('/result')

#Add Post method to the decorator to allow for form submission. 
@app.route('/view', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        userid = session['userid']
        userid=str(userid)
        mydir=app.config['UPLOADED_IMAGE_COPY']+userid+'/'
        # files = glob.glob(app.config['UPLOADED_IMAGE_COPY']+userid+'/*')
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        mydir=app.config['RESULT_COPY']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        mydir=app.config['UPLOAD_GZ']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        mydir=app.config['RESULT_FILE']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        mydir=app.config['Dicom']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        print("Old Predicted file removed")
        mydir=app.config['GZ_EXTRACT']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        mydir=app.config['Zip_zip']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        
        mydir = app.config['RAR_EXTRACT']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        mydir = app.config['Zip_Extract']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        print("Removing folders inside these folders")
        mydir=app.config['UPLOADED_IMAGE']+userid+'/'
        if os.path.exists(mydir):
            shutil.rmtree(mydir)
        print("Old Original file removed")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        uploaded_files = request.files.getlist("file")
        print("uploaded_files", uploaded_files)
        print("Length_uploaded_files", len(uploaded_files))

        if file:
            filename = secure_filename(file.filename)
            
            print("Filename and extension", filename)
            filename_,extension = os.path.splitext(os.path.join(app.config['UPLOADED_IMAGE'])+userid+"/"+filename)
            print(os.path.join(app.config['UPLOADED_IMAGE'])+userid+"/"+filename)
            print("Filename and extension1", filename, extension)
            isExist =app.config['UPLOADED_IMAGE']+userid+"/"
            print("isExist", isExist)
            if not os.path.exists(isExist):
                    os.mkdir(isExist)
            #Add extension
            if((len(uploaded_files)<2) and filename.split('.')[1]=="nii" and extension!=".gz"):
                # print("Confition: (len(uploaded_files)<2) and filename.split('.')[1]==nii  and extension!=.gz")
                file.save(os.path.join(app.config['UPLOADED_IMAGE']+userid+"/","img.nii")) 
                             
            elif((len(uploaded_files)>3) and filename.split('.')[1]!="nii" and extension==".IMA"):
                # print("len(uploaded_files)>3) and filename.split('.')[1]!=nii and extension==.IMA")
                #print("if")
                #Save to some folder 
                folder = app.config['Dicom']+userid+"/"
                # isdicomExist=app.config['Dicom']+userid+"/"
                if not os.path.exists(folder):
                    os.mkdir(folder)
                for file in uploaded_files:
                    #print("file is:", file)
                    #print("filename is:", file.filename)                  
                    file.save(os.path.join(folder, file.filename))
                #Save nifti file path
                path_to_save_nifti_file=app.config['UPLOADED_IMAGE']+userid+"/"
                # if not os.path.exists(path_to_save_nifti_file):
                #         os.mkdir(path_to_save_nifti_file)
                    
                # # file.save(os.path.join(path_to_save_nifti_file, "/img.nii"))               
                # print("path_to_save_nifti_file", path_to_save_nifti_file)
                # print("folder", folder)
                #Convert to nifti and save 
                dicom2nifti.dicom_series_to_nifti(folder, path_to_save_nifti_file+"img.nii")
            elif((len(uploaded_files)<2) and (extension==".gz" or extension==".rar")):
                if(extension==".gz"):
                    print("(len(uploaded_files)<2) and (extension==.gz or extension==.rar)")
                    isgzexist=app.config['UPLOAD_GZ']+userid
                    print("gz file is:", os.path.join(app.config['UPLOAD_GZ'])+userid+"/"+filename)
                    if not os.path.exists(isgzexist):
                        os.mkdir(isgzexist)
                    file.save(os.path.join(app.config['UPLOAD_GZ']+userid+"/", file.filename))
                    print("file",file)
                    print("filename",file.filename)
                    iszexist=app.config['GZ_EXTRACT']+userid
                    if not os.path.exists(iszexist):
                        os.mkdir(iszexist)
                    patoolib.extract_archive((os.path.join(app.config['UPLOAD_GZ'])+userid+"/"+file.filename), outdir=app.config['GZ_EXTRACT']+userid+"/")
                    print("save the file")                   
                    origin = app.config['GZ_EXTRACT']+userid+'/'
                    target = app.config['UPLOADED_IMAGE']+userid+'/'

                    # Fetching the list of all the files
                    files = os.listdir(origin)
                    print("file_to_copy", files)
                    # Fetching all the files to directory
                    for file_name in files:
                        shutil.copy(origin+file_name, target+"img.nii")
                elif(extension==".rar"):
                    print("Confition: extension==.rar")
                    print("gz file is:", os.path.join(app.config['UPLOAD_GZ'])+userid+"/"+filename)
                    israrexist=app.config['UPLOAD_GZ']+userid
                    if not os.path.exists(israrexist):
                        os.mkdir(israrexist)
                    file.save(os.path.join(app.config['UPLOAD_GZ']+userid+"/", file.filename))
                    print("file",file)
                    print("filename",file.filename)
                    isexistrar=app.config['RAR_EXTRACT']+userid
                    if not os.path.exists(isexistrar):
                        os.mkdir(isexistrar)
                    patoolib.extract_archive((os.path.join(app.config['UPLOAD_GZ'])+userid+"/"+file.filename), outdir=isexistrar)
                    print("save the file")
                    folder = app.config['RAR_EXTRACT']+userid+"/"+((file.filename).split())[0]+"/"
                    print("Folder to extract: ", app.config['RAR_EXTRACT']+userid+"/"+((file.filename).split())[0]+"/")
                    #Save nifti file path
                    path_to_save_nifti_file=app.config['UPLOADED_IMAGE']+userid+"/img.nii"
                    #Convert to nifti and save 
                    dicom2nifti.dicom_series_to_nifti(folder, path_to_save_nifti_file)
            elif((len(uploaded_files)<2) and extension==".zip"):
                print("Confition: extension==.zip")
                iszipexist=app.config['Zip_zip']+userid
                if not os.path.exists(iszipexist):
                     os.mkdir(iszipexist)
                file.save(os.path.join(app.config['Zip_zip']+userid+"/", file.filename))
                file_name_zip_extract = os.path.join(app.config['Zip_zip'])+userid+"/"+file.filename
                print("file_name_zip_extract", file_name_zip_extract)
                folder = app.config['Zip_Extract']+userid
                if not os.path.exists(folder):
                     os.mkdir(folder)
                with zipfile.ZipFile(file_name_zip_extract, 'r') as zip_ref:
                    zip_ref.extractall(app.config['Zip_Extract']+userid)

                
                #Save nifti file path
                path_to_save_nifti_file=app.config['UPLOADED_IMAGE']+userid+"/img.nii"
                #Convert to nifti and save 
                dicom2nifti.dicom_series_to_nifti(folder, path_to_save_nifti_file)
            else:
                print("Confition: Else printed as formats doesnt match")
                pass

            origin = app.config['UPLOADED_IMAGE']+userid+'/'
            target =app.config['UPLOADED_IMAGE_COPY']+userid+'/'
            if not os.path.exists(target):
                os.mkdir(target)

            # Fetching the list of all the files
            files = os.listdir(origin)
            print("file_to_copy", files)
            # Fetching all the files to directory
            for file_name in files:
                shutil.copy(origin+file_name, target+file_name)
            print("Files are copied successfully")
            return redirect('/viewfile')


if __name__ == "__main__":
    #app.run(host='192.168.1.104')     #ip4 address
    app.run()