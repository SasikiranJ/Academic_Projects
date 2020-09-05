# coding=utf-8
import tensorflow as tf
import numpy as np
import keras
import os
import time


# SQLite for information
import sqlite3

# Keras
from keras.models import load_model, model_from_json
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
from keras import applications

# Flask utils
from flask import Flask, url_for, render_template, request,send_from_directory,redirect
from werkzeug.utils import secure_filename

os.environ['KERAS_BACKEND']='theano'

vgg16 = applications.VGG16(include_top= False, weights= "imagenet")
# Define a flask app
app = Flask(__name__)



model1 = load_model("models/bottleneck_fc_model.h5")
model2 = load_model("models/chest/bottleneck_chest_model.h5")
model3 = load_model("models/bottleneck_ecg_model.h5")




def info(field):

    if field == 1:
        conn = sqlite3.connect("models/diseases")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM oct")
        rows = cursor.fetchall()
        return rows
    elif field == 2:
        conn = sqlite3.connect("models/diseases")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chest")
        rows = cursor.fetchall()
        return rows

    elif field == 3:
        conn = sqlite3.connect("models/diseases")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM heart")
        rows  =  cursor.fetchall()
        return rows
def heart_predict(img_path):
    img1 = load_img(img_path, target_size=(224, 224))
    # convert to array
    img1 = img_to_array(img1)
    # normalize the array
    img1 = np.expand_dims(img1, axis=0)
    img1 /= 255
    
    img2 = load_img("uploads/0.jpg", target_size=(224,224))

    img2 = img_to_array(img2)

    img2 = np.expand_dims(img2,axis=0)

    img2 /= 255
    
    
    
    dist = np.linalg.norm(img1- img2)
    print(dist)

    
    if dist<50:
        return "ecg"
    else:
        return "not ecg"

def retina_predict(img_path):
    img1 = load_img(img_path, target_size=(224, 224))
    # convert to array
    img1 = img_to_array(img1)
    # normalize the array
    img1 = np.expand_dims(img1, axis=0)
    img1 /= 255
    
    img2 = load_img("uploads/cnv.jpg", target_size=(224,224))

    img2 = img_to_array(img2)

    img2 = np.expand_dims(img2,axis=0)

    img2 /= 255
    
    
    
    dist = np.linalg.norm(img1- img2)
    

    
    if dist<150:
        return "retina"
    else:
        return "not retina"

def lungs_predict(img_path):
    # load image with target size
    img1 = load_img(img_path, target_size=(224, 224))
    # convert to array
    img1 = img_to_array(img1)
    # normalize the array
    img1 = np.expand_dims(img1, axis=0)
    img1 /= 255
    
    img2 = load_img("uploads/person1949_bacteria_4880.jpeg", target_size=(224,224))

    img2 = img_to_array(img2)

    img2 = np.expand_dims(img2,axis=0)

    img2 /= 255
    
    
    
    dist = np.linalg.norm(img1- img2)

    
    if dist<100:
        return "lungs"
    else:
        return "not lungs"
    


def model_predict(img_path,field):
    # load image with target size
    image = load_img(img_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.
    bt_prediction = vgg16.predict(image)

    if field == 1:

        preds = model1.predict_classes(bt_prediction)
        return int(preds)
    elif field == 2:
        preds = model2.predict_classes(bt_prediction)
        
        return int(preds)
    elif field == 3:
        preds = model3.predict_classes(bt_prediction)

        return int(preds)

    


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('web.html')

#To predict Retinal disease
@app.route('/predict1', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)

        retina = retina_predict(img_path)
        if retina == "retina":
        
            preds = model_predict(img_path,1)
        # Make prediction
        
            rows = info(1)
            res = np.asarray(rows[preds])
            value = (preds == int(res[0]))
            if value:
                Sno, Name = [i for i in res]
                if Name != "NORMAL":
                    return render_template('result1.html', Sno = Sno,msg="Your retina affected with ", result = Name, filee=f.filename)
                else:
                    return render_template('result1.html', Sno = Sno,msg="Your retina is ", result = Name, filee=f.filename)

        else:
            return render_template('result1.html', Error="ERROR: Please try again. Uploaded image not a OCT image",filee = f.filename)
        # return result
    return None

#To predict pneumonia
@app.route('/predict2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)
        
        lungs = lungs_predict(img_path)
        if lungs == "lungs":

            preds = model_predict(img_path,2)
            # Make prediction
        
            rows = info(2)
            res = np.asarray(rows[preds])
            value = (preds == int(res[0]))
            if value:
                Sno, Name = [i for i in res]
                if Name == "PNEUMONIA":
                    return render_template('result2.html', Sno = Sno,msg="Your Lungs are effected with ", result = Name, filee=f.filename)
                else:
                    return render_template('result2.html', Sno = Sno,msg="Your Lungs are ", result = Name, filee=f.filename)
        else:
            return render_template('result2.html', Error="ERROR: Please try again or Try to upload correct chest Xray image", filee=f.filename)
            # return result
    return None

#To predict Heart disease
@app.route('/predict3', methods=['GET', 'POST'])
def upload3():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)
        
        heart = heart_predict(img_path)
        if heart == "ecg":

            preds = model_predict(img_path,3)
            # Make prediction
        
            rows = info(3)
            res = np.asarray(rows[preds])
            value = (preds == int(res[0]))
            if value:
                Sno, Name = [i for i in res]
                if Name == "Normal":
                    return render_template('result2.html', Sno = Sno,msg="Your heart beat is ", result = Name, filee=f.filename)
                elif Name == "Unknown-Beats":
                    return render_template('result2.html', Sno = Sno,msg="Unknown Beats ", result = Name, filee=f.filename)
                else:
                    return render_template('result2.html', Sno = Sno,msg="Your heart beat is affected with ", result = Name, filee=f.filename)
        else:
            return render_template('result2.html', Error="ERROR: Please try again or Try to upload correct ecg image", filee=f.filename)
            # return result
    return None



@app.route('/predict1/<filename>')
def send_file1(filename):
    return send_from_directory('uploads', filename)

@app.route('/predict2/<filename>')
def send_file2(filename):
    return send_from_directory('uploads', filename)

@app.route('/predict3/<filename>')
def send_file3(filename):
    return send_from_directory('uploads', filename)





if __name__ == '__main__':
    app.run(threaded = True)
    
    
