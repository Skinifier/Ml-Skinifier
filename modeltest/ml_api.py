import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing import image
from flask import Flask, jsonify, request
import cv2
import os

class_names = [
    'Jerawat',
    'Karsinoma',
    'Dermatitis Atopik',
    'Selulitis',
    'Eksim',
    'Eksantema',
    'Herpes',
    'Pigmentasi',
    'Lupus',
    'Melanoma',
    'Dermatitis',
    'Psoriasis',
    'Keratosis',
    'Penyakit Sistemik',
    'Infeksi Jamur',
    'Biduran',
    'Tumor Vaskular',
    'Vaskulitis',
    'Kutil'
]

app = Flask(__name__)

model_path = 'https://storage.googleapis.com/mlmodelbucket/cnn_model.h5'
cnn_model = tf.keras.models.load_model(model_path)


@app.route('/predict', methods=['POST'])
def predictions():
    file = request.files['imagefile']
    file_path = "./images/" + file.filename
    file.save(file_path)
    
    try:
        images= []
        img_size = (192, 192, 3)
        
        img_path = file_path
        
        
        img = image.load_img(img_path, target_size=img_size)
        img = image.img_to_array(img)
        img_array = np.asarray(cv2.resize(cv2.imread(img_path, cv2.IMREAD_COLOR), img_size[0:2])[:, :, ::-1])
        images.append(img_array)
        images = np.asarray(images)
        
        predictions = cnn_model.predict(images, verbose=0)[0]
        
        result = class_names[np.argmax(predictions)]
        
        os.remove(file_path)

        return jsonify({'status':200,'message':'Prediction Succes','prediction':{'prediction':str(result)}})
    except Exception as e:
        print(e)
        return jsonify({'Error': str(e)})

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World'})


if __name__ == '__main__':
    app.run(debug=False)
