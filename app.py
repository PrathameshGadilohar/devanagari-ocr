import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from PIL import Image
import base64
import io
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('devnagari_model.h5')
labelencoder = pickle.load(open('labelencoder.pkl', 'rb'))


def predict_character(image):
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    logit_y = model(image)
    pred_y = labelencoder.inverse_transform(np.argmax(logit_y, axis=-1))
    return pred_y[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert_image', methods=['POST'])
def convert_image():
    image_data = request.json['imageData']
    image_binary = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_binary)).convert('L')
    image_resized = image.resize((32, 32))
    pixel_values = np.array(image_resized.getdata()).reshape((32, 32))
    prediction = predict_character(pixel_values)
    return jsonify({'prediction_text': 'Character : {}'.format(prediction)})


if (__name__ == "__main__"):
    app.run(debug=True)
