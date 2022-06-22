#Import all the required libraries

import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
import keras
from keras.preprocessing.image import load_img

from tensorflow.keras.models import Model

from keras.models import load_model
from PIL import Image

from flask import Flask, render_template, request

############################################################################################################

# Setting  parameters

embedding_dim = 256 
units = 512

#top 5,000 words +1
vocab_size = 5001

max_length = 31

############################################################################################################

# Encoder using CNN Keras subclassing method

class Encoder(Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim) #build your Dense layer with relu activation
        
    def call(self, features):
        features =  self.dense(features) # extract the features from the image shape: (batch, 8*8, embed_dim)
        features =  tf.keras.activations.relu(features, alpha=0.01, max_value=None, threshold=0)
        return features

############################################################################################################

class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) 
        self.W2 = tf.keras.layers.Dense(units) 
        self.V = tf.keras.layers.Dense(1) 
        self.units=units

    def call(self, features, hidden):
        hidden_with_time_axis = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))  
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1) 
        context_vector = attention_weights * features 
        context_vector = tf.reduce_sum(context_vector, axis=1)  
        return context_vector, attention_weights


############################################################################################################

class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units) #iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) #build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #build your Dense layer
        

    def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden) #create your context vector & attention weights from attention model
        embed = self.embed(x) # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis = -1) # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output,state = self.gru(embed) # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)
        
        return output, state, attention_weights
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

############################################################################################################

#decoder object
decoder=Decoder(embedding_dim, units, vocab_size)

############################################################################################################
#loading saved models

saved_decoder_model=keras.models.load_model(
    "decoder", custom_objects={"CustomModel": Decoder}
)

saved_encoder_model=keras.models.load_model(
    "encoder", custom_objects={"CustomModel": Encoder}
)

extract_features=load_model('feature_extractor.h5')

tok= open('tokenizer.pkl', 'rb')
tokenizer=pickle.load(tok)


############################################################################################################

# function for loading image for the model
IMAGE_SHAPE = (299, 299)
def load_images(image_path) :
  img = tf.io.read_file(image_path, name = None)
  img = tf.image.decode_jpeg(img, channels=0)
  img = tf.image.resize(img, IMAGE_SHAPE)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

#function for filtering the predicted text
def filt_text(text):
    filt=['<start>','<unk>','<end>'] 
    temp= text.split()
    [temp.remove(j) for k in filt for j in temp if k==j]
    text=' '.join(temp)
    return text

# predicting caption
def evaluate(image):
    hidden = decoder.init_state(batch_size=1)

    temp_input = tf.expand_dims(load_images(image)[0], 0) 
    img_tensor_val = extract_features(temp_input) 
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = saved_encoder_model (img_tensor_val) 

    dec_input = tf.expand_dims([2], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, _ = saved_decoder_model(dec_input, features, hidden) 
        

        predicted_id = tf.argmax(predictions[0]).numpy() 
        result.append (tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result,predictions

        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result,predictions




app = Flask(__name__)
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/showcaption', methods=['GET', 'POST'])
def after():

    img = request.files['imagefile']
    img.save('static/file.jpg')
    test_path=r'static/file.jpg'
    result, _ = evaluate(test_path)   
    pred_caption=' '.join(result).rsplit(' ', 1)[0]
    pred_caption=filt_text(pred_caption)
    print(pred_caption)

    return render_template('index.html', data=pred_caption)

if __name__ == "__main__":
    app.run(debug=True)