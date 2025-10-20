import random
from typing import List, Union
from sklearn.base import BaseEstimator
#import dice_ml
#from dice_ml import Dice
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class User:
    def __init__(self, believe_level, rethink_level, fairness):
        self.believe_level = believe_level
        self.rethink_level = rethink_level
        self.fairness = fairness
        
    def believe(self):
        
        if self.believe_level > 1:
            return None
        else:
            return random.choices(population=[True, False], 
                              weights=[self.believe_level, 1-self.believe_level], k=1)[0]
    
    def rethink(self):
        return random.choices(population=[True, False], 
                              weights=[self.rethink_level, 1-self.rethink_level], k=1)[0]
    
    def fairness_percentage(self):
        return self.fairness
    
class GroundTruther(User):
    def __init__(self, believe_level, rethink_level, fairness, expertise):
        super().__init__(believe_level, rethink_level, fairness)
        self.expertise = expertise
        
    def predict(self, record, ground):

        return random.choices(population=[ground, not ground], 
                              weights=[self.expertise, 1-self.expertise], k=1)[0]

class ModelBased(User):
    def __init__(self, believe_level, rethink_level, fairness, model, X, Y):
        super().__init__(believe_level, rethink_level, fairness)
        self.model = model.fit(X, Y)
        
    def predict(self, record, ground):
        return self.model.predict([record])[0]
    
class Real():
        
    def predict(self):
        return input()
   
    def believe(self):
        return input()
    
    def rethink(self):
        return input()


class RiverModelWrapper(BaseEstimator):
    def __init__(self, river_model, target_column, feature_names=None):
        self.river_model = river_model
        self.target_column = target_column
        self.feature_names = feature_names

    def predict_one(self, x):
        return self.river_model.predict_one(x)
    
    def predict_proba_one(self, x):
        return self.river_model.predict_proba_one(x)
    
    def learn_one(self, x, y):
        self.river_model.learn_one(x, y)

    def fit(self, X, Y):
        if isinstance(X, np.ndarray):
            for x, y in zip(X, Y):
                x = {name: value for name, value in zip(self.feature_names, x)}
                self.river_model.learn_one(x, y)
                
        elif isinstance(X, pd.DataFrame):
            X = X.to_dict(orient='records')
            for x, y in zip(X, Y):
                self.river_model.learn_one(x, y)
        else:
            self.river_model.learn_one(X, Y)
        
        return self
   
    def predict(self, X):
       
        if isinstance(X, pd.DataFrame):
            X = X.to_dict(orient='records')
            return [self.river_model.predict_one(x) for x in X]
        
        elif isinstance(X, np.ndarray):
            X = X.squeeze()
           
            
            if X.ndim == 1:
                x_d = [{name: value for name, value in zip(self.feature_names, X)}]
                
                return np.array([self.river_model.predict_one(x) for x in x_d]).reshape(-1, 1).flatten()
            else:
                preds = []
                for row in X:
                    x_dict = {name: value for name, value in zip(self.feature_names, row)}
                    preds.append(self.river_model.predict_one(x_dict))
                
                return np.array(preds).flatten() 
        
        else:
            return self.river_model.predict_one(X)
    
    def predict_proba(self, X):
      
        probas = []
        
        if isinstance(X, pd.DataFrame): 
            X = X.to_dict(orient='records')
            for x in X:
                
                try:
                   
                    prediction_proba = self.river_model.predict_proba_one(x)
                    probas.append([prediction_proba.get(False, 0), prediction_proba.get(True, 0)])
                except:
                    probas.append([0, 0])
                #print(probas)
                
        elif isinstance(X, np.ndarray):
            
            
            if X.ndim == 1:
                x_d = [{name: value for name, value in zip(self.feature_names, X)}]
                
            else:
                x_d = []
                for row in X:
                    x_d.append({name: value for name, value in zip(self.feature_names, row)})
            
            for x in x_d:
                prediction_proba = self.river_model.predict_proba_one(x)
                try:
                    probas.append([prediction_proba.get(False, 0), prediction_proba.get(True, 0)])
                except:
                    probas.append([0, 0])
        else:
            raise TypeError("Formato di input non supportato: utilizzare DataFrame o array NumPy.")
       
        return np.array(probas)
    

class AE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: tf.Tensor, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat




from tensorflow.keras import layers, Model

class CustomEncoder(Model):
    def __init__(self, hidden_dim=64, latent_dim=10):
        super(CustomEncoder, self).__init__()
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(hidden_dim // 2, activation="relu")
        self.latent_layer = layers.Dense(latent_dim, activation="sigmoid")  # Sigmoid per valori normalizzati

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.latent_layer(x)
    
class CustomDecoder(Model):
    def __init__(self, hidden_dim=64, output_dim=None):
        super(CustomDecoder, self).__init__()
        self.dense1 = layers.Dense(hidden_dim // 2, activation="relu")
        self.dense2 = layers.Dense(hidden_dim, activation="relu")
        self.output_layer = layers.Dense(output_dim, activation="linear")  # "sigmoid" se i dati sono normalizzati

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)    
    

































































































            

                
        
   

class HeAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: tf.Tensor, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class DecoderList(tf.keras.Model):
    def __init__(self, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    def call(self, input: Union[tf.Tensor, List[tf.Tensor]], **kwargs):
        return [self.decoder(input, **kwargs)]
    
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()    