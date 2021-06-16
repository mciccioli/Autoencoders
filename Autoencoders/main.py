import json

from Autoencoder import Autoencoder

from font import font1, font2, font3
import numpy as np


def hex_to_bin_array(hex_array):
    to_ret = []
    for a in hex_array:
        matrix = list(map(lambda x: list(bin(x)[2:].zfill(5)), a))
        flattened = list(map(lambda b: -1 if int(b) == 0 else 1, np.array(matrix).flatten()))
        to_ret.append(flattened)
    return to_ret

with open("config.json") as file:
    config = json.load(file)

exercise = config["EJ"] 
font_number = config["FONT"] 
epochs = config["EPOCHS"]
eta = config["ETA"]
beta = config["BETA"]
division_layer = config["DIVISION_LAYER"]
momentum = config["MOMENTUM"]
momentum = False
momentum_factor = config["MOMENTUM_FACTOR"]  
adaptive_learning = config["ADAPTIVE_LEARNING"]
adaptive_learning = False
adaptive_learning_epochs = config["ADAPTIVE_LEARNING_EPOCHS"]
adaptive_learning_a = config["ADAPTIVE_LEARNING_A"]
adaptive_learning_b = config["ADAPTIVE_LEARNING_B"]            



if font_number == 1:
    font_bin = hex_to_bin_array(font1)
elif font_number == 2:
    font_bin = hex_to_bin_array(font2)
else:
    font_bin = hex_to_bin_array(font3)





if exercise == '1a':

    autoencoder = Autoencoder(font_bin, eta, beta, division_layer, momentum , momentum_factor, adaptive_learning, adaptive_learning_epochs, adaptive_learning_a, adaptive_learning_b)
    autoencoder.initialize_network()
    autoencoder.initialize_weights()
  
    autoencoder.interval_train(3, 2000)