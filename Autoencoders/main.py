import json

from Autoencoder import Autoencoder, Autoencoder2

from font import font1, font2, font3
import numpy as np


def hex_to_bin_array(hex_array):
    to_ret = []
    for a in hex_array:
        matrix = list(map(lambda x: list(bin(x)[2:].zfill(5)), a))
        print(a)
        print(matrix)
        flattened = list(map(lambda b: -1 if int(b) == 0 else 1, np.array(matrix).flatten()))
        to_ret.append(flattened)
    return to_ret

def hex_to_bin_array2(hex_array):
    to_ret = []
    for a in hex_array:
        matrix = list(map(lambda x: list(bin(x)[2:].zfill(8)), a))
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

colors = [
        [0xFF, 0x00, 0x00], 
        [0x00, 0xFF, 0x00],
        [0x00, 0x00, 0xFF], 
        [0x73, 0x3B, 0xCE],
        [0x46, 0x46, 0x4E]
    ]

print(colors)

if font_number == 1:
    font_bin = hex_to_bin_array(font1)
elif font_number == 2:
    font_bin = hex_to_bin_array(font2)
elif font_number == 3:
    font_bin = hex_to_bin_array(font3)
else:
    bin = hex_to_bin_array2(colors)

symbols2 = [
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']

if exercise == '1a':

    autoencoder = Autoencoder(font_bin, eta, beta, division_layer, momentum , momentum_factor, adaptive_learning, adaptive_learning_epochs, adaptive_learning_a, adaptive_learning_b)
    autoencoder.initialize_network()
    autoencoder.initialize_weights()
  
    autoencoder.interval_train(3, epochs)
    autoencoder.graph(font_bin,symbols2 )

if exercise == '2':
    autoencoder = Autoencoder(bin, eta, beta, division_layer, momentum, momentum_factor, adaptive_learning, adaptive_learning_epochs, adaptive_learning_a, adaptive_learning_b)
    autoencoder.initialize_network()
    autoencoder.initialize_weights()

    learnt = autoencoder.interval_train(3, epochs)
    print(f"Activations: {autoencoder.get_activations()}")
    autoencoder.graph(bin, ['RED', 'GREEN', 'BLUE', 'VIOLET', 'GRAY'])
    ans = autoencoder.decode(1, 0, exercise)
    ans1 = autoencoder.decode(0, 1, exercise)