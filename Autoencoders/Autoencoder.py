
import math
import random
import copy
from typing import Counter
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:

    def __init__(self, data, eta, beta, division_layer, momentum, momentum_factor, adaptive_learning, adaptive_learning_epochs, adaptive_learning_a, adaptive_learning_b):
        self.data =data
        self.eta = eta
        self.division_layer = division_layer
        self.momentum = momentum
        self.momentum_factor = momentum_factor
        self.adaptive_learning = adaptive_learning
        self.adaptive_learning_epochs = adaptive_learning_epochs
        self.adaptive_learning_a = adaptive_learning_a
        self.adaptive_learning_b = adaptive_learning_b
        self.beta = 0.5
        self.input_size = len(data[0])
        # Pesos [capa destino, neurona destina, nuerona origen] 
        self.weights = []
        # Valor de las neuronas [capa, índice]
        self.activations = []
        # Error
        self.d = []
        # cantidad de capas que va a tener mi red
        self.totalLayers = 0
        # cantidad de neuronas que voy a tener por capa
        self.nodesPerLayer = []


    def initialize_network(self):
        initial_size = self.input_size
        neuron_counter = initial_size
        # obtengo la cantidad total de capas de mi red
        while neuron_counter > 2:
            # recordar que son dos redes en una (por eso el +2)
            self.totalLayers += 2
            # uso math.floor para obtener un número entero --> Devuelve el máximo entero menor o igual a un número
            neuron_counter = math.floor(neuron_counter / self.division_layer)
        self.totalLayers += 1
        # inicializo el arreglo 
        self.nodesPerLayer= [0 for i in range(self.totalLayers)]
        neuron_counter = initial_size
        for i in range(math.floor(self.totalLayers/2)):
            # sumo +1 a todo por el bias
            # como la segunda red estaq invertida, voy de atrás para adelante
            self.nodesPerLayer[i] = neuron_counter + 1                            
            self.nodesPerLayer[self.totalLayers - 1 - i] = neuron_counter + 1   
            neuron_counter = math.floor(neuron_counter / self.division_layer)
        # espacio latente (Z), tengo 2 capas y agrego una más por el bias
        self.nodesPerLayer[math.floor(self.totalLayers/2)] = 3
        # La capa de salida no tiene bias --> le resto el bias que le agregue antes               
        self.nodesPerLayer[-1] -= 1  
        # imprimo la cantidad de neuronas por capa                                         
        for i in range(self.totalLayers):
            if i != self.totalLayers - 1: # resto 1 al imprimir porque el bias no se imprime
                neurons = self.nodesPerLayer[i] - 1
            else: # la última capa no tiene bias
                neurons = self.nodesPerLayer[i]
            print("Capa", i, ": ", neurons, "Neuronas")
            # inicializo el valor de activación de cada neurona
            self.activations.append([0.0 for j in range(self.nodesPerLayer[i])])
        if self.momentum:
            self.initialize_momentum()


    def initialize_weights(self):
        self.weights = []
        self.weights.append(np.random.rand(0,0))
        for i in range(0, self.totalLayers - 1):
            # le agrego el bias a cada capa
            self.activations[i][0] = 1 
            self.weights.append(np.random.rand(self.nodesPerLayer[i+1], self.nodesPerLayer[i]) - 0.5)               
       
    def initialize_momentum(self):
        self.momentum_values = []
        self.momentum_values.append(np.zeros((0,0)))
        for i in range(self.totalLayers - 1):
            self.momentum_values.append(np.zeros((self.nodesPerLayer[i + 1], self.nodesPerLayer[i])))


    def g(self, x):
        return np.tanh(self.beta * x)

    def g_derivative(self, x):
        cosh2 = (np.cosh(self.beta*x)) ** 2
        return self.beta / cosh2


    # hmi = sumatoriaj (Wij * Vmj)
    def h(self, m, i, nodes_in_the_layer, weights, activations):
        hmi = 0
        for j in range(0, nodes_in_the_layer):
            hmi += weights[m][i][j] * activations[m-1][j]
        return hmi

    
    def interval_train(self, step, epochs):
        error_epochs = 0
        np.random.shuffle(self.data)
        # si tengo un salto mayor a la longitud de la data, hago que el salto sea igual a la longitud --> es como hacer directo train
        if(step >= len(self.data)):
            step = len(self.data)
       # range(start, stop, step)
        for width in range(step-1, len(self.data) + step - 1, step):
            if width > len(self.data):
                width = len(self.data)
            data = self.data[0:width+1]
            print("Entrenando", len(data), " letras")
            error, lowest_error, learned = self.train(data, epochs)
            error_epochs += len(error)
            print( "\tAprendí ", learned, "/", len(data), " letras")
        print("El error total fue de ", error_epochs)
        return error, lowest_error, learned
           
            

    def train(self, data, epochs):
        if self.momentum:
            self.initialize_momentum()
        learning_rate = self.eta
        error_over_time = []

        for i in range(self.totalLayers):
            # inicializo el error en cada capa
            self.d.append(np.zeros(self.nodesPerLayer[i]))
        for i in range(1, self.totalLayers - 1):
            # le agrego el bias a cada capa
            self.activations[i][0] = 1                 
        epoch = 0
        last_error = 0
        growing_eta = 0
        descend_eta = 0
        lowest_error = 100000
        current_error = 100000
        while current_error != 0  and epoch < epochs:
            epoch += 1
            # mezclo la data de entrenamiento para que al iterar sea azaroso
            np.random.shuffle(data)
            for mu in range(len(data)):
                
                # Paso 2 (V0 tiene los ejemplos iniciales)
                self.activations[0][0] = 1.0  # bias
                for k in range(len(data[0])):
                    self.activations[0][k+1] = data[mu][k]

                # PASO 3: Propagar la entrada hasta a capa de salida
                self.propagate_input()
                
                # PASO 4: Calcular δ para la capa de salida
                #δMi = g_derivada(hMi)(ζµi − VMi)
                for i in range(0, self.nodesPerLayer[self.totalLayers - 1]):
                    hMi = self.h(self.totalLayers - 1, i, self.nodesPerLayer[self.totalLayers - 2], self.weights, self.activations)
                    self.d[self.totalLayers - 1][i] = self.g_derivative(hMi)*(data[mu][i] - self.activations[self.totalLayers - 1][i])
                
                # PASO 5: Retropropagar δM
                self.retropropagate_error()

                # PASO 6: Actualizar los pesos de las conexiones
                self.update_weights()

            # Medir error con pesos actuales TODO: REVISAR
            learned_letters = 0
            for mu in data:
                for k in range(len(mu)):
                    self.activations[0][k+1] = mu[k]
                for m in range(1, self.totalLayers - 1):
                    for i in range(1, self.nodesPerLayer[m]):
                        hmi = self.h(m, i, self.nodesPerLayer[m-1], self.weights, self.activations)
                        self.activations[m][i] = self.g(hmi)
                for i in range(0, self.nodesPerLayer[self.totalLayers - 1]):
                    hMi = self.h(self.totalLayers - 1, i, self.nodesPerLayer[self.totalLayers - 2], self.weights, self.activations)
                    self.activations[self.totalLayers - 1][i] = self.g(hMi)
                perceptron_output = self.activations[self.totalLayers - 1]
                wrong_pixels = 0
                for bit in range(len(perceptron_output)):
                    if(perceptron_output[bit] * mu[bit] < 0):
                        wrong_pixels += 1
                if(wrong_pixels == 0):
                    learned_letters += 1


            current_error = len(data)-learned_letters
            if (self.adaptive_learning):
                if(current_error - last_error) <= 0:
                    growing_eta +=1
                    descend_eta = 0
                else:
                    descend_eta += 1
                    growing_eta = 0
                if growing_eta >= self.adaptive_learning_epochs:
                    self.eta += self.adaptive_learning_a 
                    growing_eta = 0
                elif descend_eta >= self.adaptive_learning_epochs:
                    self.eta -= self.adaptive_learning_b * self.eta
                    descend_eta = 0
            #print(learned_letters)
            #print(current_error)
            error_over_time.append(current_error)
            last_error = current_error
            if(current_error < lowest_error):
                lowest_error = current_error
                
        return error_over_time, lowest_error, learned_letters

    def propagate_input(self):
        # Vmi = g(hmi) para todo m desde 1 hasta M
        for m in range(1, self.totalLayers - 1):
            for i in range(1, self.nodesPerLayer[m]):
                hmi = self.h(m, i, self.nodesPerLayer[m-1], self.weights, self.activations)
                self.activations[m][i] = self.g(hmi)
        # en la última capa no tengo el bias --> empiezo de cero
        for i in range(self.nodesPerLayer[self.totalLayers - 1]):
            hmi = self.h(self.totalLayers - 1, i, self.nodesPerLayer[self.totalLayers - 2], self.weights, self.activations)
            self.activations[self.totalLayers - 1][i] = self.g(hmi)

    def retropropagate_error(self):
        # δm−1i = g_derivada(hm−1i) sumatoriaj(w(mji) * δmj) para todo m entre M y 2
        # range(start, stop, step) --> empiezo de la última capa y voy bajando hasta la primera
        for m in range(self.totalLayers - 1, 1, -1):                                           
            for i in range(0, self.nodesPerLayer[m-1]):
                # hm−1i
                hprevmi = self.h(m-1, i, self.nodesPerLayer[m-2], self.weights, self.activations)
                error_sum = 0
                # sumatoriaj(w(mji) * δmj)
                for j in range(0, self.nodesPerLayer[m]):
                    error_sum += self.weights[m][j][i] * self.d[m][j]
                # δm−1i = g_derivada(hm−1i) sumatoriaj(w(mji) * δmj)
                self.d[m-1][i] = self.g_derivative(hprevmi) * error_sum

    def update_weights(self):
        # wmij (nuevo) = wmij (viejo) + ∆wmij donde ∆wmij = η * δmi * Vm−1j
        for m in range(1, self.totalLayers):
            for i in range(self.nodesPerLayer[m]):
                for j in range(self.nodesPerLayer[m-1]):
                    # ∆wmij donde ∆wmij = η * δmi * Vm−1j
                    delta = self.eta * self.d[m][i] * self.activations[m-1][j]
                    if self.momentum:
                        self.momentum_values[m][i][j] = self.momentum_factor * self.momentum_values[m][i][j] + (1 - self.momentum_factor) * delta
                        self.weights[m][i][j] += self.momentum_values[m][i][j]
                    else:
                        # wmij (nuevo) = wmij (viejo) + ∆wmij
                        self.weights[m][i][j] += delta

    def test(self, test_data):
        for input in test_data:
            print(" Expected Output / Perceptron Output ")
            for k in range(len(input)):
                self.activations[0][k+1] = input[k] 
            for m in range(1, self.totalLayers - 1):
                for i in range(1, self.nodesPerLayer[m]):
                    hmi = self.h(m, i, self.nodesPerLayer[m-1], self.weights, self.activations)
                    self.activations[m][i] = self.g(hmi)
            for i in range(0, self.nodesPerLayer[self.totalLayers - 1]):
                hMi = self.h(self.totalLayers - 1, i, self.nodesPerLayer[self.totalLayers - 2], self.weights, self.activations)
                self.activations[self.totalLayers - 1][i] = self.g(hMi)
            perceptron_output = self.activations[self.totalLayers - 1]
            for bit in range(len(perceptron_output)):
                if(perceptron_output[bit] > 0): 
                    perceptron_output[bit] = 1
                else: 
                    perceptron_output[bit] = -1
            #Print the original letter
            for j in range(7):
                for i in range(5):
                    if(self.activations[0][1 + i + j * 5] > 0): 
                        # end especifica que imprimir al final. Por default imprime un salto de línea
                        # como no quiero un salto de línea (estoy imprimiendo por bits), especifico que imprima un espacio
                        print("*", end = "")
                    else: 
                        print(" ", end = "")
                print("\t\t", end="")
                for i in range(5):
                    if(perceptron_output[i + j * 5] > 0): 
                        print("*", end = "")
                    else: 
                        print(" ", end = "")
                print("")
    
    def get_activations(self):
        return self.activations

    def graph(self, data, symbols):
        # usa la función cla() para limpias los axis
        plt.cla()
        # el contador me ayuda a guardar el valor del espacio latente para cada símbolo
        counter = 0
        # el espacio latente es de 2 dimensiones
        latent_values = [[None, None] for i in range(len(data))]
        for input in data:
            for k in range(len(input)):
                self.activations[0][k+1] = input[k]
            for m in range(1, self.totalLayers - 1):
                for i in range(1, self.nodesPerLayer[m]):
                    hmi = self.h(m, i, self.nodesPerLayer[m-1], self.weights, self.activations)
                    self.activations[m][i] = self.g(hmi)
            for i in range(0, self.nodesPerLayer[self.totalLayers - 1]):
                hMi = self.h(self.totalLayers - 1, i, self.nodesPerLayer[self.totalLayers - 2], self.weights, self.activations)
                self.activations[self.totalLayers - 1][i] = self.g(hMi)
            # salida del perceptrón 
            perceptron_output = self.activations[self.totalLayers - 1]
            # valor en el espacio latente (capa intermedia)
            x = self.activations[math.floor(self.totalLayers/2)][1]
            y = self.activations[math.floor(self.totalLayers/2)][2]
            # arreglo donde guardo los valores del espacio latente 
            latent_values[counter] = [x, y]
            # annotate() me sirve para anotar los valores de x e y con su respectivo texto (los símbolos en este caso)
            if len(symbols) != 3 and len(symbols) != 4 and len(symbols) != 5:
                plt.scatter(x, y)
                plt.annotate(symbols[counter], xy=(x,y), textcoords='data')
            else:
                plt.scatter(x, y, c=symbols[counter])
            counter += 1
        plt.grid()
        plt.show() 

    def decode(self, a, b):
        self.activations[math.floor(self.totalLayers/2)][1] = a
        self.activations[math.floor(self.totalLayers/2)][2] = b
        for m in range(math.floor(self.totalLayers/2)+1, self.totalLayers - 1):
            for i in range(1, self.nodesPerLayer[m]):
                hmi = self.h(m, i, self.nodesPerLayer[m-1], self.weights, self.activations)
                self.activations[m][i] = self.g(hmi)
        for i in range(0, self.nodesPerLayer[self.totalLayers - 1]):
            hMi = self.h(self.totalLayers - 1, i, self.nodesPerLayer[self.totalLayers - 2], self.weights, self.activations)
            self.activations[self.totalLayers - 1][i] = self.g(hMi)
        perceptron_output = self.activations[self.totalLayers - 1]
        for bit in range(len(perceptron_output)):
            if(perceptron_output[bit] > 0): 
                perceptron_output[bit] = 1
            else: 
                perceptron_output[bit] = -1
        # imrpimo la letra
        print("Decoding complete:")
        for j in range(7):
            for i in range(5):
                if(perceptron_output[i + j * 5] > 0): 
                    print("*", end = "")
                else: 
                    print(" ", end = "")
            print("")
        print("\n")

class Autoencoder2:
    def __init__(self, weights ):
        self.weights = weights

    def get_weights(self):
        return self.weights

class AutoencoderFactory:
    def train(layers, nodes_per_layer, data, eta, epochs, beta):
        g = lambda x : x if x > 0 else 0
        gder = lambda x : 1 if x > 0 else 0

        error = []
        division_layer = int(layers/2)
        weights = [ np.random.rand(0,0) if i == 0 else np.random.rand(nodes_per_layer[i], nodes_per_layer[i-1]) - 0.5 for i in range(layers)]
        
        current_error = np.iinfo(np.int32).max

        for epoch in range(epochs):
            activations = [[ 1.0 if j == 0 else 0.0 for j in range(nodes_per_layer[i]+1)] for i in layers]

            if current_error == 0:
                break

            np.random.shuffle(data)
            for i in range(len(data)):
                V = [data[0]]
                h = []
                for j in range(layers):
                    V.append(g(weights[j]*V[len(V)-1]))
                    h.append(weights[j]*V[len(V)-1])
                delta = gder(h[layers-1])*(data[i]-g(h[layers-1]))
                for j in range(1,layers):
                    delta = gder(h[layers-j-1])*(weights[j]*V[layers-j-1])
                    weights[j] = weights[j] + eta*delta*V[layers-j-1]
        return Autencoder2( weigths )