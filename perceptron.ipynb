import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import numpy as np
%matplotlib inline

def random_coord(n):
    coord=[]
    for _ in range(n):
        coord.append([int(random.uniform(1,6)),int(random.uniform(1,6))])
    return coord

def plot_decision_boundary(training_data, model):
    x_min, x_max = x1.min() - 1, x1.max() + 1
    weights = -model.weights[0] / model.weights[1]
    line_points = np.linspace(x_min, x_max)
    plt.plot(line_points, weights * line_points -
             (1 / model.weights[1]))

def plot_hyperplane(training_data, labels, weights, bias):
    x_min, x_max = training_data.min() - 1, training_data.max() + 1
    slope = - weights[0]/weights[1]
    intercept = - bias/weights[1]
    x_hyperplane = np.linspace(x_min, x_max)
    y_hyperplane = slope * x_hyperplane + intercept
    fig = plt.figure(figsize=(8,6))
    plt.scatter(training_data[:,0], training_data[:,1], c=labels)
    plt.plot(x_hyperplane, y_hyperplane, '-')
    plt.title("Clasificación")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        #Inicializamos la clase con pesos aleatorio
        self.weights =  np.random.rand(input_size+1)
        #learning rate default a 0.01
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        #se calcula el producto punto y se le suma el bias
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        #si la sumaroria es mayor de cero, el threshold es alcanzado  y retorna 1. Se usa esta función de activación
        return 1 if summation > 0 else 0

    def train(self, training_data, labels):
        #se ejecuta para todas las epocas que se reciban como parametros
        for _ in range(self.epochs):
            #se concatenan los datos de entrada y los valores esperados para iterar sobre ellos
            for inputs, label in zip(training_data, labels):
                #se predice el valor
                prediction = self.predict(inputs)
                #se calcula el error de la predicción
                error = label - prediction
                #se recalculan los pesos basados en el error y los valores de la entrada
                self.weights[1:] += self.learning_rate * error * inputs
                #se calcula el peso del bias
                self.weights[0] += self.learning_rate * error

training_data=np.array([[4, 1],[2, 1],[3, 4],[5, 3],[2 ,3],[3, 5],[2, 1],[1, 1],[5, 1],[5, 2]])
classification = np.array([0,1,1,0,1,0,1,1,0,0])

perceptron=Perceptron(input_size=2)
perceptron.train(training_data, classification)
predictions=[]
for data in test_data:
    predictions.append(perceptron.predict(data))

plot_hyperplane(test_data, predictions, perceptron.weights[1:], perceptron.weights[0])

class PerceptronDesc:
    def __init__(self,  input_size,  learning_rate=0.01, epochs=100):
        self.epochs = epochs
        self.learning_rate = learning_rate
        #se suma 1 por la columna de bias que se agregara
        self.weights = np.random.randn(input_size+1) 

    def predict(self, weights, data ):
       #se hace la prediccion
       return np.dot(data, weights) > 0

    
    def _add_bias(self,training_data):
        #se inserta una columna de bias (w0), al inicio
        return np.insert(training_data, 0, np.ones(training_data.shape[0]), axis=1)
   
    def train(self, training_data, labels):
        #se crea la primera columna de bias 
        training_data = self._add_bias(training_data)
        for epoch in range(self.epochs):
            # Batch Gradient Descent
            y_hat = self.predict(self.weights, training_data)  
            # se calcula la funcion de perdida por minimos cuadrados (MSE)
            loss = 0.5*(y_hat - labels)**2
            # derivadas
            dldh = (y_hat - labels)
            dhdw = training_data
            dldweights = np.dot(dldh, dhdw)
            # actualizar pesos
            self.weights = self.weights - self.learning_rate*dldweights

training_data=np.array([ [4, 4], [4, 2], [1, 1], [3, 1], [2, 3], [1, 1], [3, 3], [1, 4], [3, 4], [1, 3] ])
classification = np.array([1,0,0,0,1,0,1,1,0,1])

perceptron2=PerceptronDesc(input_size=2)
perceptron2.train(training_data, classification)
test_data=np.array(random_coord(40))
#se saca una copia con los datos originales para plotearlos
test2=test_data.copy()
#se agrega la coumna de bias para poder hacer el producto puntod e la matriz
test_data=np.insert(test_data, 0, np.ones(test_data.shape[0]), axis=1)
predictions=[]
for data in test_data:
    predictions.append(perceptron2.predict(perceptron2.weights, data))

plot_hyperplane(test2, predictions, perceptron2.weights[1:], perceptron2.weights[0])
