import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib as plt

nnfs.init()

np.random.seed(0)

X =  [[1, 2, 3, 2.5],
      [2.0,5.0,-1.0,2.0],
      [-1.5,2.7,3.3,-0.8]]

omega = X, y = spiral_data(100, 3)

print(f' The matrix is {omega}')

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

class Activation_ReLU: 
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        
        
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros((points*classes), dtype="uint8")
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number + 1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


    
    
        

'''
How ReLU works!

self.output = np.maximum(0,inputs)

more simple:
for i in inputs:
    if i>0:
        output.append(i)
    elif i <=0:
        output.append(0)

'''

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        

layer1 = Layer_Dense(2,5)  
#layer2 = Layer_Dense(5,2)
activation1 = Activation_ReLU()


layer1.forward(X)
activation1.forward(layer1.output)
print(f'this is the output from first layer! {layer1.output}')
#print(layer1.output)
#layer2.forward(layer1.output)
#print(layer2.output)
#print (len(layer2.output))