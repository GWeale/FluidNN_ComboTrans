import numpy as np
from src.neural_network import NeuralNetwork
from src.transformer import Transformer

class CombinedModel:
    def __init__(self, input_size, hidden_sizes, transformer_params, output_size, learning_rate=0.001):
        self.nn = NeuralNetwork(input_size, hidden_sizes, transformer_params['d_model'], learning_rate)
        self.transformer = Transformer(**transformer_params)
    
    def forward(self, x):
        x_nn = self.nn.forward(x)
        x_trans = self.transformer.forward(x_nn.reshape(x_nn.shape[0], 1, -1))
        return x_trans.reshape(x_trans.shape[0], -1)
    
    def backward(self, grad):
        grad_trans = grad.reshape(grad.shape[0], 1, -1).dot(self.transformer.linear.T)
        self.transformer.backward(grad_trans)
        grad_nn = self.nn.backward(grad_trans.reshape(grad_trans.shape[0], -1))
        return grad_nn
    
    def update(self):
        self.nn.update()
        self.transformer.update()
    
    def train(self, X, Y, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                predictions = self.forward(X_batch)
                loss = self.nn.loss_fn.forward(predictions, Y_batch)
                grad = self.nn.loss_fn.backward()
                self.backward(grad)
                self.update()
