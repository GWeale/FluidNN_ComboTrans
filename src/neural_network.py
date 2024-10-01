import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))
    
    def forward(self, x):
        return x.dot(self.W) + self.b
    
    def backward(self, x, grad_output):
        self.grad_W = x.T.dot(grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output.dot(self.W.T)
    
    def update(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    
    def backward(self, grad_output):
        return grad_output * self.mask

class Softmax:
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out
    
    def backward(self, grad_output):
        return grad_output

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return -np.mean(np.sum(targets * np.log(predictions + 1e-9), axis=1))
    
    def backward(self):
        return (self.predictions - self.targets) / self.predictions.shape[0]

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes)-1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.layers.append(ReLU())
        self.layers.append(Softmax())
        self.loss_fn = CrossEntropyLoss()
        self.lr = learning_rate
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def update(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.update(self.lr)
    
    def train(self, X, Y, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                predictions = self.forward(X_batch)
                loss = self.loss_fn.forward(predictions, Y_batch)
                grad = self.loss_fn.backward()
                self.backward(grad)
                self.update()
