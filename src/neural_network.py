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

class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps
        
    def forward(self, x):
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad_output):
        return grad_output * self.gamma

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        
    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1-self.p, x.shape) / (1-self.p)
            return x * self.mask
        return x
    
    def backward(self, grad_output):
        return grad_output * self.mask

class GELU:
    def forward(self, x):
        self.x = x
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def backward(self, grad_output):
        return grad_output * (0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (self.x + 0.044715 * self.x**3))) + 
                            self.x * 0.5 * (1 - np.tanh(np.sqrt(2/np.pi) * (self.x + 0.044715 * self.x**3))**2) * 
                            np.sqrt(2/np.pi) * (1 + 0.134145 * self.x**2))

class ResidualBlock:
    def __init__(self, features):
        self.linear1 = Linear(features, features)
        self.linear2 = Linear(features, features)
        self.norm1 = LayerNorm(features)
        self.norm2 = LayerNorm(features)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)
        self.activation = GELU()
    
    def forward(self, x):
        self.input = x
        out = self.norm1.forward(x)
        out = self.linear1.forward(out)
        out = self.activation.forward(out)
        out = self.dropout1.forward(out)
        out = self.norm2.forward(out)
        out = self.linear2.forward(out)
        out = self.dropout2.forward(out)
        return out + self.input
    
    def backward(self, grad_output):
        grad = self.dropout2.backward(grad_output)
        grad = self.linear2.backward(self.norm2.x_norm, grad)
        grad = self.norm2.backward(grad)
        grad = self.dropout1.backward(grad)
        grad = self.activation.backward(grad)
        grad = self.linear1.backward(self.norm1.x_norm, grad)
        grad = self.norm1.backward(grad)
        return grad + grad_output

class FocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        pt = predictions * targets + (1 - predictions) * (1 - targets)
        self.pt = pt
        focal_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight *= (1 - pt) ** self.gamma
        return -np.mean(focal_weight * np.log(predictions + 1e-9))
    
    def backward(self):
        return (self.predictions - self.targets) * (
            self.alpha * self.targets + (1 - self.alpha) * (1 - self.targets)
        ) * (1 - self.pt) ** (self.gamma - 1)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.layers = []
        self.training = True
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes)-1):
            # Add residual block if input and output dimensions match
            if i > 0 and layer_sizes[i] == layer_sizes[i-1]:
                self.layers.append(ResidualBlock(layer_sizes[i]))
            
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.layers.append(LayerNorm(layer_sizes[i+1]))
                self.layers.append(GELU())
                self.layers.append(Dropout(0.2))
        
        self.layers.append(Softmax())
        self.loss_fn = FocalLoss()
        self.initial_lr = learning_rate
        self.current_step = 0
    
    def get_learning_rate(self):
        # Cosine learning rate schedule with warm-up
        warmup_steps = 1000
        if self.current_step < warmup_steps:
            return self.initial_lr * self.current_step / warmup_steps
        else:
            return self.initial_lr * 0.5 * (1 + np.cos(
                np.pi * (self.current_step - warmup_steps) / (10000 - warmup_steps)
            ))
    
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
                layer.update(self.get_learning_rate())
    
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
