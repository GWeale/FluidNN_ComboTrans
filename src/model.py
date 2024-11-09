import numpy as np
from src.neural_network import NeuralNetwork
from src.transformer import Transformer, StateSpaceBlock

class AdaptiveOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {param: np.zeros_like(value) for param, value in params.items()}
        self.v = {param: np.zeros_like(value) for param, value in params.items()}
        self.t = 0
        
    def step(self):
        self.t += 1
        for param, value in self.params.items():
            grad = value.grad
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * grad**2
            
            m_hat = self.m[param] / (1 - self.beta1**self.t)
            v_hat = self.v[param] / (1 - self.beta2**self.t)
            
            value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class CombinedModel:
    def __init__(self, input_size, hidden_sizes, transformer_params, output_size, learning_rate=0.001):
        self.nn = NeuralNetwork(input_size, hidden_sizes, transformer_params['d_model'], learning_rate)
        self.transformer = Transformer(**transformer_params)
        self.ssm = StateSpaceBlock(transformer_params['d_model'], transformer_params['d_state'])
        
        self.dropout_rate = 0.1
        self.training = True
        self.optimizer = AdaptiveOptimizer(self._get_parameters(), lr=learning_rate)
        
    def _get_parameters(self):
        params = {}
        # Collect all trainable parameters
        for i, layer in enumerate(self.nn.layers):
            params[f'nn_layer_{i}'] = layer
        params['transformer'] = self.transformer
        params['ssm'] = self.ssm
        return params
        
    def forward(self, x, return_attention=False):
        # Neural network path
        x_nn = self.nn.forward(x)
        
        # Apply dropout during training
        if self.training:
            mask = np.random.binomial(1, 1-self.dropout_rate, x_nn.shape) / (1-self.dropout_rate)
            x_nn *= mask
            
        # Reshape for transformer
        x_trans = x_nn.reshape(x_nn.shape[0], 1, -1)
        
        # Parallel processing through transformer and SSM
        trans_out = self.transformer.forward(x_trans)
        ssm_out = self.ssm.forward(x_trans)
        
        # Combine outputs with learned weights
        alpha = np.random.rand()  # This should be a learned parameter
        combined = alpha * trans_out + (1 - alpha) * ssm_out
        
        output = combined.reshape(combined.shape[0], -1)
        
        if return_attention:
            return output, self.transformer.get_attention_weights()
        return output
    
    def train(self, X, Y, epochs=1000, batch_size=32, validation_data=None):
        self.training = True
        metrics_tracker = MetricsTracker()
        
        for epoch in range(epochs):
            self._train_epoch(X, Y, batch_size, metrics_tracker)
            
            if validation_data is not None:
                self._validate(validation_data[0], validation_data[1], metrics_tracker)
                
            if epoch % 10 == 0:
                self._print_metrics(epoch, metrics_tracker)
    
    def _train_epoch(self, X, Y, batch_size, metrics_tracker):
        indices = np.random.permutation(len(X))
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]
            
            predictions = self.forward(X_batch)
            loss = self.nn.loss_fn.forward(predictions, Y_batch)
            grad = self.nn.loss_fn.backward()
            
            self.backward(grad)
            self.optimizer.step()
            
            metrics_tracker.update(Y_batch, predictions, phase='train')
