import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.Wq = np.random.randn(d_model, d_model) * np.sqrt(2. / d_model)
        self.Wk = np.random.randn(d_model, d_model) * np.sqrt(2. / d_model)
        self.Wv = np.random.randn(d_model, d_model) * np.sqrt(2. / d_model)
        self.Wo = np.random.randn(d_model, d_model) * np.sqrt(2. / d_model)
    
    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0,2,1,3)
    
    def scaled_dot_product_attention(self, Q, K, V):
        matmul_qk = Q.dot(K.transpose(0,1,3,2))
        dk = K.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        attention_weights = self.softmax(scaled_attention_logits)
        output = attention_weights.dot(V)
        return output
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    def forward(self, v, k, q):
        batch_size = q.shape[0]
        Q = q.dot(self.Wq)
        K = k.dot(self.Wk)
        V = v.dot(self.Wv)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        scaled_attention = self.scaled_dot_product_attention(Q, K, V)
        scaled_attention = scaled_attention.transpose(0,2,1,3).reshape(batch_size, -1, self.d_model)
        output = scaled_attention.dot(self.Wo)
        return output

class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2. / d_model)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2. / d_ff)
    
    def forward(self, x):
        return self.relu(x.dot(self.W1)).dot(self.W2)
    
    def relu(self, x):
        return np.maximum(0, x)

class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones((1, d_model))
        self.beta = np.zeros((1, d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta

class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
    
    def forward(self, x):
        attn_output = self.mha.forward(x, x, x)
        out1 = self.layernorm1.forward(x + attn_output)
        ffn_output = self.ffn.forward(out1)
        out2 = self.layernorm2.forward(out1 + ffn_output)
        return out2

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_dim, output_dim):
        self.layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.linear = np.random.randn(d_model, output_dim) * np.sqrt(2. / d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x.dot(self.linear)
