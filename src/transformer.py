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

class StateSpaceBlock:
    def __init__(self, d_model, d_state, dt_min=0.001, dt_max=0.1):
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters
        self.A = np.random.randn(d_state, d_state) * 0.1
        self.B = np.random.randn(d_state, d_model) * 0.1
        self.C = np.random.randn(d_model, d_state) * 0.1
        self.D = np.eye(d_model)
        
        # Selective scan parameters
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.delta_W = np.random.randn(d_model, d_model) * 0.1
        self.delta_b = np.zeros(d_model)
    
    def get_delta(self, x):
        # Compute dynamic timesteps
        delta = np.tanh(x.dot(self.delta_W) + self.delta_b)
        return self.dt_min + (self.dt_max - self.dt_min) * delta
    
    def selective_scan(self, x, h0=None):
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h0 = np.zeros((batch_size, self.d_state))
        
        h = h0
        outputs = []
        
        for t in range(seq_len):
            # Compute dynamic timestep
            dt = self.get_delta(x[:, t])
            
            # Discretize continuous state space equation
            A_dt = np.eye(self.d_state) + dt[:, None, None] * self.A
            B_dt = dt[:, None, None] * self.B
            
            # Update state
            h = h.dot(A_dt) + x[:, t].dot(B_dt)
            
            # Compute output
            y = h.dot(self.C.T) + x[:, t].dot(self.D)
            outputs.append(y)
        
        return np.stack(outputs, axis=1)
    
    def forward(self, x):
        return self.selective_scan(x)

class HybridTransformerLayer:
    def __init__(self, d_model, num_heads, d_ff, d_state):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ssm = StateSpaceBlock(d_model, d_state)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        self.layernorm3 = LayerNormalization(d_model)
        
    def forward(self, x):
        # Multi-head attention path
        attn_output = self.mha.forward(x, x, x)
        out1 = self.layernorm1.forward(x + attn_output)
        
        # State space path
        ssm_output = self.ssm.forward(out1)
        out2 = self.layernorm2.forward(out1 + ssm_output)
        
        # Feed-forward path
        ffn_output = self.ffn.forward(out2)
        out3 = self.layernorm3.forward(out2 + ffn_output)
        
        return out3

class HybridTransformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, d_state, input_dim, output_dim):
        self.layers = [
            HybridTransformerLayer(d_model, num_heads, d_ff, d_state) 
            for _ in range(num_layers)
        ]
        self.linear = np.random.randn(d_model, output_dim) * np.sqrt(2. / d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x.dot(self.linear)
