import numpy as np

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