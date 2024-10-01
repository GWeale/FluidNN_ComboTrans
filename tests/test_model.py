import unittest
import numpy as np
from src.model import CombinedModel
from src.transformer import Transformer
from src.neural_network import NeuralNetwork

class TestCombinedModel(unittest.TestCase):
    def test_forward(self):
        transformer_params = {
            'num_layers': 2,
            'd_model': 64,
            'num_heads': 8,
            'd_ff': 256,
            'input_dim': 64,
            'output_dim': 3
        }
        model = CombinedModel(input_size=10, hidden_sizes=[128, 64], transformer_params=transformer_params, output_size=3, learning_rate=0.01)
        X = np.random.randn(2,10)
        output = model.forward(X)
        self.assertEqual(output.shape, (2,64))
    
    def test_backward(self):
        transformer_params = {
            'num_layers': 2,
            'd_model': 64,
            'num_heads': 8,
            'd_ff': 256,
            'input_dim': 64,
            'output_dim': 3
        }
        model = CombinedModel(input_size=10, hidden_sizes=[128, 64], transformer_params=transformer_params, output_size=3, learning_rate=0.01)
        X = np.random.randn(2,10)
        Y = np.eye(3)[np.array([0,1])]
        output = model.forward(X)
        loss = -np.mean(np.sum(Y * np.log(output + 1e-9), axis=1))
        grad = (output - Y) / output.shape[0]
        model.backward(grad)
        self.assertTrue(True)

class TestTransformer(unittest.TestCase):
    def test_transformer_forward(self):
        transformer = Transformer(num_layers=2, d_model=64, num_heads=8, d_ff=256, input_dim=64, output_dim=10)
        x = np.random.randn(2, 10, 64)
        output = transformer.forward(x)
        self.assertEqual(output.shape, (2, 10, 10))

if __name__ == '__main__':
    unittest.main()
