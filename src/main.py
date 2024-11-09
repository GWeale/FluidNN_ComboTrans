import numpy as np
from src.model import CombinedModel
from src.utils import load_data, accuracy, save_results

def cross_entropy_loss(pred, true):
    return -np.mean(np.sum(true * np.log(pred + 1e-9), axis=1))

X_train, X_test, Y_train, Y_test = load_data('data/dataset.csv')
transformer_params = {
    'num_layers': 2,
    'd_model': 64,
    'num_heads': 8,
    'd_ff': 256,
    'd_state': 16,
    'input_dim': 64,
    'output_dim': Y_train.shape[1]
}
model = CombinedModel(input_size=X_train.shape[1], hidden_sizes=[128, 64], transformer_params=transformer_params, output_size=Y_train.shape[1], learning_rate=0.01)
model.train(X_train, Y_train, epochs=500, batch_size=64)
pred_train = model.forward(X_train)
pred_test = model.forward(X_test)
train_loss = cross_entropy_loss(pred_train, Y_train)
test_loss = cross_entropy_loss(pred_test, Y_test)
train_acc = accuracy(pred_train, Y_train)
test_acc = accuracy(pred_test, Y_test)
save_results(pred_test, 'results/predictions.csv')
print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
