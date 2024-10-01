import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1).values
    Y = df['target'].values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(Y)
    return train_test_split(X, Y, test_size=0.2, random_state=42)

def accuracy(pred, true):
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(true, axis=1)
    return np.mean(pred_labels == true_labels)

def save_results(predictions, filepath):
    df = pd.DataFrame(predictions)
    df.to_csv(filepath, index=False)
