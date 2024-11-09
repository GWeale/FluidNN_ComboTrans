import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.special import softmax

class DataProcessor:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.encoder = OneHotEncoder(sparse=False)
        self.scaler = StandardScaler()
        
    def process_data(self, filepath):
        df = pd.read_csv(filepath)
        X = self._preprocess_features(df)
        y = self._preprocess_targets(df)
        return self._create_cross_validation_splits(X, y)
    
    def _preprocess_features(self, df):
        X = df.drop('target', axis=1)
        # Add feature engineering
        X['feature_interactions'] = X.iloc[:, 0] * X.iloc[:, 1]
        X['feature_squares'] = X.iloc[:, 0] ** 2
        X = self.scaler.fit_transform(X)
        return X
    
    def _preprocess_targets(self, df):
        y = df['target'].values.reshape(-1, 1)
        return self.encoder.fit_transform(y)
    
    def _create_cross_validation_splits(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        return [(X[train_idx], X[val_idx], y[train_idx], y[val_idx]) 
                for train_idx, val_idx in skf.split(X, np.argmax(y, axis=1))]

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'f1_scores': [], 'confusion_matrices': []
        }
    
    def update(self, y_true, y_pred, phase='train'):
        pred_probs = softmax(y_pred, axis=1)
        loss = self._calculate_loss(y_true, pred_probs)
        acc = self._calculate_accuracy(y_true, pred_probs)
        f1 = self._calculate_f1_score(y_true, pred_probs)
        conf_matrix = self._calculate_confusion_matrix(y_true, pred_probs)
        
        self.metrics[f'{phase}_loss'].append(loss)
        self.metrics[f'{phase}_acc'].append(acc)
        self.metrics['f1_scores'].append(f1)
        self.metrics['confusion_matrices'].append(conf_matrix)
        
        return loss, acc, f1
    
    def _calculate_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
    
    def _calculate_accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
    
    def _calculate_f1_score(self, y_true, y_pred):
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        
        f1_scores = []
        for class_idx in range(y_true.shape[1]):
            tp = np.sum((pred_labels == class_idx) & (true_labels == class_idx))
            fp = np.sum((pred_labels == class_idx) & (true_labels != class_idx))
            fn = np.sum((pred_labels != class_idx) & (true_labels == class_idx))
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
            f1_scores.append(f1)
            
        return np.mean(f1_scores)
    
    def _calculate_confusion_matrix(self, y_true, y_pred):
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        n_classes = y_true.shape[1]
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for i in range(len(true_labels)):
            conf_matrix[true_labels[i]][pred_labels[i]] += 1
            
        return conf_matrix
