"""Facies classifier using PyTorch CNN for lithology classification."""

import torch
import torch.nn as nn
import numpy as np
import os


class FaciesCNN(nn.Module):
    def __init__(self, in_features=6, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FaciesClassifier:
    def __init__(self, in_features=6, num_classes=5, lr=0.001, epochs=100, batch_size=64, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaciesCNN(in_features=in_features, num_classes=num_classes).to(self.device)
        self.name = "PyTorch CNN (FaciesClassifier)"
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes
        self._input_mean = None
        self._input_std = None

    def _to_tensor(self, X):
        X = np.array(X, dtype=np.float32)
        if self._input_mean is not None:
            X = (X - self._input_mean) / (self._input_std + 1e-10)
        return torch.FloatTensor(X).to(self.device)

    def train(self, X_train, y_train):
        self._input_mean = X_train.mean(axis=0)
        self._input_std = X_train.std(axis=0)

        X_t = self._to_tensor(X_train)
        y_t = torch.LongTensor(np.array(y_train, dtype=np.int64)).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X)
            logits = self.model(X_t)
            preds = logits.argmax(dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X)
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=1)
        return proba.cpu().numpy()

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        y_test = np.array(y_test)
        accuracy = float(np.mean(preds == y_test))
        f1_scores = []
        for c in range(self.num_classes):
            tp = np.sum((preds == c) & (y_test == c))
            fp = np.sum((preds == c) & (y_test != c))
            fn = np.sum((preds != c) & (y_test == c))
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)
        f1_macro = float(np.mean(f1_scores))
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "n_test": int(len(y_test)),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "input_mean": self._input_mean.tolist() if self._input_mean is not None else None,
            "input_std": self._input_std.tolist() if self._input_std is not None else None,
            "config": {"in_features": self.model.net[0].in_features, "num_classes": self.num_classes},
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        cfg = state["config"]
        self.model = FaciesCNN(in_features=cfg["in_features"], num_classes=cfg["num_classes"]).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.num_classes = cfg["num_classes"]
        if state["input_mean"] is not None:
            self._input_mean = np.array(state["input_mean"])
        if state["input_std"] is not None:
            self._input_std = np.array(state["input_std"])
        self.model.eval()
        return self
