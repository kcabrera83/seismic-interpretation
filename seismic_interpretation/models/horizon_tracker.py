"""Horizon tracker using PyTorch CNN for porosity regression."""

import torch
import torch.nn as nn
import numpy as np
import os
import json


class HorizonCNN(nn.Module):
    def __init__(self, in_features=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HorizonTracker:
    def __init__(self, in_features=6, lr=0.001, epochs=100, batch_size=64, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HorizonCNN(in_features=in_features).to(self.device)
        self.name = "PyTorch CNN (HorizonTracker)"
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
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
        y_t = torch.FloatTensor(np.array(y_train, dtype=np.float32)).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = self._to_tensor(X)
            preds = self.model(X_t)
        return preds.cpu().numpy()

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        y_test = np.array(y_test, dtype=np.float64)
        mse = float(np.mean((y_test - preds) ** 2))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(y_test - preds))),
            "r2": float(1 - np.sum((y_test - preds) ** 2) / np.sum((y_test - y_test.mean()) ** 2)),
            "n_test": int(len(y_test)),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "input_mean": self._input_mean.tolist() if self._input_mean is not None else None,
            "input_std": self._input_std.tolist() if self._input_std is not None else None,
            "config": {"in_features": self.model.net[0].in_features},
        }
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        cfg = state["config"]
        self.model = HorizonCNN(in_features=cfg["in_features"]).to(self.device)
        self.model.load_state_dict(state["model_state"])
        if state["input_mean"] is not None:
            self._input_mean = np.array(state["input_mean"])
        if state["input_std"] is not None:
            self._input_std = np.array(state["input_std"])
        self.model.eval()
        return self
