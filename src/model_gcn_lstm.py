import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        # Simulate GCN adjacency via simple averaging (since we don’t have graph edges)
        adj_out = x.mean(dim=1, keepdim=True).expand_as(x)
        out = self.linear(x + adj_out)
        return F.relu(out)

class GCN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(GCN_LSTM, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        x = self.gcn1(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out
